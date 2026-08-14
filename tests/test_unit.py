"""Unit tests for core modules"""
import asyncio
import torch
import pytest
from unittest.mock import AsyncMock, Mock, patch


class TestSchedulerRequestQueue:
    """Unit tests for the request queue"""

    def test_import_request_queue(self):
        """Test that request queue can be imported"""
        try:
            from scheduler.request_queue import request_queue
            assert request_queue is not None
        except ImportError:
            pytest.skip("Scheduler module not available")

    def test_import_inference_request(self):
        """Test that InferenceRequest can be imported"""
        try:
            from scheduler.request import InferenceRequest
            assert InferenceRequest is not None
        except ImportError:
            pytest.skip("Scheduler module not available")

    def test_inference_request_stop_sequences_default_and_passthrough(self):
        """`stop_sequences` defaults to an empty list, and a passed-in list is kept as-is."""
        try:
            from scheduler.request import InferenceRequest
        except ImportError:
            pytest.skip("Scheduler module not available")

        no_stop = InferenceRequest(prompt="hi", max_tokens=5, temperature=0.5)
        assert no_stop.stop_sequences == []

        with_stop = InferenceRequest(prompt="hi", max_tokens=5, temperature=0.5, stop_sequences=["\nuser:"])
        assert with_stop.stop_sequences == ["\nuser:"]


class TestStopSequences:
    """Unit tests for the shared stop-sequence matching helper."""

    def test_find_stop_index_returns_earliest_match(self):
        try:
            from utils.stop_sequences import find_stop_index
        except ImportError:
            pytest.skip("utils.stop_sequences not available")

        text = "hello user: goodbye assistant:"
        assert find_stop_index(text, ["assistant:", "user:"]) == text.index("user:")

    def test_find_stop_index_no_match_returns_none(self):
        try:
            from utils.stop_sequences import find_stop_index
        except ImportError:
            pytest.skip("utils.stop_sequences not available")

        assert find_stop_index("hello world", ["user:"]) is None

    def test_find_stop_index_ignores_empty_strings(self):
        try:
            from utils.stop_sequences import find_stop_index
        except ImportError:
            pytest.skip("utils.stop_sequences not available")

        assert find_stop_index("hello", ["", "world"]) is None


class TestGenerateRequestSchema:
    """Unit tests for the `stop` field on `GenerateRequest`."""

    def test_stop_field_defaults_to_none(self):
        from schemas.schemas import GenerateRequest

        req = GenerateRequest(prompt="hi")
        assert req.stop is None

    def test_stop_field_accepts_list_of_strings(self):
        from schemas.schemas import GenerateRequest

        req = GenerateRequest(prompt="hi", stop=["\nuser:", "\nUser:"])
        assert req.stop == ["\nuser:", "\nUser:"]

    def test_stop_field_rejects_empty_string(self):
        from pydantic import ValidationError
        from schemas.schemas import GenerateRequest

        with pytest.raises(ValidationError):
            GenerateRequest(prompt="hi", stop=[""])

    def test_stop_field_rejects_more_than_four(self):
        from pydantic import ValidationError
        from schemas.schemas import GenerateRequest

        with pytest.raises(ValidationError):
            GenerateRequest(prompt="hi", stop=["a", "b", "c", "d", "e"])


class TestForwardStepNoGrad:
    """Regression test for a real memory leak: forward_step() must run under
    torch.no_grad(). Without it, every forward pass builds a full autograd
    graph (activations retained across every layer) that nothing ever calls
    .backward() on or otherwise releases -- a steady per-call memory leak
    that torch.cuda/mps.empty_cache() cannot reclaim, since the memory is
    genuinely referenced (by the graph), not just cached. Confirmed via a
    live-memory diagnostic against the real model: MPS memory grew by a
    fixed amount on every single call without this fix, and was perfectly
    flat across 20+ calls with it."""

    def test_forward_step_calls_model_inside_no_grad(self):
        try:
            from engine.generator import InferenceEngine
            from unittest.mock import MagicMock
        except ImportError:
            pytest.skip("InferenceEngine not available")

        engine = InferenceEngine()
        grad_enabled_during_call = []

        class FakeOutputs:
            def __init__(self):
                # requires_grad=True mimics what a real model's output would
                # be *if* the forward call weren't inside no_grad -- so this
                # also exercises the assertion below on the returned tensor.
                self.logits = torch.zeros(1, 3, 10, requires_grad=True)
                self.past_key_values = MagicMock()

        def fake_model(**kwargs):
            grad_enabled_during_call.append(torch.is_grad_enabled())
            return FakeOutputs()

        mock_model = MagicMock(side_effect=fake_model)
        mock_model.device = "cpu"
        mock_model.dtype = torch.float32
        engine._model = mock_model

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones(1, 3, dtype=torch.long)

        logits, _ = engine.forward_step(input_ids, attention_mask)

        assert grad_enabled_during_call == [False], (
            "forward_step() must call the model inside torch.no_grad()"
        )
        assert logits.requires_grad is False, (
            "Output must not carry autograd tracking when computed under no_grad()"
        )


class TestBatchScheduler:
    """Unit tests for batch scheduling and latency metrics"""

    @pytest.mark.asyncio
    async def test_batch_scheduler_processes_active_requests(self):
        try:
            from scheduler.batch_scheduler import BatchScheduler
            from scheduler.request import InferenceRequest
            from metrics.metrics import metrics
        except ImportError:
            pytest.skip("Batch scheduler module not available")

        metrics.queue_latencies.clear()
        metrics.batch_sizes.clear()
        metrics.token_throughputs.clear()

        mock_engine = Mock()
        mock_engine.generate_batch = AsyncMock(return_value=["first-output", "second-output"])

        mock_tokenizer = Mock()
        mock_tokenizer.tokenizer = Mock(return_value={
            "input_ids": torch.tensor([[1, 2], [1, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
        })

        scheduler = BatchScheduler(mock_engine, mock_tokenizer, max_batch_size=2, queue_timeout=0.01)
        requests = [
            InferenceRequest(prompt="first prompt", max_tokens=2, temperature=0.7),
            InferenceRequest(prompt="second prompt", max_tokens=2, temperature=0.9),
        ]

        await scheduler.process_batch(requests)

        assert requests[0].future.done()
        assert requests[0].future.result() == "first-output"
        assert requests[1].future.done()
        assert requests[1].future.result() == "second-output"
        assert metrics.batch_sizes[-1] == 2
        assert metrics.token_throughputs[-1] >= 0
        mock_engine.generate_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_batch_compaction_in_inference_engine(self):
        """Test that batch compaction updates active_requests and handles early completion."""
        try:
            from engine.generator import InferenceEngine
            from scheduler.request import InferenceRequest
            from unittest.mock import MagicMock
        except ImportError:
            pytest.skip("InferenceEngine or InferenceRequest not available")

        # Initialize engine
        engine = InferenceEngine()

        # Mock the model and its configuration
        mock_model = MagicMock()
        mock_model.config.eos_token_id = 50256  # GPT-2 EOS token
        engine._model = mock_model

        # Setup requests with unequal max_tokens
        req_0 = InferenceRequest(prompt="short", max_tokens=1, temperature=0.7)
        req_1 = InferenceRequest(prompt="longer", max_tokens=3, temperature=0.9)
        requests = [req_0, req_1]

        # Prepare input_ids and attention_mask tensors
        input_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1], [1, 1]], dtype=torch.long)

        class MockModelOutput:
            def __init__(self, logits, past_key_values):
                self.logits = logits
                self.past_key_values = past_key_values

        mock_pkv = MagicMock()

        # Mock the model calls to simulate compaction:
        # Step 1: 2 requests active. Request 0 generates EOS (50256), Request 1 generates 100.
        # Step 2: 1 request active (Request 1). Generates 101.
        # Step 3: 1 request active (Request 1). Generates 102 (finishes max_tokens=3).
        call_count = 0
        def model_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                logits = torch.zeros((2, 1, 50257))
                logits[0, 0, 50256] = 100.0  # Force EOS for req_0
                logits[1, 0, 100] = 100.0    # Token 100 for req_1
                return MockModelOutput(logits, mock_pkv)
            elif call_count == 2:
                logits = torch.zeros((1, 1, 50257))
                logits[0, 0, 101] = 100.0    # Token 101 for req_1
                return MockModelOutput(logits, mock_pkv)
            else:
                logits = torch.zeros((1, 1, 50257))
                logits[0, 0, 102] = 100.0    # Token 102 for req_1
                return MockModelOutput(logits, mock_pkv)

        mock_model.side_effect = model_side_effect

        outputs = await engine.generate_batch(input_ids, attention_mask, requests)

        from tokenizer.tokenizer_service import tokenizer_service
        assert len(outputs) == 2
        assert outputs[0] == tokenizer_service.decode([50256])
        assert outputs[1] == tokenizer_service.decode([100, 101, 102])
        assert call_count == 3
        mock_pkv.batch_select_indices.assert_called_once()

        # Regression guard: generate_batch must mutate the InferenceRequest
        # objects directly (not a wrapper), or downstream token-throughput
        # metrics silently read an always-empty generated_tokens list.
        assert req_0.generated_tokens == [50256]
        assert req_1.generated_tokens == [100, 101, 102]
        assert req_0.finished is True
        assert req_1.finished is True


class TestModelSwap:
    """Unit tests for runtime model hot-swapping (scheduler/model_swap.py)."""

    def _drain(self, request_q):
        while not request_q.empty():
            request_q.queue.get_nowait()

    def test_model_loader_reload_keeps_old_model_on_failure(self):
        """If the new model fails to load, the previously-loaded model must stay active."""
        try:
            from engine.model_loader import ModelLoader
        except ImportError:
            pytest.skip("ModelLoader not available")

        loader = ModelLoader()
        old_model = Mock(name="old-model")
        old_model.to.return_value = old_model

        with patch("engine.model_loader.AutoModelForCausalLM") as mock_cls:
            mock_cls.from_pretrained.return_value = old_model
            loader.load()
        assert loader.model is old_model

        with patch("engine.model_loader.AutoModelForCausalLM") as mock_cls:
            mock_cls.from_pretrained.side_effect = RuntimeError("bad repo id")
            with pytest.raises(RuntimeError):
                loader.reload("some/bad-model")

        assert loader.model is old_model

    @pytest.mark.asyncio
    async def test_swap_model_waits_for_active_requests_then_swaps(self):
        """The swap must not touch the model/tokenizer until active_requests is empty."""
        try:
            from scheduler.model_swap import swap_model
            from scheduler.request_queue import request_queue, batch_request_queue
        except ImportError:
            pytest.skip("Required modules not available")

        self._drain(request_queue)
        self._drain(batch_request_queue)

        fake_scheduler = Mock()
        fake_scheduler.active_requests = ["still-running"]

        with patch("scheduler.model_swap.tokenizer_service") as mock_tok, \
             patch("scheduler.model_swap.model_loader") as mock_loader, \
             patch("scheduler.model_swap.engine") as mock_engine:

            task = asyncio.create_task(swap_model("new/model", fake_scheduler, drain_timeout=2.0))
            await asyncio.sleep(0.05)
            assert not task.done(), "swap_model must wait while active_requests is non-empty"

            fake_scheduler.active_requests = []
            result = await task

        assert result == "new/model"
        mock_tok.reload.assert_called_once_with("new/model")
        mock_loader.reload.assert_called_once_with("new/model")
        mock_engine.invalidate_model_cache.assert_called_once()
        fake_scheduler.invalidate_paged_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_swap_model_times_out_when_requests_never_drain(self):
        try:
            from scheduler.model_swap import swap_model
        except ImportError:
            pytest.skip("Required modules not available")

        fake_scheduler = Mock()
        fake_scheduler.active_requests = ["stuck-forever"]

        with pytest.raises(TimeoutError):
            await swap_model("new/model", fake_scheduler, drain_timeout=0.1)

    @pytest.mark.asyncio
    async def test_swap_model_rolls_back_tokenizer_on_model_reload_failure(self):
        """A failed model reload must not leave the tokenizer paired with a model
        that was never actually loaded."""
        try:
            from scheduler.model_swap import swap_model
            from scheduler.request_queue import request_queue, batch_request_queue
            from settings.settings import model_settings
        except ImportError:
            pytest.skip("Required modules not available")

        self._drain(request_queue)
        self._drain(batch_request_queue)

        fake_scheduler = Mock()
        fake_scheduler.active_requests = []
        previous_name = model_settings.model_name

        with patch("scheduler.model_swap.tokenizer_service") as mock_tok, \
             patch("scheduler.model_swap.model_loader") as mock_loader, \
             patch("scheduler.model_swap.engine") as mock_engine:

            mock_loader.reload.side_effect = RuntimeError("bad repo id")

            with pytest.raises(RuntimeError):
                await swap_model("bad/model", fake_scheduler, drain_timeout=1.0)

        mock_tok.reload.assert_any_call("bad/model")
        mock_tok.reload.assert_any_call(previous_name)
        mock_engine.invalidate_model_cache.assert_not_called()
        fake_scheduler.invalidate_paged_cache.assert_not_called()


class TestRepetitionPenalty:
    """Unit tests guarding against repetition penalty silently becoming a no-op."""

    def test_apply_repetition_penalty_accepts_ragged_per_row_histories(self):
        """`apply_repetition_penalty` must accept a list of per-row 1D tensors of
        different lengths, not just a dense same-length-per-row tensor -- the
        continuous scheduler's per-request histories are never uniform length
        across a mixed prefill/decode batch."""
        try:
            from engine.generator import InferenceEngine
        except ImportError:
            pytest.skip("InferenceEngine not available")

        engine = InferenceEngine()
        vocab_size = 6
        logits = torch.zeros(2, vocab_size)

        # Row 0's history is longer than row 1's -- a dense tensor couldn't
        # represent this without padding (which would corrupt torch.unique).
        row_histories = [torch.tensor([1, 2, 3, 1]), torch.tensor([4])]

        result = engine.apply_repetition_penalty(logits, row_histories, penalty=2.0)

        # Positive logits (all start at 0, treated as >= 0) get divided by
        # penalty for every token that appeared in that row's own history.
        assert result[0, 1] == pytest.approx(0.0 / 2.0)
        assert result[0, 2] == pytest.approx(0.0 / 2.0)
        assert result[0, 3] == pytest.approx(0.0 / 2.0)
        assert result[0, 4] == 0.0, "Token 4 never appeared in row 0's history"
        assert result[1, 4] == pytest.approx(0.0 / 2.0)
        assert result[1, 1] == 0.0, "Token 1 never appeared in row 1's history"

    def test_forward_and_sample_penalizes_against_full_history_not_new_tokens_only(self):
        """Regression test: `_forward_and_sample` must pass each request's FULL
        generated history (prompt + everything generated so far) into repetition
        penalty, not `batch_inputs.input_ids` (this step's new-tokens-only
        slice -- a single token during decode). Passing just the new-tokens
        slice makes the penalty a no-op after the first decode step, since
        torch.unique() on one token has nothing left to penalize -- letting the
        model repeat/ramble without the configured penalty ever pushing back."""
        try:
            from scheduler.continuous_scheduler import ContinuousScheduler
            from scheduler.request import InferenceRequest
        except ImportError:
            pytest.skip("Required modules not available")

        mock_engine = Mock()
        mock_engine.device = "cpu"
        mock_engine.forward_step = Mock(return_value=(torch.zeros(1, 5), None))
        mock_engine.apply_repetition_penalty = Mock(side_effect=lambda logits, input_ids: logits)
        mock_engine.sample = Mock(return_value=torch.tensor([[1]]))

        scheduler = ContinuousScheduler(mock_engine, Mock(), max_batch_size=1, timeout=0.01)
        req = InferenceRequest(prompt="hi", max_tokens=5, temperature=0.0)
        # Full history (prompt + generated so far) is longer than a single
        # new-tokens-this-step slice would be.
        req.input_ids = torch.tensor([[10, 11, 12, 13, 14]])
        scheduler.active_requests = [req]

        batch_inputs = Mock(
            input_ids=torch.tensor([[14]]),  # this step's new-tokens-only slice
            attention_mask=torch.ones(1, 5),
            past_key_values=None,
            position_ids=torch.tensor([[4]]),
            logit_gather_indices=torch.tensor([0]),
        )

        scheduler._forward_and_sample(batch_inputs, scheduler.active_requests)

        mock_engine.apply_repetition_penalty.assert_called_once()
        _, passed_input_ids = mock_engine.apply_repetition_penalty.call_args[0]
        assert len(passed_input_ids) == 1
        assert torch.equal(passed_input_ids[0], req.input_ids[0]), (
            "Expected the request's full input_ids history, not batch_inputs.input_ids"
        )
        assert passed_input_ids[0].numel() == 5, "Full history has 5 tokens; the new-tokens-only slice has 1"


class TestDeviceMemoryPressure:
    """Unit tests for utils/device_cache.py's proactive, usage-based clearing."""

    def test_device_memory_pressure_uses_mps_apis_when_available(self):
        try:
            from utils.device_cache import device_memory_pressure
        except ImportError:
            pytest.skip("utils.device_cache not available")

        with patch("utils.device_cache.torch.backends.mps.is_available", return_value=True), \
             patch("utils.device_cache.torch.mps.recommended_max_memory", return_value=100), \
             patch("utils.device_cache.torch.mps.driver_allocated_memory", return_value=75):
            assert device_memory_pressure("mps") == pytest.approx(0.75)

    def test_device_memory_pressure_returns_none_for_cpu(self):
        try:
            from utils.device_cache import device_memory_pressure
        except ImportError:
            pytest.skip("utils.device_cache not available")

        assert device_memory_pressure("cpu") is None

    def test_device_memory_pressure_returns_none_on_error(self):
        try:
            from utils.device_cache import device_memory_pressure
        except ImportError:
            pytest.skip("utils.device_cache not available")

        with patch("utils.device_cache.torch.backends.mps.is_available", return_value=True), \
             patch("utils.device_cache.torch.mps.recommended_max_memory", side_effect=RuntimeError("boom")):
            assert device_memory_pressure("mps") is None

    def test_maybe_empty_device_cache_clears_above_threshold(self):
        try:
            from utils.device_cache import maybe_empty_device_cache
        except ImportError:
            pytest.skip("utils.device_cache not available")

        with patch("utils.device_cache.device_memory_pressure", return_value=0.9), \
             patch("utils.device_cache.empty_device_cache") as mock_clear:
            cleared = maybe_empty_device_cache("mps", threshold=0.7)

        assert cleared is True
        mock_clear.assert_called_once_with("mps")

    def test_maybe_empty_device_cache_leaves_low_pressure_alone(self):
        try:
            from utils.device_cache import maybe_empty_device_cache
        except ImportError:
            pytest.skip("utils.device_cache not available")

        with patch("utils.device_cache.device_memory_pressure", return_value=0.2), \
             patch("utils.device_cache.empty_device_cache") as mock_clear:
            cleared = maybe_empty_device_cache("mps", threshold=0.7)

        assert cleared is False
        mock_clear.assert_not_called()

    def test_maybe_empty_device_cache_leaves_unmeasurable_devices_alone(self):
        try:
            from utils.device_cache import maybe_empty_device_cache
        except ImportError:
            pytest.skip("utils.device_cache not available")

        with patch("utils.device_cache.device_memory_pressure", return_value=None), \
             patch("utils.device_cache.empty_device_cache") as mock_clear:
            cleared = maybe_empty_device_cache("cpu", threshold=0.7)

        assert cleared is False
        mock_clear.assert_not_called()


class TestContinuousSchedulerDeviceCache:
    """Unit tests for device-cache clearing around ContinuousScheduler._step()."""

    def _make_scheduler_and_request(self):
        from scheduler.continuous_scheduler import ContinuousScheduler
        from scheduler.request import InferenceRequest

        mock_engine = Mock()
        mock_engine.device = "cpu"
        scheduler = ContinuousScheduler(mock_engine, Mock(), max_batch_size=1, timeout=0.01)

        req = InferenceRequest(prompt="hi", max_tokens=5, temperature=0.0)
        req.input_ids = torch.tensor([[1, 2]])
        scheduler.active_requests = [req]
        scheduler._prepare_batch = Mock(return_value=Mock(past_width=0, new_lengths=[2]))
        return scheduler

    @pytest.mark.asyncio
    async def test_step_clears_device_cache_before_retrying_failed_forward_pass(self):
        """A transient (e.g. OOM-shaped) forward-pass failure should free cached
        device memory before the single retry -- retrying against the exact same
        memory state that just failed would almost certainly fail again."""
        try:
            scheduler = self._make_scheduler_and_request()
        except ImportError:
            pytest.skip("Required modules not available")

        call_count = 0

        def fake_forward_and_sample(batch_inputs, reqs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated OOM")
            return torch.tensor([[3]]), None

        scheduler._forward_and_sample = fake_forward_and_sample
        scheduler._dispatch_tokens = AsyncMock()

        with patch("scheduler.continuous_scheduler.empty_device_cache") as mock_empty_cache:
            await scheduler._step()

        assert call_count == 2, "Expected exactly one retry after the first failure"
        mock_empty_cache.assert_called_once_with("cpu")

    @pytest.mark.asyncio
    async def test_step_clears_device_cache_when_scheduler_goes_idle(self):
        """Once the last active request finishes for this step, cached device
        memory should be released rather than left to accumulate until the next
        request arrives."""
        try:
            scheduler = self._make_scheduler_and_request()
        except ImportError:
            pytest.skip("Required modules not available")

        scheduler._forward_and_sample = Mock(return_value=(torch.tensor([[3]]), None))

        async def fake_dispatch(*args, **kwargs):
            scheduler.active_requests = []  # simulate the request finishing this step

        scheduler._dispatch_tokens = fake_dispatch

        with patch("scheduler.continuous_scheduler.empty_device_cache") as mock_empty_cache:
            await scheduler._step()

        mock_empty_cache.assert_called_once_with("cpu")

    @pytest.mark.asyncio
    async def test_step_checks_memory_pressure_every_step_while_busy(self):
        """A long, never-idle session (active_requests never empties, no
        failures) must still be checked for memory pressure every step --
        otherwise it would only ever clear reactively (on retry) or once
        idle, and a single long-running request that never hits either could
        accumulate cached memory all the way to the device's ceiling."""
        try:
            scheduler = self._make_scheduler_and_request()
        except ImportError:
            pytest.skip("Required modules not available")

        scheduler._forward_and_sample = Mock(return_value=(torch.tensor([[3]]), None))
        scheduler._dispatch_tokens = AsyncMock()  # never empties active_requests

        with patch("scheduler.continuous_scheduler.maybe_empty_device_cache") as mock_maybe_clear, \
             patch("scheduler.continuous_scheduler.empty_device_cache") as mock_empty_cache:
            await scheduler._step()
            await scheduler._step()
            await scheduler._step()

        assert mock_maybe_clear.call_count == 3, "Pressure should be checked every step while busy"
        mock_maybe_clear.assert_called_with("cpu")
        mock_empty_cache.assert_not_called()  # unconditional clear is only for the idle/retry paths

    def test_fail_single_request_sends_generic_message_not_raw_exception_text(self):
        """The client-facing SSE error must never contain internal exception
        detail (stack-trace-flavored text, memory sizes, ...) -- only the
        generic INTERNAL_ERROR_MESSAGE. The real exception still goes onto the
        request's future for internal bookkeeping."""
        try:
            from scheduler.continuous_scheduler import ContinuousScheduler
            from scheduler.request import InferenceRequest
            from utils.errors import INTERNAL_ERROR_MESSAGE
        except ImportError:
            pytest.skip("Required modules not available")

        mock_engine = Mock()
        mock_engine.device = "cpu"
        scheduler = ContinuousScheduler(mock_engine, Mock(), max_batch_size=1, timeout=0.01)
        scheduler._free_block_table = Mock()

        req = InferenceRequest(prompt="hi", max_tokens=5, temperature=0.0)
        sensitive_exc = RuntimeError(
            "MPS backend out of memory (MPS allocated: 20.13 GiB, max allowed: 20.13 GiB)."
        )
        scheduler.active_requests = [req]

        scheduler._fail_single_request(req, sensitive_exc)

        assert req.future.exception() is sensitive_exc
        sentinel_type, message = req.queue.get_nowait()
        assert sentinel_type == "[ERROR]"
        assert message == INTERNAL_ERROR_MESSAGE
        assert "MPS" not in message and "GiB" not in message

    @pytest.mark.asyncio
    async def test_step_isolates_bad_request_instead_of_failing_whole_batch(self):
        """Regression test for whole-batch retry granularity: if the batch
        forward pass fails twice in a row, the scheduler must retry each
        active request in its own batch of one -- a request that still fails
        alone is failed individually, but co-batched requests that succeed
        alone must still complete normally instead of being failed alongside
        it."""
        try:
            from scheduler.continuous_scheduler import ContinuousScheduler
            from scheduler.request import InferenceRequest
        except ImportError:
            pytest.skip("Required modules not available")

        mock_engine = Mock()
        mock_engine.device = "cpu"
        scheduler = ContinuousScheduler(mock_engine, Mock(), max_batch_size=2, timeout=0.01)

        good_req = InferenceRequest(prompt="good", max_tokens=5, temperature=0.0)
        good_req.input_ids = torch.tensor([[1, 2]])
        bad_req = InferenceRequest(prompt="bad", max_tokens=5, temperature=0.0)
        bad_req.input_ids = torch.tensor([[3, 4]])
        scheduler.active_requests = [good_req, bad_req]

        scheduler._prepare_batch = Mock(return_value=Mock(past_width=0, new_lengths=[1, 1]))

        def fake_build_batch_inputs(reqs):
            return Mock(past_width=0, new_lengths=[1] * len(reqs))

        scheduler._build_batch_inputs = fake_build_batch_inputs

        call_count = 0

        def fake_forward_and_sample(batch_inputs, reqs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # The two whole-batch attempts (initial + retry) both fail.
                raise RuntimeError("simulated failure")
            # Isolation retries: one request per call, in list order.
            if reqs[0] is bad_req:
                raise RuntimeError("simulated failure, persists in isolation")
            return torch.tensor([[9]]), None

        scheduler._forward_and_sample = fake_forward_and_sample

        dispatched = []

        async def fake_dispatch(reqs, next_tokens, new_past, past_width, new_lengths):
            dispatched.extend(reqs)

        scheduler._dispatch_tokens = fake_dispatch

        with patch("scheduler.continuous_scheduler.empty_device_cache"):
            await scheduler._step()

        assert dispatched == [good_req], "The good request should still be dispatched normally"
        assert bad_req.future.exception() is not None, "The bad request should be failed"
        assert bad_req not in scheduler.active_requests, "The bad request should be removed from the active pool"


class TestModelSwapDeviceCleanup:
    """Unit tests for the paged-cache memory cleanup added to scheduler/model_swap.py."""

    @pytest.mark.asyncio
    async def test_swap_model_releases_old_paged_cache_memory_after_invalidating(self):
        """The old PagedKVCache's tensors are only actually dropped (by refcount)
        once `invalidate_paged_cache()` runs -- so the gc/cache-empty cleanup
        must happen *after* that call, not just once during model_loader.reload()
        (which runs earlier, while the old paged cache is still referenced)."""
        try:
            from scheduler.model_swap import swap_model
            from scheduler.request_queue import request_queue, batch_request_queue
        except ImportError:
            pytest.skip("Required modules not available")

        while not request_queue.empty():
            request_queue.queue.get_nowait()
        while not batch_request_queue.empty():
            batch_request_queue.queue.get_nowait()

        fake_scheduler = Mock()
        fake_scheduler.active_requests = []

        call_order = []

        with patch("scheduler.model_swap.tokenizer_service") as mock_tok, \
             patch("scheduler.model_swap.model_loader") as mock_loader, \
             patch("scheduler.model_swap.engine") as mock_engine, \
             patch("scheduler.model_swap.gc") as mock_gc, \
             patch("scheduler.model_swap.empty_device_cache") as mock_empty_cache:

            fake_scheduler.invalidate_paged_cache = Mock(side_effect=lambda: call_order.append("invalidate"))
            mock_gc.collect = Mock(side_effect=lambda: call_order.append("gc"))
            mock_empty_cache.side_effect = lambda device: call_order.append("empty_cache")

            result = await swap_model("new/model", fake_scheduler, drain_timeout=1.0)

        assert result == "new/model"
        mock_tok.reload.assert_called_once_with("new/model")
        mock_loader.reload.assert_called_once_with("new/model")
        mock_engine.invalidate_model_cache.assert_called_once()
        fake_scheduler.invalidate_paged_cache.assert_called_once()
        mock_gc.collect.assert_called_once()
        mock_empty_cache.assert_called_once()
        assert call_order == ["invalidate", "gc", "empty_cache"], (
            "Cleanup must run after invalidate_paged_cache() drops the old cache's last reference"
        )


class TestCliErrorHandling:
    """Unit tests for the CLI's server-error sanitization (cli/main.py)."""

    def test_extract_detail_returns_safe_detail_from_json_body(self):
        try:
            import httpx
            from ephemeris_cli.main import _extract_detail
        except ImportError:
            pytest.skip("Required modules not available")

        response = httpx.Response(500, json={"detail": "Internal server error"})
        assert _extract_detail(response) == "Internal server error"

    def test_extract_detail_falls_back_when_detail_missing(self):
        try:
            import httpx
            from ephemeris_cli.main import _extract_detail, _INTERNAL_ERROR_MESSAGE
        except ImportError:
            pytest.skip("Required modules not available")

        response = httpx.Response(500, json={"unexpected": "shape"})
        assert _extract_detail(response) == _INTERNAL_ERROR_MESSAGE

    def test_extract_detail_falls_back_on_non_json_body(self):
        try:
            import httpx
            from ephemeris_cli.main import _extract_detail, _INTERNAL_ERROR_MESSAGE
        except ImportError:
            pytest.skip("Required modules not available")

        response = httpx.Response(500, text="<html>not json</html>")
        assert _extract_detail(response) == _INTERNAL_ERROR_MESSAGE


class TestCliClientConfig:
    """Unit tests for the CLI's server-address resolution (cli/config.py)."""

    def test_packaged_config_supplies_the_default_address(self, monkeypatch):
        from ephemeris_cli.config import load_config, resolve_base_url

        monkeypatch.delenv("EPHEMERIS_CLIENT_URL", raising=False)
        monkeypatch.delenv("EPHEMERIS_CLIENT_CONFIG", raising=False)

        # No address is hardcoded in the CLI: the packaged YAML is what makes
        # a bare `ephemeris-serve start` resolve to anything at all.
        resolved = resolve_base_url(config=load_config())
        assert resolved.url.startswith("http")
        assert resolved.source == "client config"

    def test_env_var_overrides_config_file(self, monkeypatch):
        from ephemeris_cli.config import resolve_base_url

        monkeypatch.setenv("EPHEMERIS_CLIENT_URL", "https://ephemeris.example.com")
        resolved = resolve_base_url(config={"base_url": "http://127.0.0.1:8080"})
        assert resolved.url == "https://ephemeris.example.com"
        assert resolved.source == "$EPHEMERIS_CLIENT_URL"

    def test_url_option_overrides_env_var(self, monkeypatch):
        from ephemeris_cli.config import resolve_base_url

        monkeypatch.setenv("EPHEMERIS_CLIENT_URL", "https://from-env.example.com")
        resolved = resolve_base_url(url="https://from-flag.example.com", config={})
        assert resolved.url == "https://from-flag.example.com"
        assert resolved.source == "--url"

    def test_port_option_keeps_configured_host_and_scheme(self, monkeypatch):
        from ephemeris_cli.config import resolve_base_url

        monkeypatch.delenv("EPHEMERIS_CLIENT_URL", raising=False)
        resolved = resolve_base_url(port=9000, config={"base_url": "https://ephemeris.example.com:8080"})
        assert resolved.url == "https://ephemeris.example.com:9000"

    def test_host_option_keeps_configured_port(self, monkeypatch):
        from ephemeris_cli.config import resolve_base_url

        monkeypatch.delenv("EPHEMERIS_CLIENT_URL", raising=False)
        resolved = resolve_base_url(host="10.0.0.5", config={"base_url": "http://127.0.0.1:8080"})
        assert resolved.url == "http://10.0.0.5:8080"

    def test_url_combined_with_host_is_rejected(self):
        from ephemeris_cli.config import ClientConfigError, resolve_base_url

        with pytest.raises(ClientConfigError):
            resolve_base_url(url="http://a.example.com", host="b.example.com", config={})

    def test_bare_host_is_normalized_to_http(self):
        from ephemeris_cli.config import normalize_base_url

        assert normalize_base_url("ephemeris.example.com") == "http://ephemeris.example.com"
        assert normalize_base_url("ephemeris.example.com:8080") == "http://ephemeris.example.com:8080"

    def test_trailing_slash_is_stripped(self):
        from ephemeris_cli.config import normalize_base_url

        # Every request path the CLI builds starts with "/", so a trailing
        # slash here would produce "//health".
        assert normalize_base_url("https://ephemeris.example.com/") == "https://ephemeris.example.com"

    def test_non_http_scheme_is_rejected(self):
        from ephemeris_cli.config import ClientConfigError, normalize_base_url

        with pytest.raises(ClientConfigError):
            normalize_base_url("ftp://ephemeris.example.com")

    def test_user_config_file_overrides_packaged_default(self, monkeypatch, tmp_path):
        from ephemeris_cli.config import load_config, resolve_base_url

        monkeypatch.delenv("EPHEMERIS_CLIENT_URL", raising=False)
        monkeypatch.delenv("EPHEMERIS_CLIENT_CONFIG", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

        user_config = tmp_path / "ephemeris-serve" / "client.yaml"
        user_config.parent.mkdir(parents=True)
        user_config.write_text(
            'client_config:\n  defaults:\n    base_url: "https://prod.example.com"\n'
        )

        assert resolve_base_url(config=load_config()).url == "https://prod.example.com"

    def test_timeout_falls_back_to_config_then_option(self, monkeypatch):
        from ephemeris_cli.config import resolve_timeout

        assert resolve_timeout(None, {"timeout_seconds": 45.0}) == 45.0
        assert resolve_timeout(5.0, {"timeout_seconds": 45.0}) == 5.0


class TestApiKeyAuth:
    """Unit tests for API-key authentication (api/auth.py)."""

    @staticmethod
    def _configure(monkeypatch, keys="", admin_keys=""):
        from settings.settings import secret_settings

        monkeypatch.setattr(secret_settings, "api_keys", keys, raising=False)
        monkeypatch.setattr(secret_settings, "admin_api_keys", admin_keys, raising=False)

    def test_auth_disabled_when_no_keys_configured(self, monkeypatch):
        from api.auth import auth_enabled, require_api_key

        self._configure(monkeypatch)
        # Local development and the test suite rely on this staying open.
        assert auth_enabled() is False
        assert require_api_key(authorization=None) is None

    def test_valid_key_is_accepted(self, monkeypatch):
        from api.auth import require_api_key

        self._configure(monkeypatch, keys="secret-one,secret-two")
        assert require_api_key(authorization="Bearer secret-two") == "secret-two"

    def test_missing_header_is_rejected_with_401(self, monkeypatch):
        from fastapi import HTTPException
        from api.auth import require_api_key

        self._configure(monkeypatch, keys="secret-one")
        with pytest.raises(HTTPException) as excinfo:
            require_api_key(authorization=None)
        assert excinfo.value.status_code == 401

    def test_wrong_key_is_rejected_with_401(self, monkeypatch):
        from fastapi import HTTPException
        from api.auth import require_api_key

        self._configure(monkeypatch, keys="secret-one")
        with pytest.raises(HTTPException) as excinfo:
            require_api_key(authorization="Bearer nope")
        assert excinfo.value.status_code == 401

    def test_non_bearer_scheme_is_rejected(self, monkeypatch):
        from fastapi import HTTPException
        from api.auth import require_api_key

        self._configure(monkeypatch, keys="secret-one")
        with pytest.raises(HTTPException):
            require_api_key(authorization="Basic secret-one")

    def test_admin_key_satisfies_the_ordinary_tier(self, monkeypatch):
        from api.auth import require_api_key

        self._configure(monkeypatch, keys="", admin_keys="admin-key")
        assert require_api_key(authorization="Bearer admin-key") == "admin-key"

    def test_ordinary_key_cannot_swap_the_model(self, monkeypatch):
        from fastapi import HTTPException
        from api.auth import require_admin_api_key

        self._configure(monkeypatch, keys="user-key", admin_keys="admin-key")
        with pytest.raises(HTTPException) as excinfo:
            require_admin_api_key(token="user-key")
        assert excinfo.value.status_code == 403

    def test_admin_key_can_swap_the_model(self, monkeypatch):
        from api.auth import require_admin_api_key

        self._configure(monkeypatch, keys="user-key", admin_keys="admin-key")
        assert require_admin_api_key(token="admin-key") == "admin-key"

    def test_model_swap_refused_when_no_admin_key_configured(self, monkeypatch):
        from fastapi import HTTPException
        from api.auth import require_admin_api_key

        # Keys exist but no admin tier: refuse rather than letting every key
        # load an arbitrary model.
        self._configure(monkeypatch, keys="user-key", admin_keys="")
        with pytest.raises(HTTPException) as excinfo:
            require_admin_api_key(token="user-key")
        assert excinfo.value.status_code == 403

    def test_generation_routes_declare_the_auth_dependency(self):
        from api.routes import router
        from api.auth import require_admin_api_key, require_api_key

        protected = {}
        for route in router.routes:
            calls = {dep.dependency for dep in route.dependencies}
            for method in route.methods:
                protected[(method, route.path)] = calls

        assert require_api_key in protected[("POST", "/generate")]
        assert require_api_key in protected[("POST", "/generate_batch")]
        assert require_api_key in protected[("GET", "/metrics")]
        # The model-swap route is the dangerous one: admin tier only.
        assert require_admin_api_key in protected[("POST", "/model")]


class TestSecretSettingTolerance:
    """SecretSetting must not crash on env entries it doesn't own."""

    def test_unknown_env_file_entries_are_ignored(self, tmp_path, monkeypatch):
        from settings.settings import SecretSetting

        env_file = tmp_path / ".env"
        # EPHEMERIS_API_KEY (singular) is the *client's* variable, and an
        # EnvironmentFile in a real deployment carries plenty else besides.
        # Any of it previously raised ValidationError at import time, taking
        # the server down before it could log why.
        env_file.write_text(
            "HF_KEY=hf_test\n"
            "EPHEMERIS_CLIENT_API_KEY=client-side-key\n"
            "EPHEMERIS_SERVER_API_KEYS=server-key\n"
            "SOMETHING_UNRELATED=x\n"
        )
        monkeypatch.delenv("EPHEMERIS_SERVER_API_KEYS", raising=False)

        settings = SecretSetting(_env_file=str(env_file))
        assert settings.api_keys == "server-key"
        assert settings.hf_key == "hf_test"


class TestClientPathMigration:
    """The CLI renamed to `ephemeris`; its old on-disk paths still resolve."""

    def test_new_config_dir_used_when_neither_exists(self, monkeypatch, tmp_path):
        from ephemeris_cli import config as client_config

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert client_config.user_config_path() == tmp_path / "ephemeris" / "client.yaml"

    def test_legacy_config_dir_used_when_only_it_exists(self, monkeypatch, tmp_path):
        from ephemeris_cli import config as client_config

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        (tmp_path / "ephemeris-serve").mkdir()
        # An upgrade must not silently ignore a config written under the old name.
        assert client_config.user_config_path() == tmp_path / "ephemeris-serve" / "client.yaml"

    def test_new_config_dir_wins_when_both_exist(self, monkeypatch, tmp_path):
        from ephemeris_cli import config as client_config

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        (tmp_path / "ephemeris-serve").mkdir()
        (tmp_path / "ephemeris").mkdir()
        # Once migrated, the old directory must never resurface.
        assert client_config.user_config_path() == tmp_path / "ephemeris" / "client.yaml"

    def test_user_level_env_follows_the_same_directory(self, monkeypatch, tmp_path, real_env_file_paths):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        (tmp_path / "ephemeris-serve").mkdir()
        # The user-level .env must live beside client.yaml, legacy dir included.
        assert tmp_path / "ephemeris-serve" / ".env" in real_env_file_paths()

    def test_history_file_names(self):
        from ephemeris_cli import main as client_main

        assert client_main._HISTORY_FILE.name == ".ephemeris_history"
        assert client_main._LEGACY_HISTORY_FILE.name == ".ephemeris_serve_history"

    def test_cli_titles_itself_ephemeris(self):
        from ephemeris_cli import main as client_main

        assert client_main._TITLE == "EPHEMERIS"


class TestClientEnvFiles:
    """The client reads its own .env, and only the keys it owns."""

    def test_env_file_supplies_url_and_key(self, monkeypatch, tmp_path):
        from ephemeris_cli import config as client_config

        monkeypatch.delenv("EPHEMERIS_CLIENT_URL", raising=False)
        monkeypatch.delenv("EPHEMERIS_CLIENT_API_KEY", raising=False)

        env_file = tmp_path / ".env"
        env_file.write_text(
            "EPHEMERIS_CLIENT_URL=https://from-dotenv.example.com\n"
            "EPHEMERIS_CLIENT_API_KEY=dotenv-key\n"
        )
        monkeypatch.setattr(client_config, "env_file_paths", lambda: [env_file])

        assert client_config.resolve_base_url(config={}).url == "https://from-dotenv.example.com"
        assert client_config.resolve_api_key(config={}).value == "dotenv-key"

    def test_server_keys_in_a_shared_env_file_are_ignored(self, monkeypatch, tmp_path):
        from ephemeris_cli import config as client_config

        monkeypatch.delenv("EPHEMERIS_CLIENT_API_KEY", raising=False)

        # The scopes must not bleed even when one file holds both sides --
        # exactly what happens running the CLI from the server's repo root.
        env_file = tmp_path / ".env"
        env_file.write_text(
            "HF_KEY=hf_secret\n"
            "EPHEMERIS_SERVER_API_KEYS=server-only-key\n"
            "EPHEMERIS_SERVER_ADMIN_API_KEYS=server-admin-key\n"
        )
        monkeypatch.setattr(client_config, "env_file_paths", lambda: [env_file])

        assert client_config.load_env_files() == {}
        assert client_config.resolve_api_key(config={}) is None

    def test_process_environment_beats_env_file(self, monkeypatch, tmp_path):
        from ephemeris_cli import config as client_config

        env_file = tmp_path / ".env"
        env_file.write_text("EPHEMERIS_CLIENT_API_KEY=from-file\n")
        monkeypatch.setattr(client_config, "env_file_paths", lambda: [env_file])
        monkeypatch.setenv("EPHEMERIS_CLIENT_API_KEY", "from-shell")

        assert client_config.resolve_api_key(config={}).value == "from-shell"

    def test_later_env_files_win(self, monkeypatch, tmp_path):
        from ephemeris_cli import config as client_config

        monkeypatch.delenv("EPHEMERIS_CLIENT_URL", raising=False)
        low = tmp_path / "low.env"
        high = tmp_path / "high.env"
        low.write_text("EPHEMERIS_CLIENT_URL=http://low.example.com\n")
        high.write_text("EPHEMERIS_CLIENT_URL=http://high.example.com\n")
        monkeypatch.setattr(client_config, "env_file_paths", lambda: [low, high])

        assert client_config.resolve_base_url(config={}).url == "http://high.example.com"

    def test_parser_handles_quotes_comments_and_export(self, monkeypatch, tmp_path):
        from ephemeris_cli import config as client_config

        env_file = tmp_path / ".env"
        env_file.write_text(
            "# a comment\n"
            "\n"
            'export EPHEMERIS_CLIENT_API_KEY="quoted-key"\n'
            "EPHEMERIS_CLIENT_URL='http://quoted.example.com'\n"
            "MALFORMED_LINE_WITHOUT_EQUALS\n"
        )
        parsed = client_config._parse_env_file(env_file)
        assert parsed["EPHEMERIS_CLIENT_API_KEY"] == "quoted-key"
        assert parsed["EPHEMERIS_CLIENT_URL"] == "http://quoted.example.com"

    def test_missing_env_files_are_not_an_error(self, tmp_path):
        from ephemeris_cli import config as client_config

        assert client_config._parse_env_file(tmp_path / "nope.env") == {}


class TestClientServerIsolation:
    """The client distribution must stay installable without the server."""

    def test_client_package_imports_nothing_from_the_server(self):
        import subprocess
        import sys
        from pathlib import Path

        cli_root = Path(__file__).resolve().parent.parent / "packages" / "ephemeris-cli"
        # A fresh interpreter with only the client package on the path -- if
        # anything in it reaches for torch/fastapi/the server's settings, this
        # fails here rather than in a user's thin install.
        probe = (
            "import sys; sys.path.insert(0, %r);"
            "import ephemeris_cli.main, ephemeris_cli.config;"
            "heavy = [m for m in ('torch','transformers','fastapi','uvicorn','settings','api') "
            "if m in sys.modules];"
            "print(','.join(heavy))" % str(cli_root)
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(cli_root)
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "", f"client pulled in server modules: {result.stdout.strip()}"


class TestCliAuthConfig:
    """Unit tests for the CLI's API-key resolution (cli/config.py)."""

    def test_env_var_beats_config_file(self, monkeypatch):
        from ephemeris_cli.config import resolve_api_key

        monkeypatch.setenv("EPHEMERIS_CLIENT_API_KEY", "from-env")
        resolved = resolve_api_key(config={"api_keys": "from-file"})
        assert resolved.value == "from-env"
        assert resolved.source == "$EPHEMERIS_CLIENT_API_KEY"

    def test_option_beats_env_var(self, monkeypatch):
        from ephemeris_cli.config import resolve_api_key

        monkeypatch.setenv("EPHEMERIS_CLIENT_API_KEY", "from-env")
        assert resolve_api_key("from-flag", config={}).value == "from-flag"

    def test_absent_key_resolves_to_none(self, monkeypatch):
        from ephemeris_cli.config import resolve_api_key

        monkeypatch.delenv("EPHEMERIS_CLIENT_API_KEY", raising=False)
        # A server with no keys configured accepts unauthenticated requests.
        assert resolve_api_key(config={}) is None

    def test_auth_headers_omitted_when_no_key(self):
        from ephemeris_cli.config import auth_headers

        assert auth_headers(None) == {}
        assert auth_headers("k") == {"Authorization": "Bearer k"}

    def test_mask_secret_never_reveals_the_middle(self):
        from ephemeris_cli.config import mask_secret

        masked = mask_secret("abcdefghijklmnop")
        assert masked.startswith("abcd") and masked.endswith("mnop")
        assert "efghijkl" not in masked
        assert mask_secret("short") == "*****"


class TestCLIRendering:
    """Terminal rendering rules for the `ephemeris` REPL's boxes and prompt."""

    def _render(self, capsys, columns, func):
        """Run `func` with the terminal reported as `columns` wide, return its lines."""
        import os
        import shutil

        import ephemeris_cli.main as cli_main

        real_size = shutil.get_terminal_size
        shutil.get_terminal_size = lambda fallback=(80, 24): os.terminal_size((columns, 24))
        try:
            func(cli_main)
        finally:
            shutil.get_terminal_size = real_size
        return capsys.readouterr().out.splitlines()

    def test_long_row_wraps_instead_of_breaking_the_border(self, capsys):
        long_url = "http://a-very-long-hostname-for-testing.example.com:8443/api-prefix"
        lines = self._render(
            capsys, 60, lambda m: m._box_row(f"Connected to {long_url}", m._box_width())
        )

        assert len(lines) > 1, "the long line should have wrapped onto several rows"
        widths = {len(line) for line in lines}
        assert len(widths) == 1, f"rows must all be the same width, got {widths}"
        assert all(line.startswith("│ ") and line.endswith(" │") for line in lines)

    def test_unbreakable_token_is_split_rather_than_overflowing(self, capsys):
        lines = self._render(capsys, 40, lambda m: m._box_row("x" * 200, m._box_width()))

        assert len(set(len(line) for line in lines)) == 1
        assert "".join(line[2:-2].rstrip() for line in lines) == "x" * 200

    def test_oversized_top_label_does_not_widen_the_box(self, capsys):
        lines = self._render(
            capsys,
            40,
            lambda m: (m._box_top("L" * 200, m._box_width()), m._box_bottom(m._box_width())),
        )

        assert len(lines[0]) == len(lines[1])

    def test_welcome_box_never_prints_any_of_the_api_key(self, capsys):
        from ephemeris_cli.config import ResolvedValue

        secret = "sk-abcdefghijklmnopqrstuvwxyz0123456789"
        lines = self._render(
            capsys,
            100,
            lambda m: m._print_welcome(
                "http://127.0.0.1:8000",
                "client config",
                ResolvedValue(secret, "$EPHEMERIS_CLIENT_API_KEY"),
                "balanced (temperature 0.7)",
            ),
        )
        rendered = "\n".join(lines)

        assert secret not in rendered
        # Not even the leading/trailing runs a masked rendering would show.
        assert secret[:4] not in rendered and secret[-4:] not in rendered
        assert "API key: set (from $EPHEMERIS_CLIENT_API_KEY)" in rendered

    def test_prompt_is_plain_under_libedit(self, monkeypatch):
        """libedit counts inline escapes as columns, so a recalled line wraps early
        and backspace stops erasing; fenced escapes it hoists ahead of the text,
        losing the colour anyway. An uncoloured prompt is the only one that edits."""
        import ephemeris_cli.main as cli_main

        monkeypatch.setattr(cli_main, "_stdout_is_tty", lambda: True)
        monkeypatch.setattr(cli_main, "_readline_is_libedit", lambda: True)

        assert cli_main._readline_safe_prompt("you> ") == "you> "

    def test_prompt_is_fenced_and_yellow_under_gnu_readline(self, monkeypatch):
        import ephemeris_cli.main as cli_main

        monkeypatch.setattr(cli_main, "_stdout_is_tty", lambda: True)
        monkeypatch.setattr(cli_main, "_readline_is_libedit", lambda: False)

        styled = cli_main._readline_safe_prompt("you> ")

        # Every escape run fenced, so readline counts the prompt as 5 columns.
        assert styled == "\001\033[33m\033[1m\002you> \001\033[0m\002"

    def test_prompt_is_unstyled_when_stdout_is_not_a_terminal(self, monkeypatch):
        import ephemeris_cli.main as cli_main

        monkeypatch.setattr(cli_main, "_stdout_is_tty", lambda: False)
        assert cli_main._readline_safe_prompt("you> ") == "you> "


class TestCLIHistoryFile:
    """Reading the REPL history file written by either readline flavor."""

    def test_libedit_file_is_decoded_not_taken_literally(self, tmp_path):
        import ephemeris_cli.main as cli_main

        # Exactly what macOS's libedit writes: a header line, then entries with
        # every space escaped as an octal \040.
        path = tmp_path / "history"
        path.write_text("_HiStOrY_V2_\nfirst\\040unique\\040message\n2\\040+\\0402\\040in\\040one\\040word\n")

        assert cli_main._read_history_entries(path) == [
            "first unique message",
            "2 + 2 in one word",
        ]

    def test_gnu_readline_file_is_read_verbatim(self, tmp_path):
        import ephemeris_cli.main as cli_main

        path = tmp_path / "history"
        path.write_text("a real message\nliteral \\040 stays\n")

        # No libedit header, so nothing is an escape sequence here.
        assert cli_main._read_history_entries(path) == ["a real message", "literal \\040 stays"]

    def test_session_noise_is_dropped(self, tmp_path):
        import ephemeris_cli.main as cli_main

        path = tmp_path / "history"
        path.write_text("/exit\nhello\nhello\n\n/quit\nhello again\n")

        assert cli_main._read_history_entries(path) == ["hello", "hello again"]

    def test_missing_file_is_empty_history(self, tmp_path):
        import ephemeris_cli.main as cli_main

        assert cli_main._read_history_entries(tmp_path / "nope") == []

    def test_history_is_capped_to_the_configured_length(self, tmp_path):
        import ephemeris_cli.main as cli_main

        path = tmp_path / "history"
        path.write_text("".join(f"message {i}\n" for i in range(cli_main._HISTORY_LENGTH + 50)))

        entries = cli_main._read_history_entries(path)
        assert len(entries) == cli_main._HISTORY_LENGTH
        # The cap keeps the newest entries, not the oldest.
        assert entries[-1] == f"message {cli_main._HISTORY_LENGTH + 49}"

    def test_save_rewrites_the_file_without_the_noise(self, tmp_path, monkeypatch):
        import ephemeris_cli.main as cli_main

        path = tmp_path / "history"
        monkeypatch.setattr(cli_main, "_HISTORY_FILE", path)
        monkeypatch.setattr(
            cli_main, "_current_history_entries", lambda: ["hi", "hi", "/exit", "  ", "bye"]
        )

        cli_main._save_readline_history()

        assert path.read_text() == "hi\nbye\n"
        # Written in the flavor-neutral format both readers can be handed back.
        assert cli_main._read_history_entries(path) == ["hi", "bye"]

    def test_saved_file_is_not_world_readable(self, tmp_path, monkeypatch):
        import ephemeris_cli.main as cli_main

        path = tmp_path / "history"
        monkeypatch.setattr(cli_main, "_HISTORY_FILE", path)
        monkeypatch.setattr(cli_main, "_current_history_entries", lambda: ["a prompt"])

        cli_main._save_readline_history()

        # Prompts can carry anything the user typed; keep them owner-only.
        assert path.stat().st_mode & 0o077 == 0


class TestPagedKVCache:
    """Pure-tensor unit tests for the paged KV cache storage layer (no model needed)."""

    def _make_cache(self, num_layers=2, num_kv_heads=2, head_dim=4, block_size=4):
        try:
            from cache.paged_kv_cache import PagedKVCache
        except ImportError:
            pytest.skip("Paged KV cache module not available")
        return PagedKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            initial_capacity_blocks=4,
        )

    def _kv_step(self, cache, n_tokens, fill_value):
        """Build (keys_per_layer, values_per_layer) filled with a distinct value per call."""
        shape = (cache.num_kv_heads, n_tokens, cache.head_dim)
        keys = [torch.full(shape, float(fill_value)) for _ in range(cache.num_layers)]
        values = [torch.full(shape, float(fill_value) + 0.5) for _ in range(cache.num_layers)]
        return keys, values

    def test_append_and_gather_round_trip(self):
        """Appended K/V must come back unchanged, in order, for the real length only."""
        from cache.paged_kv_cache import BlockTable

        cache = self._make_cache(block_size=4)
        table = BlockTable()

        keys, values = self._kv_step(cache, n_tokens=3, fill_value=1)
        cache.append(table, keys, values)

        assert table.length == 3
        gathered_k, gathered_v, real_lengths = cache.gather_dense([table])
        assert real_lengths == [3]
        for layer_idx in range(cache.num_layers):
            assert gathered_k[layer_idx].shape == (1, cache.num_kv_heads, 3, cache.head_dim)
            assert torch.equal(gathered_k[layer_idx], keys[layer_idx].unsqueeze(0))
            assert torch.equal(gathered_v[layer_idx], values[layer_idx].unsqueeze(0))

    def test_append_across_block_boundary(self):
        """Appending past a block's capacity must allocate a new block and stay contiguous."""
        from cache.paged_kv_cache import BlockTable

        cache = self._make_cache(block_size=4)
        table = BlockTable()

        first_keys, first_values = self._kv_step(cache, n_tokens=4, fill_value=1)
        cache.append(table, first_keys, first_values)
        assert len(table.block_ids) == 1

        second_keys, second_values = self._kv_step(cache, n_tokens=3, fill_value=2)
        cache.append(table, second_keys, second_values)
        assert len(table.block_ids) == 2
        assert table.length == 7

        gathered_k, gathered_v, real_lengths = cache.gather_dense([table])
        assert real_lengths == [7]
        expected_k = torch.cat([first_keys[0], second_keys[0]], dim=1)
        assert torch.equal(gathered_k[0], expected_k.unsqueeze(0))

    def test_gather_dense_left_pads_across_batch(self):
        """Shorter requests in a batch are left-padded to the longest request's length."""
        from cache.paged_kv_cache import BlockTable

        cache = self._make_cache(block_size=4)
        short_table = BlockTable()
        long_table = BlockTable()

        s_keys, s_values = self._kv_step(cache, n_tokens=2, fill_value=1)
        cache.append(short_table, s_keys, s_values)
        l_keys, l_values = self._kv_step(cache, n_tokens=5, fill_value=2)
        cache.append(long_table, l_keys, l_values)

        gathered_k, gathered_v, real_lengths = cache.gather_dense([short_table, long_table])
        assert real_lengths == [2, 5]
        assert gathered_k[0].shape == (2, cache.num_kv_heads, 5, cache.head_dim)
        # Left padding: short_table's real data must sit in the last 2 columns.
        assert torch.equal(gathered_k[0][0, :, -2:, :], s_keys[0])
        assert torch.equal(gathered_k[0][0, :, :3, :], torch.zeros(cache.num_kv_heads, 3, cache.head_dim))

    def test_free_and_reuse_no_cross_request_bleed(self):
        """A freed block, once reused by a new request, must not leak into any other
        still-active request's gathered view."""
        from cache.paged_kv_cache import BlockTable

        cache = self._make_cache(block_size=4)
        req_a = BlockTable()
        req_b = BlockTable()

        a_keys, a_values = self._kv_step(cache, n_tokens=4, fill_value=1)
        cache.append(req_a, a_keys, a_values)
        b_keys, b_values = self._kv_step(cache, n_tokens=4, fill_value=2)
        cache.append(req_b, b_keys, b_values)

        cache.free(req_a)
        assert req_a.block_ids == []
        assert req_a.length == 0

        req_c = BlockTable()
        c_keys, c_values = self._kv_step(cache, n_tokens=4, fill_value=3)
        cache.append(req_c, c_keys, c_values)

        # req_b's data must be untouched by req_a's free + req_c's reuse.
        gathered_k, _, _ = cache.gather_dense([req_b, req_c])
        assert torch.equal(gathered_k[0][0], b_keys[0])
        assert torch.equal(gathered_k[0][1], c_keys[0])

    def test_is_valid_detects_corruption(self):
        from cache.paged_kv_cache import BlockTable

        cache = self._make_cache(block_size=4)

        empty_table = BlockTable()
        assert cache.is_valid(empty_table) is True

        table = BlockTable()
        keys, values = self._kv_step(cache, n_tokens=3, fill_value=1)
        cache.append(table, keys, values)
        assert cache.is_valid(table) is True

        corrupted = BlockTable(block_ids=list(table.block_ids), length=table.length)
        cache.free(table)
        # `corrupted` still references blocks that are now back on the free list.
        assert cache.is_valid(corrupted) is False

    def test_pool_grows_when_exhausted(self):
        from cache.paged_kv_cache import BlockTable

        cache = self._make_cache(block_size=4)
        initial_capacity = cache.capacity

        tables = [BlockTable() for _ in range(10)]
        for i, table in enumerate(tables):
            keys, values = self._kv_step(cache, n_tokens=4, fill_value=i)
            cache.append(table, keys, values)

        assert cache.capacity > initial_capacity
        gathered_k, _, real_lengths = cache.gather_dense(tables)
        assert real_lengths == [4] * 10
        for i in range(10):
            assert torch.equal(gathered_k[0][i], torch.full((cache.num_kv_heads, 4, cache.head_dim), float(i)))


def _run_scheduler_step(scheduler, engine):
    """Run one `_prepare_batch`/`forward_step`/`_dispatch_tokens` cycle, greedily
    (temperature=0 on every request, for deterministic comparisons)."""
    batch_inputs = scheduler._prepare_batch()
    assert batch_inputs is not None
    logits, new_past = engine.forward_step(
        batch_inputs.input_ids,
        batch_inputs.attention_mask,
        batch_inputs.past_key_values,
        position_ids=batch_inputs.position_ids,
        logit_gather_indices=batch_inputs.logit_gather_indices,
    )
    next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
    asyncio.run(
        scheduler._dispatch_tokens(
            scheduler.active_requests, next_tokens, new_past, batch_inputs.past_width, batch_inputs.new_lengths
        )
    )
    return next_tokens


def _make_scheduler_request(prompt, engine, tokenizer_service):
    from scheduler.request import InferenceRequest

    req = InferenceRequest(prompt=prompt, max_tokens=10, temperature=0.0)
    encoded = tokenizer_service.encode(req.prompt, return_tensors=True)
    req.input_ids = encoded["input_ids"].to(engine.device)
    return req


class TestTokenizerService:
    """Unit tests for tokenizer service"""

    def test_tokenizer_encode(self, test_prompt):
        """Test tokenizer encoding"""
        try:
            from tokenizer.tokenizer_service import tokenizer_service
        except ImportError:
            pytest.skip("Tokenizer service not available")

        # encode() with default return_tensors=False returns a list
        tokens = tokenizer_service.encode(test_prompt)
        assert tokens is not None
        # Can be list or dict depending on tokenizer state
        assert isinstance(tokens, (list, tuple, dict))
        if isinstance(tokens, dict):
            assert 'input_ids' in tokens or len(tokens) > 0
        else:
            assert len(tokens) > 0

    def test_tokenizer_decode(self):
        """Test tokenizer decoding"""
        try:
            from tokenizer.tokenizer_service import tokenizer_service
        except ImportError:
            pytest.skip("Tokenizer service not available")

        test_tokens = [101, 1045, 2001, 102]  # Sample tokens
        decoded = tokenizer_service.decode(test_tokens)
        assert decoded is not None
        assert isinstance(decoded, str)

    def test_kv_cache_hit_miss_rate(self, test_prompt):
        """Test KV cache hit/miss behavior during scheduler steps."""
        try:
            from scheduler.continuous_scheduler import ContinuousScheduler
            from scheduler.request import InferenceRequest
            from engine.model_loader import model_loader
            from engine.generator import engine
            from tokenizer.tokenizer_service import tokenizer_service
        except ImportError:
            pytest.skip("Required modules not available")

        model_loader.load()
        tokenizer_service.load()

        scheduler = ContinuousScheduler(engine, tokenizer_service, max_batch_size=1, timeout=0.01)
        request = InferenceRequest(prompt=test_prompt, max_tokens=10, temperature=1.0)

        encoded = tokenizer_service.encode(request.prompt, return_tensors=True)
        request.input_ids = encoded["input_ids"].to(engine.device)
        scheduler.active_requests = [request]

        misses = 0
        hits = 0
        total_steps = 3

        non_eos_token = 0
        if engine.eos_token_id == non_eos_token:
            non_eos_token = 1

        for _ in range(total_steps):
            batch_inputs = scheduler._prepare_batch()
            assert batch_inputs is not None
            if batch_inputs.past_key_values is None:
                misses += 1
            else:
                hits += 1

            logits, new_past = engine.forward_step(
                batch_inputs.input_ids,
                batch_inputs.attention_mask,
                batch_inputs.past_key_values,
                position_ids=batch_inputs.position_ids,
                logit_gather_indices=batch_inputs.logit_gather_indices,
            )
            next_tokens = torch.tensor([[non_eos_token]], dtype=torch.long, device=engine.device)
            asyncio.run(
                scheduler._dispatch_tokens(
                    scheduler.active_requests, next_tokens, new_past, batch_inputs.past_width, batch_inputs.new_lengths
                )
            )

        assert request.block_table.length > 0, "Expected request KV cache to be populated after the first step"
        assert misses == 1, f"Expected exactly one cache miss, got {misses}"
        assert hits == total_steps - 1, f"Expected {total_steps - 1} cache hits, got {hits}"

    def test_incremental_batch_with_different_past_lengths_matches_solo_runs(self):
        """Two requests with different (paged) past lengths sharing an incremental
        batch step must each produce the same next-token as if run alone --
        proving the per-row left-padding in `_prepare_batch`'s incremental
        branch doesn't leak between rows."""
        try:
            from scheduler.continuous_scheduler import ContinuousScheduler
            from engine.model_loader import model_loader
            from engine.generator import engine
            from tokenizer.tokenizer_service import tokenizer_service
        except ImportError:
            pytest.skip("Required modules not available")

        model_loader.load()
        tokenizer_service.load()

        short_prompt = "Hi"
        long_prompt = "Tell me a detailed story about a journey through the mountains"

        # Solo baseline: each request alone through a miss step then a hit step.
        solo_next_tokens = {}
        for prompt in (short_prompt, long_prompt):
            solo_scheduler = ContinuousScheduler(engine, tokenizer_service, max_batch_size=1, timeout=0.01)
            req = _make_scheduler_request(prompt, engine, tokenizer_service)
            solo_scheduler.active_requests = [req]
            _run_scheduler_step(solo_scheduler, engine)  # miss (full prompt)
            solo_next_tokens[prompt] = _run_scheduler_step(solo_scheduler, engine)  # hit (incremental)

        # Combined: both requests share one batch through the same two steps.
        # Their prompts differ in length, so once both reach the incremental
        # branch they have different `block_table.length` -- exactly the
        # multi-request left-padding path this test targets.
        combined_scheduler = ContinuousScheduler(engine, tokenizer_service, max_batch_size=2, timeout=0.01)
        req_short = _make_scheduler_request(short_prompt, engine, tokenizer_service)
        req_long = _make_scheduler_request(long_prompt, engine, tokenizer_service)
        combined_scheduler.active_requests = [req_short, req_long]
        _run_scheduler_step(combined_scheduler, engine)  # miss (full prompt, padded together)
        combined_next_tokens = _run_scheduler_step(combined_scheduler, engine)  # hit (incremental, padded together)

        assert combined_next_tokens[0].item() == solo_next_tokens[short_prompt].item(), (
            "Short-prompt request's next token changed when batched with a longer request"
        )
        assert combined_next_tokens[1].item() == solo_next_tokens[long_prompt].item(), (
            "Long-prompt request's next token changed when batched with a shorter request"
        )

    def test_mixed_prefill_and_decode_in_same_step(self):
        """The core Feature-A guarantee: a brand-new (prefill) request sharing
        a batch step with an already-decoding request must not perturb the
        decoding request's next-token prediction, and the new request's own
        prediction must match what it would get running alone."""
        try:
            from scheduler.continuous_scheduler import ContinuousScheduler
            from engine.model_loader import model_loader
            from engine.generator import engine
            from tokenizer.tokenizer_service import tokenizer_service
        except ImportError:
            pytest.skip("Required modules not available")

        model_loader.load()
        tokenizer_service.load()

        decoding_prompt = "The weather today is"
        joining_prompt = "In the beginning"

        # Solo baseline: the decoding request's 2nd step (its 1st decode,
        # following an initial prefill).
        solo_decoder = ContinuousScheduler(engine, tokenizer_service, max_batch_size=1, timeout=0.01)
        req_decoder_solo = _make_scheduler_request(decoding_prompt, engine, tokenizer_service)
        solo_decoder.active_requests = [req_decoder_solo]
        _run_scheduler_step(solo_decoder, engine)  # prefill
        expected_decoder = _run_scheduler_step(solo_decoder, engine).item()  # decode #1

        # Solo baseline: the joining request's 1st (prefill) step, run alone.
        solo_joiner = ContinuousScheduler(engine, tokenizer_service, max_batch_size=1, timeout=0.01)
        req_joiner_solo = _make_scheduler_request(joining_prompt, engine, tokenizer_service)
        solo_joiner.active_requests = [req_joiner_solo]
        expected_joiner = _run_scheduler_step(solo_joiner, engine).item()

        # Combined: the decoder does its prefill step alone, then a brand-new
        # request joins for the decoder's 2nd step -- one decoding row and
        # one prefilling row batched into the SAME forward pass. This is
        # exactly the case the old binary-split design couldn't do (it would
        # have forced the decoder back through a full recompute instead).
        combined = ContinuousScheduler(engine, tokenizer_service, max_batch_size=2, timeout=0.01)
        req_decoder = _make_scheduler_request(decoding_prompt, engine, tokenizer_service)
        combined.active_requests = [req_decoder]
        _run_scheduler_step(combined, engine)  # decoder's prefill

        req_joiner = _make_scheduler_request(joining_prompt, engine, tokenizer_service)
        combined.active_requests.append(req_joiner)
        assert combined.active_requests[0].block_table.length > 0  # decoder: mid-decode
        assert combined.active_requests[1].block_table.length == 0  # joiner: fresh prefill

        mixed_next_tokens = _run_scheduler_step(combined, engine)

        assert mixed_next_tokens[0].item() == expected_decoder, (
            "Decoding request's next token changed when a new prefill request joined its batch step"
        )
        assert mixed_next_tokens[1].item() == expected_joiner, (
            "Newly-joined prefill request's next token differed from its solo run"
        )

    def test_stop_sequence_halts_generation_and_trims_output(self):
        """A request whose stop sequence matches the very first generated token's
        text must finish after that one step, with the stop text trimmed from
        the result -- proving the scheduler checks stop sequences per-step
        rather than only at EOS/max_tokens."""
        try:
            from scheduler.continuous_scheduler import ContinuousScheduler
            from engine.model_loader import model_loader
            from engine.generator import engine
            from tokenizer.tokenizer_service import tokenizer_service
        except ImportError:
            pytest.skip("Required modules not available")

        model_loader.load()
        tokenizer_service.load()

        prompt = "The weather today is"

        # Probe (no stop sequences) to deterministically learn what the first
        # generated token decodes to, at temperature=0.0 (greedy).
        probe_scheduler = ContinuousScheduler(engine, tokenizer_service, max_batch_size=1, timeout=0.01)
        probe_req = _make_scheduler_request(prompt, engine, tokenizer_service)
        probe_scheduler.active_requests = [probe_req]
        _run_scheduler_step(probe_scheduler, engine)
        first_token_text = tokenizer_service.decode(probe_req.generated_tokens)
        assert first_token_text, "Expected the probe step to generate non-empty text"

        scheduler = ContinuousScheduler(engine, tokenizer_service, max_batch_size=1, timeout=0.01)
        req = _make_scheduler_request(prompt, engine, tokenizer_service)
        req.stop_sequences = [first_token_text]
        scheduler.active_requests = [req]

        _run_scheduler_step(scheduler, engine)

        assert req.finished is True
        assert req.future.done()
        assert first_token_text not in req.future.result()
        assert scheduler.active_requests == []


class TestStreamManager:
    """Unit tests for the stream manager and decoding"""

    @pytest.mark.asyncio
    async def test_stream_response_handles_multi_byte_characters(self):
        """Test that stream manager decodes multi-byte tokens across boundaries correctly."""
        try:
            from streaming.stream_manager import stream_response
            from scheduler.request import InferenceRequest
            from tokenizer.tokenizer_service import tokenizer_service
        except ImportError:
            pytest.skip("Required modules not available")

        # Make sure tokenizer is loaded to decode properly
        tokenizer_service.load()

        # Emoji test
        text = "Hi 😊"
        tokens = tokenizer_service.encode(text)
        assert len(tokens) > 1

        req = InferenceRequest(prompt="test prompt", max_tokens=10, temperature=0.7)

        for token in tokens:
            req.queue.put_nowait(token)
        req.queue.put_nowait("[DONE]")

        yielded_slices = []
        async for chunk in stream_response(req):
            yielded_slices.append(chunk)

        reconstructed = "".join(yielded_slices)
        assert "Hi" in reconstructed
        assert "\ufffd" not in reconstructed

    @pytest.mark.asyncio
    async def test_stream_response_trims_output_at_stop_sequence(self):
        """A stop sequence -- and anything decoded after it -- must never reach the client."""
        try:
            from streaming.stream_manager import stream_response
            from scheduler.request import InferenceRequest
            from tokenizer.tokenizer_service import tokenizer_service
        except ImportError:
            pytest.skip("Required modules not available")

        tokenizer_service.load()

        text = "Hello there user: this should never appear"
        tokens = tokenizer_service.encode(text)

        req = InferenceRequest(
            prompt="test prompt", max_tokens=50, temperature=0.7, stop_sequences=["user:"]
        )

        for token in tokens:
            req.queue.put_nowait(token)
        req.queue.put_nowait("[DONE]")

        yielded_slices = []
        async for chunk in stream_response(req):
            yielded_slices.append(chunk)

        reconstructed = "".join(yielded_slices)
        assert "Hello" in reconstructed
        assert "user:" not in reconstructed
        assert "this should never appear" not in reconstructed


class TestAPIStructure:
    """Unit tests for API structure"""

    def test_import_routes(self):
        """Test that routes can be imported"""
        try:
            from api.routes import router
            assert router is not None
        except ImportError:
            pytest.skip("API routes not available")

    def test_import_server(self):
        """Test that server app can be imported"""
        try:
            from api.server import app
            assert app is not None
        except ImportError:
            pytest.skip("API server not available")


class TestSettings:
    """Unit tests for settings and configuration"""

    def test_import_model_settings(self):
        """Test that model settings can be imported"""
        try:
            from settings.settings import model_settings
            assert model_settings is not None
        except ImportError:
            pytest.skip("Settings not available")

    def test_import_logging_settings(self):
        """Test that logging settings can be imported"""
        try:
            from settings.settings import logging_settings
            assert logging_settings is not None
        except ImportError:
            pytest.skip("Settings not available")

    def test_settings_have_required_attributes(self):
        """Test that settings have expected attributes"""
        try:
            from settings.settings import model_settings
            # Check for common model settings
            assert hasattr(model_settings, 'top_k') or \
                   hasattr(model_settings, 'top_p') or \
                   hasattr(model_settings, 'model_name')
        except ImportError:
            pytest.skip("Settings not available")


class TestLogger:
    """Unit tests for logging"""

    def test_logger_setup(self):
        """Test that logger can be set up"""
        try:
            from logger import setup_logger
            logger = setup_logger(__name__)
            assert logger is not None
        except ImportError:
            pytest.skip("Logger not available")

    def test_logger_methods(self):
        """Test that logger has expected methods"""
        try:
            from logger import setup_logger
            logger = setup_logger(__name__)
            assert callable(getattr(logger, 'debug', None))
            assert callable(getattr(logger, 'info', None))
            assert callable(getattr(logger, 'warning', None))
            assert callable(getattr(logger, 'error', None))
        except ImportError:
            pytest.skip("Logger not available")
