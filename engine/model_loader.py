import gc

import torch
from transformers import AutoModelForCausalLM
from settings.settings import model_settings
from tokenizer.tokenizer_service import tokenizer_service
from utils.device_cache import empty_device_cache
from logger import setup_logger
from settings.settings import logging_settings

logger = setup_logger(__name__, level=logging_settings.log_level, log_file=logging_settings.log_file)


class ModelLoader:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            self.model = self._build_model(model_settings.model_name)

    def reload(self, model_name: str) -> None:
        """Swap in a different model, replacing whatever is currently loaded.

        Used for runtime model switching (see `scheduler/model_swap.py`),
        which must only call this once it has confirmed no requests are in
        flight -- an in-flight request's tensors are tied to the model
        object active when it started.

        The new model is fully built before `self.model` is touched, so if
        `_build_model` raises (bad repo id, network error, OOM, ...) the
        previously-loaded model is left running untouched.
        """
        new_model = self._build_model(model_name)
        old_model = self.model
        self.model = new_model
        model_settings.model_name = model_name

        del old_model
        gc.collect()
        empty_device_cache(model_settings.device)

    def _build_model(self, model_name: str):
        device = model_settings.device
        logger.info("Loading model '%s' onto device '%s'", model_name, device)

        # MPS (Apple Silicon) works best with float16.
        # float32 models can be loaded and cast, but large models may OOM in float32 on MPS.
        dtype = torch.float16 if device == "mps" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model.to(device)
        model.eval()
        logger.info(
            "Model loaded successfully on device='%s' dtype='%s' (model_type: %s, vocab_size: %d)",
            device,
            dtype,
            type(model).__name__,
            model.config.vocab_size if hasattr(model, 'config') else 'unknown'
        )
        return model

    def _get_model(self):
        if self.model is None:
            self.load()
        return self.model
    
    def warmup(self):
        model = self._get_model()
        
        if model is None:
            raise RuntimeError("Model failed to load during warmup")

        # Request tensor outputs for model warmup
        tokens = tokenizer_service.encode("Warmup request", return_tensors=True)
        input_ids = tokens["input_ids"].to(model_settings.device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            torch.argmax(logits, dim=-1)

model_loader = ModelLoader()