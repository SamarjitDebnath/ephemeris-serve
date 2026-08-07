from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import torch


@dataclass
class BlockTable:
    """Tracks which blocks in a `PagedKVCache`'s pool belong to one request."""
    block_ids: List[int] = field(default_factory=list)
    length: int = 0  # real (unpadded) token count stored so far


class PagedKVCache:
    """Block-based KV cache storage for the continuous scheduler.

    Pure PyTorch indexing -- no fused kernel, works identically on
    MPS/CPU/CUDA. This is a storage/allocation optimization only: it
    replaces the realloc-and-copy of a monolithically growing per-request
    tensor with writes into pre-allocated fixed-size blocks. Every step
    still materializes a dense `(batch, heads, seq, head_dim)` tensor via
    `gather_dense()` for the model's forward pass -- there is no fused
    kernel available on MPS to read scattered blocks directly.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        initial_capacity_blocks: int = 64,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = device

        self.capacity = 0
        self.key_pool: List[torch.Tensor] = []
        self.value_pool: List[torch.Tensor] = []
        self.free_blocks: List[int] = []
        self._grow_pool(max(initial_capacity_blocks, 1))

    def _grow_pool(self, new_blocks: int) -> None:
        for layer_idx in range(self.num_layers):
            new_k = torch.zeros(
                (new_blocks, self.num_kv_heads, self.block_size, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            new_v = torch.zeros_like(new_k)
            if layer_idx >= len(self.key_pool):
                self.key_pool.append(new_k)
                self.value_pool.append(new_v)
            else:
                self.key_pool[layer_idx] = torch.cat([self.key_pool[layer_idx], new_k], dim=0)
                self.value_pool[layer_idx] = torch.cat([self.value_pool[layer_idx], new_v], dim=0)
        self.free_blocks.extend(range(self.capacity, self.capacity + new_blocks))
        self.capacity += new_blocks

    def _ensure_free(self, n_blocks: int) -> None:
        if len(self.free_blocks) < n_blocks:
            # Double the pool (or grow by exactly what's needed if that's
            # larger), same doubling strategy as the block pool itself, so
            # growth is amortized rather than repeated one block at a time.
            grow_by = max(self.capacity, n_blocks - len(self.free_blocks))
            self._grow_pool(grow_by)

    def allocate(self, table: BlockTable, n_tokens: int) -> None:
        """Ensure `table` has enough blocks for `n_tokens` more real tokens."""
        total_tokens = table.length + n_tokens
        needed_blocks = -(-total_tokens // self.block_size)  # ceil division
        blocks_to_add = needed_blocks - len(table.block_ids)
        if blocks_to_add > 0:
            self._ensure_free(blocks_to_add)
            for _ in range(blocks_to_add):
                table.block_ids.append(self.free_blocks.pop())

    def append(
        self,
        table: BlockTable,
        keys_per_layer: Sequence[torch.Tensor],
        values_per_layer: Sequence[torch.Tensor],
    ) -> None:
        """Append this step's new K/V for every layer, in one atomic call.

        `keys_per_layer[l]` / `values_per_layer[l]` must have shape
        `(num_kv_heads, n_new, head_dim)`. Allocates blocks as needed.
        """
        if len(keys_per_layer) != self.num_layers or len(values_per_layer) != self.num_layers:
            raise ValueError("keys_per_layer/values_per_layer must have one entry per layer")

        n_new = keys_per_layer[0].shape[1]
        if n_new == 0:
            return
        self.allocate(table, n_new)

        pos = table.length
        written = 0
        while written < n_new:
            block_idx_in_table = pos // self.block_size
            offset = pos % self.block_size
            block_id = table.block_ids[block_idx_in_table]
            take = min(self.block_size - offset, n_new - written)

            for layer_idx in range(self.num_layers):
                self.key_pool[layer_idx][block_id, :, offset:offset + take, :] = (
                    keys_per_layer[layer_idx][:, written:written + take, :]
                )
                self.value_pool[layer_idx][block_id, :, offset:offset + take, :] = (
                    values_per_layer[layer_idx][:, written:written + take, :]
                )

            written += take
            pos += take

        table.length += n_new

    def gather_dense(
        self, tables: Sequence[BlockTable]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
        """Materialize a left-padded, batched dense view for the model's forward pass.

        Returns `(keys_per_layer, values_per_layer, real_lengths)` where
        each `keys_per_layer[l]`/`values_per_layer[l]` has shape
        `(batch, num_kv_heads, max_len, head_dim)`, left-padded per row to
        the batch's longest real length. `real_lengths[i]` is request
        `i`'s true (unpadded) token count -- the metadata prefill/decode
        mixing needs to build correct attention masks and position ids.
        """
        real_lengths = [t.length for t in tables]
        max_len = max(real_lengths) if real_lengths else 0

        keys_per_layer: List[torch.Tensor] = []
        values_per_layer: List[torch.Tensor] = []

        for layer_idx in range(self.num_layers):
            rows_k = []
            rows_v = []
            for table, real_len in zip(tables, real_lengths):
                k, v = self._gather_row(table, real_len, layer_idx)
                if k.shape[2] < max_len:
                    pad_amt = max_len - k.shape[2]
                    k = torch.nn.functional.pad(k, (0, 0, pad_amt, 0), value=0.0)
                    v = torch.nn.functional.pad(v, (0, 0, pad_amt, 0), value=0.0)
                rows_k.append(k)
                rows_v.append(v)

            if rows_k:
                keys_per_layer.append(torch.cat(rows_k, dim=0))
                values_per_layer.append(torch.cat(rows_v, dim=0))
            else:
                empty = torch.zeros(
                    (0, self.num_kv_heads, max_len, self.head_dim), dtype=self.dtype, device=self.device
                )
                keys_per_layer.append(empty)
                values_per_layer.append(empty.clone())

        return keys_per_layer, values_per_layer, real_lengths

    def _gather_row(self, table: BlockTable, real_len: int, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if real_len == 0:
            empty = torch.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=self.dtype, device=self.device)
            return empty, empty.clone()

        full_blocks = real_len // self.block_size
        remainder = real_len % self.block_size

        parts_k = []
        parts_v = []
        for b in range(full_blocks):
            block_id = table.block_ids[b]
            parts_k.append(self.key_pool[layer_idx][block_id:block_id + 1])
            parts_v.append(self.value_pool[layer_idx][block_id:block_id + 1])
        if remainder > 0:
            block_id = table.block_ids[full_blocks]
            parts_k.append(self.key_pool[layer_idx][block_id:block_id + 1, :, :remainder, :])
            parts_v.append(self.value_pool[layer_idx][block_id:block_id + 1, :, :remainder, :])

        k = parts_k[0] if len(parts_k) == 1 else torch.cat(parts_k, dim=2)
        v = parts_v[0] if len(parts_v) == 1 else torch.cat(parts_v, dim=2)
        return k, v

    def free(self, table: BlockTable) -> None:
        """Release `table`'s blocks back to the free pool and reset it to empty."""
        self.free_blocks.extend(table.block_ids)
        table.block_ids = []
        table.length = 0

    def is_valid(self, table: BlockTable) -> bool:
        """Structural self-check mirroring `_is_valid_dynamic_cache`'s defensive pattern."""
        if not table.block_ids:
            return table.length == 0

        free_set = set(self.free_blocks)
        for block_id in table.block_ids:
            if block_id < 0 or block_id >= self.capacity:
                return False
            if block_id in free_set:
                return False

        min_length = (len(table.block_ids) - 1) * self.block_size + 1
        max_length = len(table.block_ids) * self.block_size
        return min_length <= table.length <= max_length
