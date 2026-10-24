"""Regression tests for allocation-time parking of chunked prefills."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from sglang.srt.mem_cache import allocation as allocation_mod
from sglang.srt.mem_cache.allocation import (
    PrefillAllocationError,
    alloc_paged_token_slots_extend,
)


class _FailingAllocator:
    page_size = 4

    def get_extend_allocation_demand(
        self,
        prefix_lens_cpu,
        seq_lens_cpu,
        *,
        conservative_num_tokens,
        shard_size,
    ):
        return conservative_num_tokens

    def alloc_extend(self, *args, **kwargs):
        return None


class _TreeCache:
    def __init__(self):
        self.token_to_kv_pool_allocator = _FailingAllocator()

    def supports_prefix_sharing(self):
        return False

    def available_and_evictable_str(self):
        return "available=0"

    def pretty_print(self):
        pass


class TestChunkedPrefillAllocationFailure(unittest.TestCase):
    def test_allocator_failure_has_typed_exception(self):
        cache = _TreeCache()
        values = torch.tensor([0], dtype=torch.int64)
        with self.assertRaises(PrefillAllocationError):
            alloc_paged_token_slots_extend(
                cache,
                values,
                values,
                torch.tensor([4], dtype=torch.int64),
                torch.tensor([4], dtype=torch.int64),
                torch.tensor([-1], dtype=torch.int64),
                extend_num_tokens=4,
            )

    def test_extend_allocation_failure_releases_fresh_resources(self):
        req = SimpleNamespace(
            prefix_len=0,
            extend_end=4,
            kv=SimpleNamespace(
                req_pool_idx=None,
                mamba_pool_idx=torch.tensor([3]),
                holds_kv=False,
                holds_mamba=True,
            ),
        )
        pool = MagicMock()
        tree_cache = SimpleNamespace(
            page_size=4,
            token_to_kv_pool_allocator=SimpleNamespace(page_size=4),
        )
        batch = SimpleNamespace(
            reqs=[req],
            device="cpu",
            tree_cache=tree_cache,
            req_to_token_pool=pool,
            hisparse_coordinator=None,
            prefix_lens=[0],
            extend_lens=[4],
            seq_lens=torch.tensor([4]),
            seq_lens_cpu=torch.tensor([4]),
            extend_num_tokens=4,
            is_dllm=lambda: False,
            maybe_evict_swa=lambda: None,
        )

        def fake_alloc_req_slots(req_to_token_pool, reqs, tree_cache):
            reqs[0].kv.req_pool_idx = 7
            reqs[0].kv.holds_kv = True
            return [7]

        with (
            unittest.mock.patch.object(
                allocation_mod, "_prefix_kv_indices", return_value=torch.empty(0)
            ),
            unittest.mock.patch.object(
                allocation_mod, "alloc_req_slots", side_effect=fake_alloc_req_slots
            ),
            unittest.mock.patch.object(
                allocation_mod,
                "alloc_paged_token_slots_extend",
                side_effect=PrefillAllocationError("full"),
            ),
            self.assertRaises(PrefillAllocationError),
        ):
            allocation_mod.alloc_for_extend(batch)

        pool.free_mamba_cache.assert_called_once_with(req)
        pool.free.assert_called_once_with(req)


if __name__ == "__main__":
    unittest.main()
