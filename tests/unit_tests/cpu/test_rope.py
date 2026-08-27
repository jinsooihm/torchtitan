# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import math
import unittest
from unittest.mock import patch

import torch
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    GQAttention,
    QKVLinear,
    VarlenAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rope import (
    _maybe_check_max_pos,
    _yarn_inv_freq,
    ComplexRoPE,
    CosSinRoPE,
    RoPE,
)
from torchtitan.models.qwen3_5.rope import MRoPE


class TestApplyRotaryEmbCosSin(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.num_tokens = 16
        self.n_heads = 4
        self.head_dim = 64
        self.xq = torch.randn(
            self.num_tokens, self.n_heads, self.head_dim, dtype=torch.bfloat16
        )
        self.xk = torch.randn(
            self.num_tokens, self.n_heads, self.head_dim, dtype=torch.bfloat16
        )
        self.rope_cache = torch.randn(
            self.num_tokens, 1, self.head_dim * 2, dtype=torch.float32
        )
        self.rope = CosSinRoPE(
            CosSinRoPE.Config(dim=self.head_dim, max_context_length=self.num_tokens)
        )

    def test_output_dtype_matches_input(self):
        xq_out, xk_out = self.rope.apply_rotary_emb(
            self.xq,
            self.xk,
            self.rope_cache,
        )
        self.assertEqual(xq_out.dtype, self.xq.dtype)
        self.assertEqual(xk_out.dtype, self.xk.dtype)

    def test_output_shape_matches_input(self):
        xq_out, xk_out = self.rope.apply_rotary_emb(
            self.xq,
            self.xk,
            self.rope_cache,
        )
        self.assertEqual(xq_out.shape, self.xq.shape)
        self.assertEqual(xk_out.shape, self.xk.shape)

    def test_computes_in_fp32(self):
        """Output must match a reference computed entirely in float32.

        Ensures inductor cannot fuse away the fp32 upcast when compiling
        adjacent ops (e.g. q_norm/k_norm) with the RoPE computation.
        """
        xq_out, xk_out = self.rope.apply_rotary_emb(
            self.xq,
            self.xk,
            self.rope_cache,
        )

        cos = self.rope_cache[..., : self.head_dim]
        sin = self.rope_cache[..., self.head_dim :]

        def rotate_half(x):
            half = x.shape[-1] // 2
            return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

        xq_ref = (
            (self.xq.float() * cos) + (rotate_half(self.xq.float()) * sin)
        ).bfloat16()
        xk_ref = (
            (self.xk.float() * cos) + (rotate_half(self.xk.float()) * sin)
        ).bfloat16()

        self.assertEqual((xq_out - xq_ref).abs().max().item(), 0.0)
        self.assertEqual((xk_out - xk_ref).abs().max().item(), 0.0)


class TestMaybeCheckMaxPos(unittest.TestCase):
    """Tests for the _maybe_check_max_pos bounds check."""

    def test_positions_within_bounds(self):
        positions = torch.tensor([0, 1, 2, 3])
        _maybe_check_max_pos(positions, max_valid_pos=3)

    def test_positions_at_boundary(self):
        positions = torch.tensor([0, 5, 10, 15])
        _maybe_check_max_pos(positions, max_valid_pos=15)

    def test_positions_out_of_bounds_raises(self):
        positions = torch.tensor([0, 1, 2, 16])
        with self.assertRaises(RuntimeError):
            _maybe_check_max_pos(positions, max_valid_pos=15)
            torch.cuda.synchronize() if torch.cuda.is_available() else None


class TestRoPEPositionBoundsComplex(unittest.TestCase):
    """RoPE complex-format apply must reject out-of-range positions."""

    def setUp(self):
        torch.manual_seed(42)
        self.head_dim = 64
        self.max_context_length = 32
        rope_cfg = ComplexRoPE.Config(
            dim=self.head_dim, max_context_length=self.max_context_length
        )
        self.rope = rope_cfg.build()
        self.assertIsInstance(self.rope, ComplexRoPE)

    def test_valid_positions(self):
        num_tokens = 16
        xq = torch.randn(num_tokens, 4, self.head_dim)
        xk = torch.randn(num_tokens, 4, self.head_dim)
        positions = torch.arange(num_tokens) % 8
        self.rope(xq, xk, positions)

    def test_out_of_range_positions_raises(self):
        num_tokens = 4
        xq = torch.randn(num_tokens, 4, self.head_dim)
        xk = torch.randn(num_tokens, 4, self.head_dim)
        positions = torch.tensor(
            [0, 1, self.max_context_length, self.max_context_length + 1]
        )
        with self.assertRaises(RuntimeError):
            self.rope(xq, xk, positions)


class TestRoPEPositionBoundsCosSin(unittest.TestCase):
    """RoPE cos/sin-format apply must reject out-of-range positions."""

    def setUp(self):
        torch.manual_seed(42)
        self.head_dim = 64
        self.max_context_length = 32
        rope_cfg = CosSinRoPE.Config(
            dim=self.head_dim, max_context_length=self.max_context_length
        )
        self.rope = rope_cfg.build()
        self.assertIsInstance(self.rope, CosSinRoPE)

    def test_valid_positions(self):
        num_tokens = 16
        xq = torch.randn(num_tokens, 4, self.head_dim)
        xk = torch.randn(num_tokens, 4, self.head_dim)
        positions = torch.arange(num_tokens) % 8
        self.rope(xq, xk, positions)

    def test_out_of_range_positions_raises(self):
        num_tokens = 4
        xq = torch.randn(num_tokens, 4, self.head_dim)
        xk = torch.randn(num_tokens, 4, self.head_dim)
        positions = torch.tensor(
            [0, 1, self.max_context_length, self.max_context_length + 1]
        )
        with self.assertRaises(RuntimeError):
            self.rope(xq, xk, positions)


class TestMRoPECache(unittest.TestCase):
    def test_rejects_invalid_sections(self):
        for sections, error in (
            ([2, 1], "must have 3 entries"),
            ([4, 3, -1], "must be non-negative"),
            ([1, 1, 1], "must sum to dim // 2"),
        ):
            with self.subTest(sections=sections):
                with self.assertRaisesRegex(ValueError, error):
                    MRoPE.Config(
                        dim=12,
                        max_context_length=8,
                        mrope_section=sections,
                    ).build()

    def test_rejects_invalid_position_width(self):
        num_tokens, head_dim = 2, 12
        rope = MRoPE.Config(
            dim=head_dim,
            max_context_length=8,
            mrope_section=[2, 2, 2],
        ).build()
        x = torch.randn(num_tokens, 1, head_dim)

        for width in (2, 4):
            with self.subTest(width=width):
                with self.assertRaisesRegex(ValueError, "must have shape"):
                    rope(x, x, torch.zeros(num_tokens, width, dtype=torch.long))

    def test_forward_accepts_three_axis_positions(self):
        torch.manual_seed(42)
        num_tokens, n_heads = 6, 4
        head_dim = 12
        rope = MRoPE.Config(
            dim=head_dim,
            max_context_length=8,
            mrope_section=[2, 2, 2],
        ).build()
        # (tokens, 3): per-token [temporal, height, width] positions.
        position_ids = torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
                [4, 5, 6],
                [5, 6, 7],
            ]
        )
        xq = torch.randn(num_tokens, n_heads, head_dim)
        xk = torch.randn(num_tokens, n_heads, head_dim)

        xq_out, xk_out = rope(xq, xk, position_ids)

        self.assertEqual(xq_out.shape, xq.shape)
        self.assertEqual(xk_out.shape, xk.shape)


class TestPrepareRoPECache(unittest.TestCase):
    def test_cossin_config_prepare_cache_matches_forward_reshape(self):
        cfg = CosSinRoPE.Config(dim=8, max_context_length=16)
        rope = cfg.build()
        positions = torch.tensor([0, 2, 4, 6])
        query = torch.randn(4, 2, 8)

        expected = rope._reshape_cache(query, positions)
        actual = cfg.prepare_cache(
            rope.cache,
            positions=positions,
            num_tokens=query.shape[0],
        )

        torch.testing.assert_close(actual, expected)

    def test_complex_config_prepare_cache_matches_forward_reshape(self):
        cfg = ComplexRoPE.Config(dim=8, max_context_length=16)
        rope = cfg.build()
        positions = torch.tensor([0, 2, 4, 6])
        query = torch.randn(4, 2, 8)

        expected = rope._reshape_cache(query, positions)
        actual = cfg.prepare_cache(
            rope.cache,
            positions=positions,
            num_tokens=query.shape[0],
        )

        torch.testing.assert_close(actual, expected)

    def test_prepare_cache_rejects_length_mismatch(self):
        cfg = ComplexRoPE.Config(dim=8, max_context_length=16)
        rope = cfg.build()

        with self.assertRaisesRegex(ValueError, "num_tokens"):
            cfg.prepare_cache(
                rope.cache,
                positions=torch.tensor([0, 1, 2]),
                num_tokens=4,
            )

    def test_mrope_config_prepare_cache_matches_forward_reshape(self):
        cfg = MRoPE.Config(
            dim=12,
            max_context_length=32,
            mrope_section=[2, 2, 2],
        )
        rope = cfg.build()
        positions = torch.tensor(
            [[0, 0, 0], [1, 2, 3], [2, 4, 6], [3, 6, 9]],
            dtype=torch.int32,
        )
        query = torch.randn(4, 2, 12)

        expected = rope._reshape_cache(query, positions)
        actual = cfg.prepare_cache(
            rope.cache,
            positions=positions,
            num_tokens=query.shape[0],
        )

        torch.testing.assert_close(actual, expected)


class TestYaRNScaling(unittest.TestCase):
    """YaRN follows the explicit scaling policy, not the cache length.

    The cache is sized to the training sequence length, which can be shorter
    than ``original_seq_len`` (e.g. fine-tuning a YaRN checkpoint on short
    sequences), so it must not decide whether YaRN applies.
    """

    def test_zero_lower_correction_boundary(self):
        dim = 128
        rope_factor = 40.0
        inv_freq = _yarn_inv_freq(
            dim=dim,
            base=10000.0,
            rope_factor=rope_factor,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=64,
            truncate=True,
        )
        unscaled_inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

        self.assertEqual(inv_freq.shape, (dim // 2,))
        torch.testing.assert_close(inv_freq[0], unscaled_inv_freq[0])
        torch.testing.assert_close(inv_freq[17], unscaled_inv_freq[17] / rope_factor)

    def test_complex_rope_applies_below_original_sequence_length(self):
        yarn = ComplexRoPE.Config(
            dim=64,
            max_context_length=2048,
            scaling="yarn",
            rope_factor=40.0,
            original_seq_len=4096,
        ).build()
        unscaled = ComplexRoPE.Config(dim=64, max_context_length=2048).build()

        self.assertFalse(torch.equal(yarn.cache[1], unscaled.cache[1]))

    def test_deepseek_mscale_applies_below_original_sequence_length(self):
        from torchtitan.models.deepseek_v3 import deepseekv3_configs
        from torchtitan.models.deepseek_v3.model import Attention

        model_config = deepseekv3_configs["debugmodel"]("flex", "standard")
        attention_config = model_config.layers[0].attention
        assert isinstance(attention_config, Attention.Config)
        attention_config.rope = dataclasses.replace(
            attention_config.rope,
            max_context_length=attention_config.rope.original_seq_len // 2,
        )
        attention = attention_config.build()

        expected_mscale = (
            0.1 * attention_config.mscale * math.log(attention_config.rope.rope_factor)
            + 1.0
        )
        expected_softmax_scale = attention.qk_head_dim**-0.5 * expected_mscale**2
        self.assertAlmostEqual(attention.softmax_scale, expected_softmax_scale)


class TestDecoderRoPEContract(unittest.TestCase):
    def test_llama_debugmodel_has_single_active_rope_contract(self):
        from torchtitan.models.common.decoder import _shared_decoder_rope_config
        from torchtitan.models.llama3 import llama3_configs

        cfg = llama3_configs["debugmodel"]("flex")
        rope_cfg = _shared_decoder_rope_config(cfg)

        self.assertIsNotNone(rope_cfg)
        self.assertEqual(rope_cfg, cfg.layers[0].attention.rope)

    def test_muse_glimmer_ignores_inactive_nope_layers(self):
        from torchtitan.models.common.decoder import _shared_decoder_rope_config
        from torchtitan.models.muse_glimmer import muse_glimmer_configs

        cfg = muse_glimmer_configs["debugmodel"]("flex")
        rope_cfg = _shared_decoder_rope_config(cfg)
        active_ropes = [
            layer.attention.rope
            for layer in cfg.layers
            if getattr(layer.attention, "use_rope", True)
        ]

        self.assertIsNotNone(rope_cfg)
        self.assertTrue(active_ropes)
        self.assertTrue(all(rope == rope_cfg for rope in active_ropes))

    def test_mixed_active_rope_contracts_raise(self):
        from types import SimpleNamespace

        from torchtitan.models.common.decoder import _shared_decoder_rope_config

        cfg = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    attention=SimpleNamespace(
                        rope=ComplexRoPE.Config(dim=8, max_context_length=16)
                    )
                ),
                SimpleNamespace(
                    attention=SimpleNamespace(
                        rope=CosSinRoPE.Config(dim=8, max_context_length=16)
                    )
                ),
            ],
            mtp_layers=[],
        )

        with self.assertRaisesRegex(ValueError, "one active decoder RoPE contract"):
            _shared_decoder_rope_config(cfg)

    def test_mtp_layers_are_included_in_contract(self):
        from types import SimpleNamespace

        from torchtitan.models.common.decoder import _shared_decoder_rope_config

        rope = ComplexRoPE.Config(dim=8, max_context_length=16)
        cfg = SimpleNamespace(
            layers=[SimpleNamespace(attention=SimpleNamespace(rope=rope))],
            mtp_layers=[SimpleNamespace(attention=SimpleNamespace(rope=rope))],
        )

        self.assertEqual(_shared_decoder_rope_config(cfg), rope)


class TestPerLayerRoPECache(unittest.TestCase):
    def test_gqa_attention_uses_layer_rope_cache(self):
        torch.manual_seed(42)
        dim = 8
        head_dim = 4
        attention = GQAttention.Config(
            n_heads=2,
            n_kv_heads=2,
            head_dim=head_dim,
            dim=dim,
            qkv_linear=QKVLinear.Config(
                head_dim=head_dim,
                wq=Linear.Config(in_features=dim, out_features=dim),
                wkv=Linear.Config(in_features=dim, out_features=dim),
            ),
            wo=Linear.Config(in_features=dim, out_features=dim),
            inner_attention=VarlenAttention.Config(),
            rope=ComplexRoPE.Config(dim=head_dim, max_context_length=16),
        ).build()

        x = torch.randn(8, dim)
        positions = torch.arange(8)
        attention_masks = create_varlen_metadata_for_document(positions)

        with patch(
            "torchtitan.models.common.attention._varlen_attn",
            side_effect=lambda q, k, v, *args, **kwargs: q,
        ):
            out = attention(x, attention_masks, positions)

        self.assertIsNotNone(attention.rope)
        self.assertEqual(out.shape, x.shape)

    def test_decoder_builds_distinct_rope_modules_per_attention_layer(self):
        from torchtitan.models.llama3 import llama3_configs

        model = llama3_configs["debugmodel"]("flex").build()
        layer_ropes = [layer.attention.rope for layer in model.layers.values()]

        self.assertTrue(all(isinstance(rope, RoPE) for rope in layer_ropes))
        self.assertEqual(len({id(rope) for rope in layer_ropes}), len(layer_ropes))

    def test_decoder_builds_distinct_rope_configs_per_attention_layer(self):
        from torchtitan.models.llama3 import llama3_configs

        cfg = llama3_configs["debugmodel"]("flex")
        layer_rope_cfgs = [layer.attention.rope for layer in cfg.layers]

        self.assertEqual(
            len({id(rope_cfg) for rope_cfg in layer_rope_cfgs}),
            len(layer_rope_cfgs),
        )

    def test_decoder_rope_layers_reference_model_owned_cache(self):
        from torchtitan.models.llama3 import llama3_configs

        model = llama3_configs["debugmodel"]("flex").build()
        layer_ropes = [layer.attention.rope for layer in model.layers.values()]

        self.assertTrue(hasattr(model, "rope_cache"))
        self.assertTrue(all(rope.cache is model.rope_cache for rope in layer_ropes))

    def test_decoder_init_states_reinitializes_and_reties_root_rope_cache(self):
        from torchtitan.models.llama3 import llama3_configs

        model = llama3_configs["debugmodel"]("flex").build()
        model.rope_cache.zero_()
        model.init_states(buffer_device=torch.device("cpu"))

        expected = model._rope_config.build().cache
        torch.testing.assert_close(model.rope_cache, expected)
        for layer in model.layers.values():
            self.assertIs(layer.attention.rope.cache, model.rope_cache)

    def test_mtp_layers_reference_model_owned_cache_after_build(self):
        from torchtitan.models.deepseek_v3 import model_registry

        model = model_registry("debugmodel", num_mtp_layers=1).model.build()

        self.assertIsNotNone(model.mtp_layers)
        for layer in model.layers.values():
            self.assertIs(layer.attention.rope.cache, model.rope_cache)
        for layer in model.mtp_layers:
            self.assertIs(layer.attention.rope.cache, model.rope_cache)

    def test_decoder_root_rope_cache_has_state_sharding(self):
        from torchtitan.models.common.decoder_sharding import (
            set_decoder_sharding_config,
        )
        from torchtitan.models.llama3 import llama3_configs

        cfg = llama3_configs["debugmodel"]("flex")
        set_decoder_sharding_config(cfg, enable_sp=False)

        self.assertIsNotNone(cfg.sharding_config)
        self.assertIn("rope_cache", cfg.sharding_config.state_shardings)

    def test_decoder_sharding_leaves_layer_rope_caches_unsharded(self):
        from torchtitan.models.llama3 import llama3_configs
        from torchtitan.models.llama3.sharding import set_llama3_sharding_config

        cfg = llama3_configs["debugmodel"]("flex")
        set_llama3_sharding_config(cfg, enable_sp=False)

        for layer_cfg in cfg.layers:
            self.assertIsNone(layer_cfg.attention.rope.sharding_config)

    def test_decoder_parallelize_reties_model_owned_rope_cache(self):
        from unittest.mock import patch

        import spmd_types as spmd

        from torchtitan.distributed.parallel_dims import ParallelDims
        from torchtitan.models.common.decoder_sharding import dense_param_placement
        from torchtitan.models.llama3 import llama3_configs
        from torchtitan.protocols.sharding import ShardingConfig

        model = llama3_configs["debugmodel"]("flex").build()
        old_rope_cache = model.rope_cache
        new_rope_cache = torch.empty_like(old_rope_cache)
        model._sharding_config = ShardingConfig(
            state_shardings={"rope_cache": dense_param_placement(tp=spmd.R)}
        )
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
        )

        def _replace_rope_cache(module, parallel_dims):
            del parallel_dims
            module.register_buffer(
                "rope_cache",
                new_rope_cache,
                persistent=False,
            )

        with (
            patch.object(
                type(model),
                "_distribute_states",
                autospec=True,
                side_effect=_replace_rope_cache,
            ),
            patch.object(ParallelDims, "get_optional_mesh", return_value=None),
        ):
            model.parallelize(parallel_dims)

        self.assertIsNot(model.rope_cache, old_rope_cache)
        self.assertIs(model.rope_cache, new_rope_cache)
        for layer in model.layers.values():
            self.assertIs(layer.attention.rope.cache, model.rope_cache)

    def test_decoder_preprocess_inputs_returns_prepared_rope_cache(self):
        from torchtitan.config import ParallelismConfig
        from torchtitan.distributed.parallel_dims import ParallelDims
        from torchtitan.models.llama3 import llama3_configs

        model = llama3_configs["debugmodel"]("flex").build()
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
        )
        parallelism = ParallelismConfig(spmd_backend="partial_dtensor")
        positions = torch.arange(8, dtype=torch.int32)
        input_dict = {
            "input": torch.randint(0, 100, (8,)),
            "labels": torch.zeros(8, dtype=torch.long),
            "positions": positions,
        }

        _inputs, _labels, batch = model.preprocess_inputs(
            input_dict,
            parallel_dims=parallel_dims,
            parallelism=parallelism,
        )

        self.assertIn("rope_cache", batch)
        expected = model._rope_config.prepare_cache(
            model.rope_cache,
            positions=positions,
            num_tokens=positions.shape[0],
        )
        torch.testing.assert_close(batch["rope_cache"], expected)

    def test_decoder_preprocess_inputs_does_not_need_local_rope_layers(self):
        from torchtitan.config import ParallelismConfig
        from torchtitan.distributed.parallel_dims import ParallelDims
        from torchtitan.models.llama3 import llama3_configs
        from torchtitan.protocols.module import ModuleDict

        model = llama3_configs["debugmodel"]("flex").build()
        model.layers = ModuleDict()
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
        )
        parallelism = ParallelismConfig(spmd_backend="partial_dtensor")
        positions = torch.arange(8, dtype=torch.int32)

        _inputs, _labels, batch = model.preprocess_inputs(
            {
                "input": torch.randint(0, 100, (8,)),
                "labels": torch.zeros(8, dtype=torch.long),
                "positions": positions,
            },
            parallel_dims=parallel_dims,
            parallelism=parallelism,
        )

        self.assertIn("rope_cache", batch)
        self.assertEqual(batch["rope_cache"].shape[0], positions.shape[0])

    def test_prepared_rope_cache_has_input_sharding_entry(self):
        from torchtitan.models.common.decoder_sharding import decoder_input_sharding

        self.assertIn("rope_cache", decoder_input_sharding())

    def test_gqa_attention_uses_prepared_rope_cache_when_provided(self):
        torch.manual_seed(42)
        dim = 8
        head_dim = 4
        attention = GQAttention.Config(
            n_heads=2,
            n_kv_heads=2,
            head_dim=head_dim,
            dim=dim,
            qkv_linear=QKVLinear.Config(
                head_dim=head_dim,
                wq=Linear.Config(in_features=dim, out_features=dim),
                wkv=Linear.Config(in_features=dim, out_features=dim),
            ),
            wo=Linear.Config(in_features=dim, out_features=dim),
            inner_attention=VarlenAttention.Config(),
            rope=ComplexRoPE.Config(dim=head_dim, max_context_length=16),
        ).build()
        x = torch.randn(8, dim)
        positions = torch.arange(8)
        attention_masks = create_varlen_metadata_for_document(positions)
        rope_cache = attention.rope.prepare_cache(
            positions=positions,
            num_tokens=positions.shape[0],
        )

        with patch.object(
            attention.rope,
            "_reshape_cache",
            side_effect=AssertionError("_reshape_cache should not run"),
        ):
            with patch(
                "torchtitan.models.common.attention._varlen_attn",
                side_effect=lambda q, k, v, *args, **kwargs: q,
            ):
                out = attention(
                    x,
                    attention_masks,
                    positions,
                    rope_cache=rope_cache,
                )

        self.assertEqual(out.shape, x.shape)

    def test_muse_glimmer_preprocess_inputs_returns_prepared_rope_cache(self):
        from torchtitan.config import ParallelismConfig
        from torchtitan.distributed.parallel_dims import ParallelDims
        from torchtitan.models.muse_glimmer import muse_glimmer_configs

        model = muse_glimmer_configs["debugmodel"]("varlen").build()
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
        )
        parallelism = ParallelismConfig(spmd_backend="partial_dtensor")
        positions = torch.arange(8, dtype=torch.int32)

        _inputs, _labels, batch = model.preprocess_inputs(
            {
                "input": torch.randint(0, 100, (8,)),
                "labels": torch.zeros(8, dtype=torch.long),
                "positions": positions,
            },
            parallel_dims=parallel_dims,
            parallelism=parallelism,
        )

        self.assertIn("rope_cache", batch)
        self.assertEqual(batch["rope_cache"].shape[0], positions.shape[0])

    def test_kimi_k2_7_preprocess_inputs_returns_prepared_rope_cache(self):
        try:
            from torchtitan.config import ParallelismConfig
            from torchtitan.distributed.parallel_dims import ParallelDims
            from torchtitan.models.kimi_k2_7 import model_registry
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(
                f"Kimi K2.7 optional dependency unavailable: {exc.name}"
            ) from exc

        model = model_registry("debugmodel").model.build()
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
        )
        parallelism = ParallelismConfig(spmd_backend="partial_dtensor")
        positions = torch.arange(8, dtype=torch.int32)

        _inputs, _labels, batch = model.preprocess_inputs(
            {
                "input": torch.randint(0, 100, (8,)),
                "labels": torch.zeros(8, dtype=torch.long),
                "positions": positions,
            },
            parallel_dims=parallel_dims,
            parallelism=parallelism,
        )

        self.assertIn("rope_cache", batch)
        self.assertEqual(batch["rope_cache"].shape[0], positions.shape[0])

    def test_pipeline_split_stage_without_layers_can_prepare_rope_cache(self):
        from torchtitan.config import ParallelismConfig
        from torchtitan.distributed.parallel_dims import ParallelDims
        from torchtitan.distributed.pipeline_parallel import _split_module
        from torchtitan.models.llama3 import llama3_configs

        model = llama3_configs["debugmodel"]("flex").build()
        stage = _split_module(model, ["tok_embeddings"])
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
        )
        parallelism = ParallelismConfig(spmd_backend="partial_dtensor")
        positions = torch.arange(8, dtype=torch.int32)

        _inputs, _labels, batch = stage.preprocess_inputs(
            {
                "input": torch.randint(0, 100, (8,)),
                "labels": torch.zeros(8, dtype=torch.long),
                "positions": positions,
            },
            parallel_dims=parallel_dims,
            parallelism=parallelism,
        )

        self.assertIn("rope_cache", batch)
        self.assertEqual(len(stage.layers), 0)
        self.assertEqual(batch["rope_cache"].shape[0], positions.shape[0])


class TestUpdateFromConfigSeqLenValidation(unittest.TestCase):
    """Reject training contexts larger than the RoPE context length."""

    def _make_trainer_config(self, seq_len):
        from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
        from torchtitan.trainer import Trainer

        return Trainer.Config(
            training=dataclasses.replace(
                TrainingConfig(),
                num_tokens_per_microbatch_per_dp_rank=seq_len,
                max_context_length=seq_len,
            ),
            parallelism=ParallelismConfig(),
            debug=DebugConfig(),
        )

    def _make_config(self):
        """Build a minimal Llama3 debug config."""
        from torchtitan.models.llama3 import llama3_configs

        return llama3_configs["debugmodel"]("flex")

    def test_rejects_oversized_seq_len(self):
        cfg = self._make_config()
        rope_max = cfg.max_context_length
        with self.assertRaises(ValueError):
            cfg.update_from_config(config=self._make_trainer_config(rope_max + 1))

    def test_accepts_valid_seq_len(self):
        cfg = self._make_config()
        rope_max = cfg.max_context_length
        cfg.update_from_config(config=self._make_trainer_config(rope_max))
        self.assertEqual(cfg.max_context_length, rope_max)

    def test_vllm_max_model_len_as_seq_len(self):
        """vLLM wrapper translates max_model_len to TrainingConfig.max_context_length.

        When the training and RoPE context lengths match, the RoPE cache stays
        at the model's intrinsic maximum.
        """
        cfg = self._make_config()
        original_max = cfg.max_context_length
        cfg.update_from_config(config=self._make_trainer_config(original_max))
        self.assertEqual(cfg.max_context_length, original_max)


if __name__ == "__main__":
    unittest.main()
