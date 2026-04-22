import hashlib
import importlib.util
import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def _compile_supported() -> bool:
    # Some nightly/staged torch wheels miss this module needed by torch.compile.
    return (
        hasattr(torch, "compile")
        and importlib.util.find_spec("torch._subclasses.schema_check_mode") is not None
    )


_CAN_COMPILE = _compile_supported()


def _safe_compile_callable(fn, label: str, **compile_kwargs):
    if not _CAN_COMPILE:
        return fn
    try:
        compiled = torch.compile(fn, **compile_kwargs)
    except Exception as exc:
        log.warning("torch.compile unavailable for %s: %s", label, exc)
        return fn

    failed = {"value": False}

    def wrapped(*args, **kwargs):
        if failed["value"]:
            return fn(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except Exception as exc:
            failed["value"] = True
            log.warning("Falling back to eager %s after compile failure: %s", label, exc)
            return fn(*args, **kwargs)

    return wrapped


def _safe_compile_module(module, label: str, **compile_kwargs):
    if not _CAN_COMPILE:
        return module
    try:
        return torch.compile(module, **compile_kwargs)
    except Exception as exc:
        log.warning("torch.compile unavailable for %s: %s", label, exc)
        return module

# ═══════════════════════════════════════════════════════════════════════════
#  Cached-context dataclass (used by the action expert KV cache)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ActionExpertCachedContext:
    contexts: Sequence[Optional[torch.Tensor]]
    cross_mask: Optional[torch.Tensor]
    cached_cross_kvs: torch.Tensor
    encoded_states: Optional[torch.Tensor]
    states_mode: str

# ═══════════════════════════════════════════════════════════════════════════
#  1. Action Expert patches — SDPA, precomputed KV, context caching
# ═══════════════════════════════════════════════════════════════════════════

def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


_compiled_modulate = _safe_compile_callable(_modulate, "_modulate")


def patch_action_expert(model):
    """Replace manual matmul attention with SDPA and add cross-attn KV caching."""
    import olmo.nn.action_expert as _ae_mod
    _ae_mod._modulate = _compiled_modulate

    ae = model.action_expert

    from olmo.nn.action_expert import ActionExpertAttention
    for mod in model.modules():
        if isinstance(mod, ActionExpertAttention):
            mod.forward = _make_ae_attn_forward(mod)
            mod.project_kv = _make_ae_project_kv(mod)

    from olmo.nn.action_expert import ActionExpertBlock, ActionExpertMLP
    for mod in model.modules():
        if isinstance(mod, ActionExpertMLP):
            mod.forward = _safe_compile_callable(mod.forward, f"{type(mod).__name__}.forward")

    for mod in model.modules():
        if isinstance(mod, ActionExpertBlock):
            mod.forward = _make_ae_block_forward(mod)

    for mod in model.modules():
        if isinstance(mod, ActionExpertBlock):
            mod.precompute_cross_kv = lambda ctx, _m=mod: _m.attn2.project_kv(ctx)

    ae.precompute_context = lambda *a, _ae=ae, **kw: _ae_precompute_context(_ae, *a, **kw)
    ae._original_forward = ae.forward
    ae.forward = lambda *a, _ae=ae, **kw: _ae_forward(_ae, *a, **kw)

    log.info("Patched action expert: SDPA + compiled modulate/MLP + KV caching")


def _make_ae_attn_forward(mod):
    def forward(x, kv=None, attn_mask=None, precomputed_kv=None):
        bsz, tgt_len, _ = x.shape
        q = mod.q_proj(x)
        q = q.view(bsz, tgt_len, mod.num_heads, mod.head_dim).transpose(1, 2)

        if precomputed_kv is not None:
            k, v = precomputed_kv
        else:
            if kv is None:
                kv = x
            src_len = kv.shape[1]
            kv_out = mod.kv_proj(kv)
            kv_out = kv_out.view(bsz, src_len, 2, mod.num_heads, mod.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv_out[0], kv_out[1]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(bsz, tgt_len, mod.hidden_size)
        out = mod.proj(out)
        out = mod.proj_drop(out)
        return out
    return forward


def _make_ae_project_kv(mod):
    def project_kv(kv_source):
        bsz, src_len, _ = kv_source.shape
        kv_out = mod.kv_proj(kv_source)
        kv_out = kv_out.view(bsz, src_len, 2, mod.num_heads, mod.head_dim).permute(2, 0, 3, 1, 4)
        return kv_out[0], kv_out[1]
    return project_kv


def _make_ae_block_forward(mod):
    def forward(x, timestep_embed, cross_context, attn_mask=None, cached_cross_kv=None):
        shifts_scales = mod.adaLN_modulation(timestep_embed).chunk(9, dim=1)
        shift_msa, scale_msa, gate_msa = shifts_scales[0], shifts_scales[1], shifts_scales[2]
        shift_mca, scale_mca, gate_mca = shifts_scales[3], shifts_scales[4], shifts_scales[5]
        shift_mlp, scale_mlp, gate_mlp = shifts_scales[6], shifts_scales[7], shifts_scales[8]

        x = x + gate_msa.unsqueeze(1) * mod.attn1(
            _compiled_modulate(mod.norm1(x), shift_msa, scale_msa))
        x = x + gate_mca.unsqueeze(1) * mod.attn2(
            _compiled_modulate(mod.norm2(x), shift_mca, scale_mca),
            kv=cross_context, attn_mask=attn_mask, precomputed_kv=cached_cross_kv)
        x = x + gate_mlp.unsqueeze(1) * mod.mlp(
            _compiled_modulate(mod.norm3(x), shift_mlp, scale_mlp))
        return x
    return forward


def _stack_cross_kvs(cached_cross_kvs: Sequence[Tuple[torch.Tensor, torch.Tensor]]
                     ) -> torch.Tensor:
    ks = torch.stack([k for k, _ in cached_cross_kvs], dim=0).contiguous()
    vs = torch.stack([v for _, v in cached_cross_kvs], dim=0).contiguous()
    return torch.stack((ks, vs), dim=0).contiguous()


def _project_context_layers(ae, encoder_hidden_states: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    if not encoder_hidden_states:
        return []
    batch_size = encoder_hidden_states[0].shape[0]
    # Fuse context projection + norm across all selected backbone layers to cut launch overhead.
    stacked = torch.cat(tuple(encoder_hidden_states), dim=0)
    projected = ae.context_norm(ae.context_proj(stacked))
    return projected.split(batch_size, dim=0)


def _ae_precompute_context(ae, encoder_hidden_states, encoder_attention_mask=None,
                           state_embeddings=None, states_mode="cross_attn"):
    bsz = encoder_hidden_states[0].shape[0]
    ctx_dtype = encoder_hidden_states[0].dtype
    encoded_states = ae._encode_states(state_embeddings)
    all_visible = (
        isinstance(encoder_attention_mask, torch.Tensor)
        and encoder_attention_mask.dtype == torch.bool
        and bool(torch.all(encoder_attention_mask))
    )
    cached_cross_kvs = []
    projected_contexts = _project_context_layers(ae, encoder_hidden_states)
    if states_mode == "self_attn":
        cross_mask = None if all_visible else ae._build_cross_attention_mask(
            encoder_attention_mask, None, bsz, ctx_dtype
        )
        for blk, ctx in zip(ae.blocks, projected_contexts):
            cached_cross_kvs.append(blk.precompute_cross_kv(ctx))
        cached_encoded_states = encoded_states
    else:
        cross_mask = None if all_visible else ae._build_cross_attention_mask(
            encoder_attention_mask, encoded_states, bsz, ctx_dtype
        )
        for blk, ctx in zip(ae.blocks, projected_contexts):
            if encoded_states is not None:
                ctx = torch.cat([ctx, encoded_states], dim=1)
            cached_cross_kvs.append(blk.precompute_cross_kv(ctx))
        # In cross-attn mode encoded_states are only used to build cross context/mask.
        # The flow loop consumes precomputed cross K/V tensors directly.
        cached_encoded_states = None

    stacked_cross_kvs = _stack_cross_kvs(cached_cross_kvs)
    # Once cross-attn K/V tensors are precomputed, the full context tensors are no longer
    # needed by block.forward; dropping them avoids redundant graph replay copies.
    contexts = tuple(None for _ in cached_cross_kvs)
    return ActionExpertCachedContext(
        contexts,
        cross_mask,
        stacked_cross_kvs,
        cached_encoded_states,
        states_mode,
    )


def _ae_forward(ae, actions, timesteps, encoder_hidden_states,
                encoder_attention_mask=None, state_embeddings=None,
                states_mode="cross_attn", cached_context=None):
    bsz, seq_len, _ = actions.shape
    timestep_embed = ae.time_embed(timesteps)
    x = ae.action_embed(actions)

    if cached_context is not None:
        encoded_states = cached_context.encoded_states
        contexts = cached_context.contexts
        cross_mask = cached_context.cross_mask
        cached_cross_kvs = cached_context.cached_cross_kvs
        states_mode = cached_context.states_mode
    else:
        encoded_states = ae._encode_states(state_embeddings)
        cached_cross_kvs = None
        all_visible = (
            isinstance(encoder_attention_mask, torch.Tensor)
            and encoder_attention_mask.dtype == torch.bool
            and bool(torch.all(encoder_attention_mask))
        )
        if states_mode == "self_attn":
            contexts = ae._prepare_context(encoder_hidden_states, None)
            cross_mask = None if all_visible else ae._build_cross_attention_mask(
                encoder_attention_mask, None, bsz, x.dtype
            )
        else:
            contexts = ae._prepare_context(encoder_hidden_states, encoded_states)
            cross_mask = None if all_visible else ae._build_cross_attention_mask(
                encoder_attention_mask, encoded_states, bsz, x.dtype
            )

    if states_mode == "self_attn" and encoded_states is not None:
        x = torch.cat([encoded_states, x], dim=1)
        pos = ae.action_pos_embed[:, :encoded_states.shape[1] + seq_len, :]
    else:
        pos = ae.action_pos_embed[:, :seq_len, :]
    x = x + pos

    for i, block in enumerate(ae.blocks):
        if cached_cross_kvs is not None:
            cross_kv = (cached_cross_kvs[0, i], cached_cross_kvs[1, i])
            context = None
        else:
            cross_kv = None
            context = contexts[i]
        x = block(x, timestep_embed, context, attn_mask=cross_mask, cached_cross_kv=cross_kv)

    output = ae.final_layer(x, timestep_embed)
    if states_mode == "self_attn" and encoded_states is not None:
        output = output[:, encoded_states.shape[1]:, :]
    return output


# ═══════════════════════════════════════════════════════════════════════════
#  2. MolmoBot patches — split backbone, CUDA graph for flow loop
# ═══════════════════════════════════════════════════════════════════════════

def patch_molmobot(model):
    """Add run_backbone_only, generate_actions_from_cache, and CUDA graph support."""
    model._use_cuda_graph = False
    model._cuda_graph = None
    model._cuda_graph_pool = None
    model._graph_captured_shape = None

    model.run_backbone_only = lambda **kw: _run_backbone_only(model, **kw)
    model.generate_actions_from_cache = lambda *a, **kw: _generate_actions_from_cache(model, *a, **kw)
    model.enable_cuda_graph = lambda: _enable_cuda_graph(model)

    _original_generate = model.generate_actions
    model._original_generate_actions = _original_generate

    @torch.no_grad()
    def patched_generate_actions(**kwargs):
        states = kwargs.pop("states", None)
        num_steps = kwargs.pop("num_steps", None)
        generator = kwargs.pop("generator", None)
        if states is None:
            raise ValueError("States must be provided")
        backbone_kw = {k: v for k, v in kwargs.items()
                       if k not in ("labels", "loss_masks")}
        layer_states, enc_mask = _run_backbone_only(model, **backbone_kw)
        return _generate_actions_from_cache(model, layer_states, enc_mask, states,
                                            num_steps=num_steps, generator=generator)

    model.generate_actions = patched_generate_actions
    log.info("Patched MolmoBot: split backbone + CUDA graph support")


@torch.no_grad()
def _run_backbone_only(model, **kwargs):
    input_ids = kwargs.get("input_ids")
    attention_mask = kwargs.get("attention_mask")
    enc_mask = model._get_encoder_attention_mask(input_ids, attention_mask)
    fwd_kw = {k: v for k, v in kwargs.items()
              if k in ("input_ids","input_embeddings","attention_mask","attention_bias",
                       "response_mask","subsegment_ids","position_ids","images",
                       "image_masks","token_pooling","low_res_token_pooling")}
    _, layer_states = model._run_backbone(
        collect_layer_hidden_states=True, output_hidden_states=False, **fwd_kw)
    if layer_states is None:
        raise RuntimeError("Failed to capture hidden states.")
    layer_states = model._select_layer_states(layer_states)
    return layer_states, enc_mask


@torch.no_grad()
def _generate_actions_from_cache(model, layer_states, enc_mask, states,
                                 num_steps=None, generator=None):
    states = model.adapt_state_based_on_mode(states)
    steps = num_steps or model.config.flow_matching_num_steps
    batch_size = layer_states[0].shape[0]
    device = layer_states[0].device
    trajectory = torch.randn(
        (batch_size, model.config.action_horizon, model.config.action_dim),
        device=device, generator=generator)

    cached_ctx = model.action_expert.precompute_context(
        layer_states, encoder_attention_mask=enc_mask,
        state_embeddings=states, states_mode=model.config.states_mode)

    if model._use_cuda_graph:
        trajectory = _run_flow_loop_cudagraph(model, trajectory, layer_states, cached_ctx, steps)
    else:
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((batch_size,), i / steps, device=device)
            velocity = model.action_expert(
                trajectory, t, layer_states, cached_context=cached_ctx)
            trajectory.add_(velocity, alpha=dt)
    return trajectory


def _enable_cuda_graph(model):
    model._use_cuda_graph = True
    model._cuda_graph = None
    model._cuda_graph_pool = None
    model._graph_captured_shape = None


def _run_flow_loop_cudagraph(model, trajectory, layer_states, cached_ctx, steps):
    batch_size = trajectory.shape[0]
    ctx_seq_len = cached_ctx.cached_cross_kvs.shape[4]
    shape_key = (batch_size, steps, ctx_seq_len)

    if model._cuda_graph is None or model._graph_captured_shape != shape_key:
        _capture_flow_graph(model, trajectory, layer_states, cached_ctx, steps)
        model._graph_captured_shape = shape_key

    gv = model._graph_vars
    gv["trajectory"].copy_(trajectory)
    gv["cross_kvs"].copy_(cached_ctx.cached_cross_kvs)
    if gv.get("cross_mask") is not None and cached_ctx.cross_mask is not None:
        gv["cross_mask"].copy_(cached_ctx.cross_mask)
    if gv.get("encoded_states") is not None and cached_ctx.encoded_states is not None:
        gv["encoded_states"].copy_(cached_ctx.encoded_states)

    model._cuda_graph.replay()
    return gv["trajectory"]


@torch.no_grad()
def _capture_flow_graph(model, trajectory, layer_states, cached_ctx, steps):
    device = trajectory.device
    batch_size = trajectory.shape[0]
    if (
        getattr(model, "_enable_compiled_ae_step", False)
        and not getattr(model, "_compiled_ae_disabled", False)
    ):
        try:
            g_traj = trajectory.clone()
            g_kvs = cached_ctx.cached_cross_kvs.clone()
            g_mask = cached_ctx.cross_mask.clone() if cached_ctx.cross_mask is not None else None
            g_enc = cached_ctx.encoded_states.clone() if cached_ctx.encoded_states is not None else None
            g_ts = [torch.full((batch_size,), i / steps, device=device) for i in range(steps)]
            g_cached = ActionExpertCachedContext(
                tuple(cached_ctx.contexts),
                g_mask,
                g_kvs,
                g_enc,
                cached_ctx.states_mode,
            )
            step_fn = _safe_compile_callable(
                lambda traj, t: model.action_expert(
                    traj, t, layer_states, cached_context=g_cached),
                "action-expert-step",
                mode="max-autotune-no-cudagraphs",
                fullgraph=False,
            )
            dt = 1.0 / steps

            for _ in range(2):
                g_traj.copy_(trajectory)
                for si in range(steps):
                    vel = step_fn(g_traj, g_ts[si])
                    g_traj.add_(vel, alpha=dt)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            g_traj.copy_(trajectory)
            with torch.cuda.graph(graph, pool=model._cuda_graph_pool):
                for si in range(steps):
                    vel = step_fn(g_traj, g_ts[si])
                    g_traj.add_(vel, alpha=dt)

            if model._cuda_graph_pool is None:
                model._cuda_graph_pool = graph.pool()
            model._cuda_graph = graph
            model._graph_vars = {
                "trajectory": g_traj,
                "cross_mask": g_mask,
                "cross_kvs": g_kvs,
                "encoded_states": g_enc,
            }
            log.info(
                "Captured compiled CUDA graph: steps=%d, ctx_len=%d",
                steps, g_kvs.shape[4],
            )
            return
        except Exception as exc:
            log.warning("Compiled AE graph capture failed, falling back to eager path: %s", exc)
            model._compiled_ae_disabled = True

    g_traj = trajectory.clone()
    g_ctx = list(cached_ctx.contexts)
    g_mask = cached_ctx.cross_mask.clone() if cached_ctx.cross_mask is not None else None
    g_kvs = cached_ctx.cached_cross_kvs.clone()
    g_enc = cached_ctx.encoded_states.clone() if cached_ctx.encoded_states is not None else None
    g_ts = [torch.full((batch_size,), i / steps, device=device) for i in range(steps)]
    g_cached = ActionExpertCachedContext(
        g_ctx, g_mask, g_kvs, g_enc, cached_ctx.states_mode)
    dt = 1.0 / steps

    for _ in range(2):
        g_traj.copy_(trajectory)
        for i in range(steps):
            vel = model.action_expert(g_traj, g_ts[i], layer_states, cached_context=g_cached)
            g_traj.add_(vel, alpha=dt)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    g_traj.copy_(trajectory)
    with torch.cuda.graph(graph, pool=model._cuda_graph_pool):
        for i in range(steps):
            vel = model.action_expert(g_traj, g_ts[i], layer_states, cached_context=g_cached)
            g_traj.add_(vel, alpha=dt)

    if model._cuda_graph_pool is None:
        model._cuda_graph_pool = graph.pool()
    model._cuda_graph = graph
    model._graph_vars = {
        "trajectory": g_traj, "contexts": g_ctx, "cross_mask": g_mask,
        "cross_kvs": g_kvs, "encoded_states": g_enc,
    }
    log.info(f"Captured eager CUDA graph: steps={steps}, ctx_len={g_kvs.shape[4]}")


# ═══════════════════════════════════════════════════════════════════════════
#  3. FlashAttention-2 patches for LLM, ViT, ActionExpert
# ═══════════════════════════════════════════════════════════════════════════

def patch_flash_attention(model):
    """Wire flash_attn_func into LLM, ViT, and action expert attention."""
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        log.warning("flash_attn not installed, skipping FA2 patches")
        return

    llm = _patch_llm_fa2(model, flash_attn_func)
    vit = _patch_vit_fa2(model, flash_attn_func)
    ae = _patch_ae_fa2(model, flash_attn_func)
    log.info(f"FlashAttention-2: {llm} LLM + {vit} ViT + {ae} AE layers")


def patch_action_expert_flash_attention(model):
    """Wire flash_attn_func only into action-expert attention."""
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        log.warning("flash_attn not installed, skipping AE FA2 patch")
        return
    ae = _patch_ae_fa2(model, flash_attn_func)
    log.info("FlashAttention-2 AE-only: %d AE layers", ae)


def _llm_flash_sdpa(q, k, v, attn_mask=None, drop_mask=None, dropout_p=0.0,
                    is_causal=False, response_dropout_p=0.0):
    """Replacement for OLMoBlock._scaled_dot_product_attention that always uses FA2 causal."""
    from flash_attn import flash_attn_func as _fa
    q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
    return _fa(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
               dropout_p=dropout_p, causal=True).transpose(1,2)


def _patch_llm_fa2(model, fa_func):
    count = 0
    for mod in model.modules():
        if hasattr(mod, "flash_attn_func") and hasattr(mod, "config") and hasattr(mod.config, "attention_type"):
            mod.flash_attn_func = fa_func
            mod.config.float32_attention = False
            mod._scaled_dot_product_attention = _llm_flash_sdpa
            count += 1
    return count


def patch_llm_flash_attention(model):
    """Enable FlashAttention-2 only for LLM attention layers."""
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        log.warning("flash_attn not installed, skipping LLM-only FA2 patch")
        return 0
    count = _patch_llm_fa2(model, flash_attn_func)
    if count:
        log.info("FlashAttention-2 (LLM-only): %d LLM layers", count)
    return count


def _patch_vit_fa2(model, fa_func):
    from olmo.nn.image_vit import ViTMultiHeadDotProductAttention
    count = 0
    for mod in model.modules():
        if isinstance(mod, ViTMultiHeadDotProductAttention):
            mod.config.float32_attention = False
            def make(m, fa):
                def fn(inputs_q, inputs_kv=None, attn_mask=None):
                    ik = inputs_kv if inputs_kv is not None else inputs_q
                    iv = ik
                    xq, xk, xv = m.wq(inputs_q), m.wk(ik), m.wv(iv)
                    xq = xq.reshape(xq.shape[:2] + (m.num_heads, m.head_dim)).to(torch.bfloat16)
                    xk = xk.reshape(xk.shape[:2] + (m.num_key_value_heads, m.head_dim)).to(torch.bfloat16)
                    xv = xv.reshape(xv.shape[:2] + (m.num_key_value_heads, m.head_dim)).to(torch.bfloat16)
                    if m.num_heads != m.num_key_value_heads:
                        xk = xk.repeat_interleave(m.num_key_value_groups, dim=2, output_size=m.num_heads)
                        xv = xv.repeat_interleave(m.num_key_value_groups, dim=2, output_size=m.num_heads)
                    o = fa(xq, xk, xv, dropout_p=0.0, causal=False)
                    o = o.reshape(o.shape[:2] + (m.embed_dim,))
                    return m.residual_dropout(m.wo(o))
                return fn
            mod.forward = make(mod, fa_func)
            count += 1
    return count


def _patch_ae_fa2(model, fa_func):
    from olmo.nn.action_expert import ActionExpertAttention
    sdpa = F.scaled_dot_product_attention
    count = 0
    for mod in model.modules():
        if isinstance(mod, ActionExpertAttention):
            def make(m, fa, _sdpa=sdpa):
                def fn(x, kv=None, attn_mask=None, precomputed_kv=None):
                    bsz, tgt, _ = x.shape
                    q = m.q_proj(x).view(bsz, tgt, m.num_heads, m.head_dim).to(torch.bfloat16)
                    if precomputed_kv is not None:
                        k, v = precomputed_kv
                        k = k.to(torch.bfloat16)
                        v = v.to(torch.bfloat16)
                        if attn_mask is None:
                            # Cached K/V is already prepared per layer; use FA2 directly
                            # in the common no-mask path to reduce flow-loop overhead.
                            o = fa(q, k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0, causal=False)
                        else:
                            o = _sdpa(
                                q.transpose(1, 2),
                                k,
                                v,
                                attn_mask=attn_mask,
                                dropout_p=0.0,
                            )
                            o = o.transpose(1, 2)
                    elif kv is None:
                        s = x.shape[1]
                        kv_out = m.kv_proj(x).view(bsz, s, 2, m.num_heads, m.head_dim).to(torch.bfloat16)
                        o = fa(q, kv_out[:,:,0], kv_out[:,:,1], dropout_p=0.0, causal=False)
                    else:
                        s = kv.shape[1]
                        kv_out = m.kv_proj(kv).view(bsz, s, 2, m.num_heads, m.head_dim).to(torch.bfloat16)
                        if attn_mask is None:
                            o = fa(q, kv_out[:,:,0], kv_out[:,:,1], dropout_p=0.0, causal=False)
                        else:
                            o = _sdpa(q.transpose(1,2), kv_out[:,:,0].transpose(1,2),
                                      kv_out[:,:,1].transpose(1,2), attn_mask=attn_mask, dropout_p=0.0)
                            o = o.transpose(1,2)
                    o = o.contiguous().view(bsz, tgt, m.hidden_size)
                    return m.proj_drop(m.proj(o))
                return fn
            mod.forward = make(mod, fa_func)
            count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════════
#  4. Compiled backbone — torch.compile on LLM + ViT blocks
# ═══════════════════════════════════════════════════════════════════════════

def patch_compile_backbone(model):
    llm = 0
    llm_skipped = 0
    for i, blk in enumerate(model.transformer.blocks):
        compiled = _safe_compile_module(blk, f"llm.block.{i}")
        model.transformer.blocks[i] = compiled
        if compiled is blk:
            llm_skipped += 1
        else:
            llm += 1

    vit = 0
    vit_skipped = 0
    vb = getattr(model, "vision_backbone", None)
    if vb:
        iv = getattr(vb, "image_vit", None)
        if iv:
            tr = getattr(iv, "transformer", None)
            if tr:
                rb = getattr(tr, "resblocks", None)
                if rb:
                    for i, blk in enumerate(rb):
                        compiled = _safe_compile_module(blk, f"vit.block.{i}")
                        rb[i] = compiled
                        if compiled is blk:
                            vit_skipped += 1
                        else:
                            vit += 1
    log.info(
        "Compiled backbone: %d LLM + %d ViT blocks (skipped: %d LLM, %d ViT)",
        llm,
        vit,
        llm_skipped,
        vit_skipped,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  5. FP8 quantization (LOSSY — changes numerical precision)
# ═══════════════════════════════════════════════════════════════════════════

def patch_fp8_quantize(model, *, quantize_backbone=True, quantize_action_expert=True):
    """Replace eligible Linear layers with FP8 static-weight, dynamic-activation matmuls.

    Converts weights to ``float8_e4m3fn`` once (with per-tensor absmax scaling),
    then at runtime casts activations to FP8 on the fly and calls
    ``torch._scaled_mm`` which runs on H100 FP8 Tensor Cores.

    Layers whose dimensions are not divisible by 16 are skipped (HW constraint).

    **This is a lossy transformation** — model outputs will differ from BF16 baseline.
    """
    E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

    def _fp8_eligible(mod):
        if not isinstance(mod, torch.nn.Linear):
            return False
        return mod.in_features % 16 == 0 and mod.out_features % 16 == 0

    eligible_mods = []

    def _prepare_linear(mod):
        """Quantize weights to FP8 and install a calibration hook."""
        w = mod.weight.data.float()
        w_amax = w.abs().max().clamp(min=1e-12)
        w_scale = (E4M3_MAX / w_amax).float()
        w_fp8 = (w * w_scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)

        mod.weight_fp8 = torch.nn.Parameter(w_fp8, requires_grad=False)
        mod.w_scale_inv = (1.0 / w_scale).to(device=w.device, dtype=torch.float32)
        mod._calib_amax = torch.zeros(1, device=w.device, dtype=torch.float32)
        eligible_mods.append(mod)

    def _install_calib_hook(mod):
        """Forward hook that tracks activation amax during calibration."""
        def hook(m, args, output):
            x = args[0]
            amax = x.detach().reshape(-1, x.shape[-1]).abs().max()
            m._calib_amax = torch.max(m._calib_amax, amax)
        mod._calib_handle = mod.register_forward_hook(hook)

    def _finalize_linear(mod):
        """Freeze calibrated activation scale and install FP8 forward."""
        mod._calib_handle.remove()
        del mod._calib_handle

        act_amax = mod._calib_amax.clamp(min=1e-12)
        act_scale = (E4M3_MAX / act_amax).float()
        mod.act_scale = act_scale
        mod.act_scale_inv = (1.0 / act_scale).float()
        del mod._calib_amax

        def fp8_forward(x, _m=mod, _e4=E4M3_MAX):
            x_flat = x.reshape(-1, x.shape[-1])
            x_fp8 = (x_flat.float() * _m.act_scale).clamp(-_e4, _e4).to(torch.float8_e4m3fn)
            out = torch._scaled_mm(
                x_fp8, _m.weight_fp8.T,
                scale_a=_m.act_scale_inv,
                scale_b=_m.w_scale_inv,
                out_dtype=torch.bfloat16,
            )
            if _m.bias is not None:
                out = out + _m.bias
            return out.view(*x.shape[:-1], out.shape[-1])
        mod.forward = fp8_forward

    bb_count = ae_count = skipped = 0

    if quantize_backbone:
        for fqn, mod in model.transformer.named_modules():
            if _fp8_eligible(mod):
                _prepare_linear(mod)
                _install_calib_hook(mod)
                bb_count += 1
            elif isinstance(mod, torch.nn.Linear):
                skipped += 1

    if quantize_action_expert:
        for fqn, mod in model.action_expert.named_modules():
            if _fp8_eligible(mod):
                _prepare_linear(mod)
                _install_calib_hook(mod)
                ae_count += 1
            elif isinstance(mod, torch.nn.Linear):
                skipped += 1

    log.info(f"FP8: calibrating {bb_count} backbone + {ae_count} AE linears "
             f"({skipped} skipped)... running calibration forward passes")

    model._fp8_eligible_mods = eligible_mods
    model._fp8_finalize = lambda: _finalize_all(eligible_mods)


def _finalize_fp8(model):
    """Called after calibration forward passes to freeze FP8 scales."""
    for mod in model._fp8_eligible_mods:
        _mod = mod
        E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

        _mod._calib_handle.remove()
        del _mod._calib_handle

        act_amax = _mod._calib_amax.clamp(min=1e-12)
        act_scale = (E4M3_MAX / act_amax).float()
        _mod.act_scale = act_scale
        _mod.act_scale_inv = (1.0 / act_scale).float()
        del _mod._calib_amax

        def fp8_forward(x, _m=_mod, _e4=E4M3_MAX):
            x_flat = x.reshape(-1, x.shape[-1])
            x_fp8 = (x_flat.float() * _m.act_scale).clamp(-_e4, _e4).to(torch.float8_e4m3fn)
            out = torch._scaled_mm(
                x_fp8, _m.weight_fp8.T,
                scale_a=_m.act_scale_inv,
                scale_b=_m.w_scale_inv,
                out_dtype=torch.bfloat16,
            )
            if _m.bias is not None:
                out = out + _m.bias
            return out.view(*x.shape[:-1], out.shape[-1])
        _mod.forward = fp8_forward

    del model._fp8_eligible_mods
    del model._fp8_finalize
    log.info("FP8: calibration complete, static scales frozen")


# ═══════════════════════════════════════════════════════════════════════════
#  6. TensorRT compilation (LOSSLESS at BF16 — minor numerical differences)
# ═══════════════════════════════════════════════════════════════════════════

def patch_tensorrt_backbone(model, *, sample_inputs=None):
    """Compile backbone (LLM + ViT) blocks with TensorRT via torch_tensorrt.

    Replaces ``torch.compile`` — do NOT use both on the same blocks.
    Uses BF16 precision by default so outputs are numerically very close
    to the ``torch.compile`` baseline (not bit-identical due to different
    kernel implementations and fusion decisions).
    """
    import torch_tensorrt  # noqa: F811

    trt_opts = {
        "enabled_precisions": {torch.bfloat16},
        "min_block_size": 1,
        "truncate_double": True,
        "use_python_runtime": True,
        "cache_built_engines": True,
        "reuse_cached_engines": True,
    }

    llm = 0
    for i, blk in enumerate(model.transformer.blocks):
        model.transformer.blocks[i] = torch.compile(
            blk, backend="torch_tensorrt", dynamic=False, options=trt_opts)
        llm += 1

    vit = 0
    vb = getattr(model, "vision_backbone", None)
    if vb:
        iv = getattr(vb, "image_vit", None)
        if iv:
            tr = getattr(iv, "transformer", None)
            if tr:
                rb = getattr(tr, "resblocks", None)
                if rb:
                    for i, blk in enumerate(rb):
                        rb[i] = torch.compile(
                            blk, backend="torch_tensorrt", dynamic=False, options=trt_opts)
                        vit += 1

    log.info(f"TensorRT backbone: {llm} LLM + {vit} ViT blocks")


# ═══════════════════════════════════════════════════════════════════════════
#  7. Backbone cache helpers
# ═══════════════════════════════════════════════════════════════════════════

def hash_inputs(input_ids, images):
    h = hashlib.blake2b(digest_size=16)
    h.update(input_ids.cpu().numpy().tobytes())
    if images is not None:
        img = images.cpu().numpy()
        h.update(img.shape.__repr__().encode())
        h.update(img.tobytes())
    return h.digest()
