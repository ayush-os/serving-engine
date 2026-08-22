"""Phase 4: paged, block-sparse extension of the hand-written FlashAttention-2
Triton kernel. Backward/autograd dropped -- inference only, runs under
@torch.inference_mode().
"""

import torch
import triton
import triton.language as tl

from serving_engine.block import BLOCK_SIZE

PAGED_K_TILE_SIZE = BLOCK_SIZE  # design-agreed: one physical block per K-tile
PAGED_Q_TILE_SIZE = 16


@triton.jit
def paged_flash_fwd_kernel(
    Q_ptr, K_cache_ptr, V_cache_ptr,
    O_ptr, L_ptr,
    block_table_ptr,           # int32[num_rows, max_blocks_per_row]
    q_lens_ptr, k_lens_ptr,    # int32[num_rows]
    q_start_ptr,               # int32[num_rows]
    q_pos_offset_ptr,          # int32[num_rows]: this row's first query's logical position within its own request

    stride_qq, stride_qh, stride_qd,
    stride_kcache_blk, stride_kcache_tok, stride_kcache_h, stride_kcache_d,
    stride_vcache_blk, stride_vcache_tok, stride_vcache_h, stride_vcache_d,
    stride_oq, stride_oh, stride_od,
    stride_lq, stride_lh,
    block_table_row_stride,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    row_index = tl.program_id(1)
    q_head = tl.program_id(2)

    q_start = tl.load(q_start_ptr + row_index)
    n_queries = tl.load(q_lens_ptr + row_index)
    n_keys = tl.load(k_lens_ptr + row_index)
    q_pos_offset = tl.load(q_pos_offset_ptr + row_index)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + q_start * stride_qq + q_head * stride_qh,
        shape=(n_queries, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + q_start * stride_oq + q_head * stride_oh,
        shape=(n_queries, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + q_start * stride_lq + q_head * stride_lh,
        shape=(n_queries,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), value=float("-inf"), dtype=tl.float32)

    Qi = tl.load(Q_block_ptr, boundary_check=(0,))

    for j in range(tl.cdiv(n_keys, K_TILE_SIZE)):
        block_id = tl.load(block_table_ptr + row_index * block_table_row_stride + j)
        kv_head = q_head // (NUM_Q_HEADS // NUM_KV_HEADS)

        K_block_ptr = tl.make_block_ptr(
            K_cache_ptr + block_id * stride_kcache_blk + kv_head * stride_kcache_h,
            shape=(min(K_TILE_SIZE, n_keys - j * K_TILE_SIZE), D),
            strides=(stride_kcache_tok, stride_kcache_d),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_cache_ptr + block_id * stride_vcache_blk + kv_head * stride_vcache_h,
            shape=(min(K_TILE_SIZE, n_keys - j * K_TILE_SIZE), D),
            strides=(stride_vcache_tok, stride_vcache_d),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        Kj = tl.load(K_block_ptr, boundary_check=(0,))
        Vj = tl.load(V_block_ptr, boundary_check=(0,))

        Si_j = tl.dot(Qi, tl.trans(Kj)) * scale

        if is_causal:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE) + q_pos_offset
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            Si_j = tl.where(causal_mask, Si_j, Si_j - 1e6)

        mi_prev = mi
        mi = tl.maximum(mi_prev, tl.max(Si_j, axis=1))
        Pi_j = tl.exp(Si_j - mi[:, None])
        correction = tl.exp(mi_prev - mi)
        li = (correction * li) + tl.sum(Pi_j, axis=1)
        Oi = tl.dot(Pi_j.to(Vj.dtype), Vj) + (correction[:, None] * Oi)

    Oi = Oi / li[:, None]
    Li = mi + tl.log(li)

    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0,))
    tl.store(L_block_ptr, Li, boundary_check=(0,))


def paged_flash_attention(q, k_cache, v_cache, metadata, num_q_heads, num_kv_heads, scale, is_causal=True):
    """Launch wrapper: pulls grid dims and tensor strides for the kernel above.
    q: [total_query, num_q_heads, head_dim]; k_cache/v_cache: [num_blocks,
    block_size, num_kv_heads, head_dim] (one layer's worth, block-structured --
    not pre-gathered). metadata is build_paged_batch_metadata()'s return dict.
    """
    total_query, _, head_dim = q.shape
    num_rows = metadata["q_lens"].shape[0]
    max_n_queries = metadata["q_lens"].max().item() if num_rows else 0

    o = torch.empty_like(q)
    l = torch.empty((total_query, num_q_heads), dtype=torch.float32, device=q.device)

    grid = (triton.cdiv(max_n_queries, PAGED_Q_TILE_SIZE), num_rows, num_q_heads)
    paged_flash_fwd_kernel[grid](
        q, k_cache, v_cache,
        o, l,
        metadata["block_table"],
        metadata["q_lens"], metadata["k_lens"],
        metadata["q_starts"],
        metadata["q_pos_offsets"],
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        o.stride(0), o.stride(1), o.stride(2),
        l.stride(0), l.stride(1),
        metadata["block_table_row_stride"],
        scale,
        D=head_dim,
        Q_TILE_SIZE=PAGED_Q_TILE_SIZE,
        K_TILE_SIZE=PAGED_K_TILE_SIZE,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        is_causal=is_causal,
    )
    return o, l


def build_paged_batch_metadata(scheduler_output, device="cuda"):
    """Per-step, per-request metadata for the kernel above -- reused across
    all layers in a step. Called from ModelRunner.forward() (see
    model_runner.py's triton_paged_attention_forward) once per step, not
    once per layer.
    """
    from serving_engine.request import RequestPhase

    q_lens, k_lens, q_starts, q_pos_offsets, block_tables = [], [], [], [], []
    q_start = 0
    for req in scheduler_output.scheduled_requests:
        if req.phase == RequestPhase.NEEDS_PREFILL:
            n_queries = scheduler_output.prefill_chunk_sizes[req.request_id]
            n_keys = req.num_computed_tokens + n_queries
            q_pos_offset = req.num_computed_tokens
        else:
            n_queries = 1
            n_keys = req.total_len
            q_pos_offset = req.total_len - 1
        q_lens.append(n_queries)
        k_lens.append(n_keys)
        q_starts.append(q_start)
        q_pos_offsets.append(q_pos_offset)
        block_tables.append(list(req.block_table))
        q_start += n_queries

    max_blocks = max((len(bt) for bt in block_tables), default=0)
    padded_block_tables = [bt + [0] * (max_blocks - len(bt)) for bt in block_tables]

    return {
        "q_lens": torch.tensor(q_lens, dtype=torch.int32, device=device),
        "k_lens": torch.tensor(k_lens, dtype=torch.int32, device=device),
        "q_starts": torch.tensor(q_starts, dtype=torch.int32, device=device),
        "q_pos_offsets": torch.tensor(q_pos_offsets, dtype=torch.int32, device=device),
        "block_table": torch.tensor(padded_block_tables, dtype=torch.int32, device=device),
        "block_table_row_stride": max_blocks,
        "total_query": q_start,
    }
