import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(Q, K, V, sm_scale,  #
                L,  #
                Out,  #
                stride_qz, stride_qh, stride_qm, stride_qk,  #
                stride_kz, stride_kh, stride_kn, stride_kk,  #
                stride_vz, stride_vh, stride_vn, stride_vk,  #
                stride_oz, stride_oh, stride_om, stride_on,  #
                Z, H, N_CTX,  #
                Z_H_N_CTX,  #
                BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,  #
                BLOCK_N: tl.constexpr,  #
                ):
    num_warps = 2
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    vk_offset = qvk_offset // stride_qm
    offs_b = tl.arange(0, num_warps)
    stride_kb = BLOCK_N * BLOCK_DMODEL // num_warps
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + offs_b[:, None, None] * 0 + offs_m[None, :, None] * stride_qm + offs_k[None, None, :] * stride_qk
    k_ptrs = K + offs_b[:, None, None] * stride_kb + offs_n[None, None, :] * stride_kn + offs_k[None, :,
                                                                                                None] * stride_kk
    v_ptrs = V + offs_b[:, None, None] * stride_kb + offs_k[None, None, :] * stride_kk + offs_n[None, :,
                                                                                                None] * stride_kn
    # K_block_ptr = tl.make_block_ptr(
    #     base=K,
    #     shape=(BLOCK_DMODEL, Z_H_N_CTX),
    #     strides=(stride_kk, stride_kn),
    #     offsets=(0, vk_offset),
    #     block_shape=(BLOCK_DMODEL, BLOCK_N),
    #     order=(0, 1),
    # )
    # V_block_ptr = tl.make_block_ptr(
    #     base=V,
    #     shape=(Z_H_N_CTX, BLOCK_DMODEL),
    #     strides=(stride_vn, stride_vk),
    #     offsets=(vk_offset, 0),
    #     block_shape=(BLOCK_N, BLOCK_DMODEL),
    #     order=(1, 0),
    # )
    # initialize offsets

    # initialize pointer to m and l
    m_i = tl.zeros([num_warps, BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([num_warps, BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([num_warps, BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout

    # Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(q_ptrs)

    q = (q * qk_scale).to(K.dtype.element_ty)
    lo = 0
    hi = N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        # -- compute qk ---
        qk = tl.zeros([num_warps, BLOCK_M, BLOCK_N // num_warps], dtype=tl.float32)
        qk += tl.dot(q, k, allow_tf32=True)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 2))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, :, None])
        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p.to(V.dtype.element_ty), v, allow_tf32=True)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_kn
        # K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        # V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(vk_offset + start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # O_ptrs = Out + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    tl.store(O_block_ptr, acc.to(K.dtype.element_ty))


def test_op():
    (Z, H, N_CTX, D_HEAD) = (1, 1, 1024, 32)
    dtype = torch.float16
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    sm_scale = 0.5
    # reference implementation
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)

    BLOCK_M = 128
    BLOCK_N = 64
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q)
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    num_warps = 2
    _fwd_kernel[grid](
        q, k, v, sm_scale,  #
        L,  #
        o,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        q.shape[0], q.shape[1], q.shape[2],  #
        q.shape[0] * q.shape[1] * q.shape[2],  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,  #
        num_warps=num_warps,  #
        num_stages=4  #
    )
    assert torch.allclose(ref_out, o, atol=1e-2, rtol=0)
