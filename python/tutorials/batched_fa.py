import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, Out, stride_qm, stride_qk, stride_kn, stride_kk, stride_vn, stride_vk, stride_om,
                stride_on, ACC_ptr, M_ptr, L_ptr, N_CTX: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_DMODEL: tl.constexpr):
    start_m = tl.program_id(0)
    num_warps: tl.constexpr = tl.extra.cuda.num_warps()

    offs_b = tl.arange(0, num_warps)
    stride_kb = (BLOCK_N // num_warps) * stride_kn
    offs_n = tl.arange(0, BLOCK_N // num_warps)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    q_ptrs = Q + tl.zeros(
        [num_warps], tl.int32)[:, None, None] + offs_m[None, :, None] * stride_qm + offs_k[None, None, :] * stride_qk
    k_ptrs = K + offs_b[:, None, None] * stride_kb + offs_n[None, None, :] * stride_kn + offs_k[None, :,
                                                                                                None] * stride_kk
    v_ptrs = V + offs_b[:, None, None] * stride_kb + offs_n[None, :, None] * stride_kn + offs_k[None,
                                                                                                None, :] * stride_kk
    # initialize pointer to m and l
    m_i = tl.zeros([num_warps, BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([num_warps, BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([num_warps, BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

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
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=2))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, :, None])
        # -- scale and update acc --
        acc *= alpha[:, :, None]
        acc += tl.dot(p.to(V.dtype.element_ty), v, allow_tf32=True)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, axis=2)
        m_i = m_i_new
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_kn
        # K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        # V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    # o2 = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # m_i_final = tl.zeros([N_CTX, 1], dtype=tl.float32) - float("inf")
    # l_i_final = tl.zeros([N_CTX, 1], dtype=tl.float32)
    # for i in range(num_warps):
    #     msi = m_i[i]
    #     lsi = l_i[i]
    #     osi = acc[i]
    #     alphai = torch.exp(m_i_final - msi)
    #     l_i_final = l_i_final * alphai + lsi
    #     o2 = o2 * alphai + osi
    #     m_i_final = msi
    # o2 = o2 / l_i_final
    # o_ptrs = Out + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    # tl.store(o_ptrs, o2.to(K.dtype.element_ty))
    acc_ptrs = ACC_ptr + offs_b[:, None, None] * BLOCK_M * BLOCK_DMODEL + offs_m[None, :, None] * BLOCK_DMODEL + offs_k[
        None, None, :]
    l_ptrs = L_ptr + offs_b[:, None] * BLOCK_M + offs_m[None, :]
    m_ptrs = M_ptr + offs_b[:, None] * BLOCK_M + offs_m[None, :]
    tl.store(acc_ptrs, acc)
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)


def get_torch_result(q, k, v, BLOCK_N, n_warps):
    # Note: BLOCK_M = N_CTX
    N_CTX = q.shape[0]
    D_HEAD = q.shape[1]
    # BLOCK_N = 32
    # n_warps = 2
    q = q * 1.44269504
    q3d = q.unsqueeze(0).expand(n_warps, -1, -1)
    m_i = torch.full((n_warps, N_CTX, 1), fill_value=float("-inf"), device="cuda")
    l_i = torch.zeros((n_warps, N_CTX, 1), device="cuda")
    acc = torch.zeros((n_warps, N_CTX, D_HEAD), device="cuda")
    for i in range(0, N_CTX, BLOCK_N):
        ki = torch.stack(k[i:i + BLOCK_N].chunk(n_warps, dim=0))
        vi = torch.stack(v[i:i + BLOCK_N].chunk(n_warps, dim=0))
        qk = torch.matmul(q3d, ki.transpose(-1, -2))
        m_i_new = torch.maximum(m_i, qk.max(dim=-1, keepdim=True)[0])
        alpha = torch.exp2(m_i - m_i_new)
        p = torch.exp2(qk - m_i_new)
        acc *= alpha
        acc += torch.matmul(p, vi.to(p.dtype))
        l_i = l_i * alpha + torch.sum(p, dim=-1, keepdim=True)
        m_i = m_i_new

    o2 = torch.zeros_like(q, device="cuda")
    m_i_final = torch.full((N_CTX, 1), device="cuda", fill_value=float("-inf"))
    l_i_final = torch.zeros((N_CTX, 1), device="cuda")
    for i in range(n_warps):
        msi = m_i[i]
        lsi = l_i[i]
        osi = acc[i]
        alphai = torch.exp2(m_i_final - msi)
        l_i_final = l_i_final * alphai + lsi
        o2 = o2 * alphai + osi
        m_i_final = msi
    o2 = o2 / l_i_final
    return o2, acc, l_i, m_i


def test_op():
    (N_CTX, D_HEAD) = (128, 32)
    dtype = torch.float16
    torch.manual_seed(20)
    q = torch.empty((N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    k = torch.empty((N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    v = torch.empty((N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    sm_scale = 1
    # reference implementation
    # p = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    # p = torch.softmax(p.float(), dim=-1).half()
    # ref_out = torch.matmul(p, v)

    BLOCK_M = N_CTX
    BLOCK_N = 32
    BLOCK_K = 32
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert BLOCK_K == Lk
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q)
    grid = (triton.cdiv(q.shape[-2], BLOCK_M), 1, 1)
    num_warps = 2
    acc = torch.zeros((num_warps, BLOCK_M, BLOCK_K), device="cuda")
    l_i = torch.zeros((num_warps, BLOCK_M), device="cuda")
    m_i = torch.full((num_warps, BLOCK_M), fill_value=float("-inf"), device="cuda")
    _fwd_kernel[grid](
        q, k, v, sm_scale, o, q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), o.stride(0),
        o.stride(1), acc, m_i, l_i, q.shape[0],  # q.shape[1],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_K, num_warps=num_warps, num_stages=1)
    torch_out2, acc_torch, l_i_torch, m_i_torch = get_torch_result(q, k, v, BLOCK_N, num_warps)
    torch_out1 = torch.matmul(torch.softmax(torch.matmul(q, k.T), dim=-1), v)
    o2 = torch.zeros_like(q, device="cuda")
    m_i_final = torch.full((N_CTX, 1), device="cuda", fill_value=float("-inf"))
    l_i_final = torch.zeros((N_CTX, 1), device="cuda")
    for i in range(num_warps):
        msi = m_i[i]
        lsi = l_i[i]
        osi = acc[i]
        alphai = torch.exp2(m_i_final - msi[:, None])
        l_i_final = l_i_final * alphai + lsi[:, None]
        o2 = o2 * alphai + osi
        m_i_final = msi[:, None]
    o2 = o2 / l_i_final

    print(f"triton_output={o2}")
    print(f"torch_output={torch_out2}")

    assert torch.allclose(torch_out2.to(o2.dtype), o2, atol=1e-2, rtol=0)
    assert torch.allclose(torch_out1.to(o2.dtype), o2, atol=1e-2, rtol=0)
