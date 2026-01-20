import math

import torch

H_KV = 1
QK_ROPE_HEAD_DIM = 64


def _build_sparse_indices(
    bs: int,
    s_q: int,
    h_kv: int,
    topk: int,
    kv_len: int,
    device: str,
    rng: torch.Generator,
) -> torch.Tensor:
    total_q = bs * s_q
    total_kv = bs * kv_len
    indices = torch.full(
        (total_q, h_kv, topk),
        total_kv,
        device=device,
        dtype=torch.int32,
    )
    for b in range(bs):
        kv_base = b * kv_len
        q_base = b * s_q
        for q in range(s_q):
            q_idx = q_base + q
            for h in range(h_kv):
                valid_k = min(topk, kv_len)
                near_mask = (
                    torch.randint(0, 32, (valid_k,), device=device, generator=rng) < 31
                )
                cur = torch.randperm(kv_len, generator=rng, device=device)[:valid_k]
                if near_mask.any():
                    cur[near_mask] = torch.randint(
                        max(0, kv_len - 20000),
                        kv_len,
                        (int(near_mask.sum().item()),),
                        device=device,
                        generator=rng,
                    )
                if valid_k < topk:
                    pad = torch.full(
                        (topk - valid_k,),
                        total_kv,
                        device=device,
                        dtype=torch.int32,
                    )
                    cur = torch.cat([cur, pad])
                indices[q_idx, h] = cur + kv_base
    return indices


def _run_once(run_id: int, device_id: int) -> None:
    device = f"cuda:{device_id}"
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    seed = 42
    batch_size = 1
    seq_len = 4096
    head_num = 128
    head_dim = 576
    topk = 2048
    v_head_dim = head_dim - QK_ROPE_HEAD_DIM
    sm_scale = 1.0 / math.sqrt(head_dim)

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    q = torch.randn(
        (batch_size * seq_len, head_num, head_dim),
        device=device,
        dtype=dtype,
        generator=rng,
    )
    kv = torch.randn(
        (batch_size * seq_len, H_KV, head_dim),
        device=device,
        dtype=dtype,
        generator=rng,
    )
    indices = _build_sparse_indices(
        batch_size, seq_len, H_KV, topk, seq_len, device, rng
    )
    perm_rng = torch.Generator(device=device)
    perm_rng.manual_seed(seed + 1000 + run_id)
    order = torch.randperm(topk, generator=perm_rng, device=device)
    indices_perm = indices.index_select(-1, order)

    from flash_mla import flash_mla_sparse_fwd

    out1, _, _ = flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=v_head_dim,
    )
    out2, _, _ = flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices_perm,
        sm_scale=sm_scale,
        d_v=v_head_dim,
    )

    rng_w = torch.Generator(device=device)
    rng_w.manual_seed(seed)
    w = torch.randn(
        (head_num, v_head_dim, v_head_dim),
        device=device,
        dtype=dtype,
        generator=rng_w,
    )
    out1_bmm = torch.bmm(out1.transpose(0, 1), w).transpose(0, 1)
    out2_bmm = torch.bmm(out2.transpose(0, 1), w).transpose(0, 1)

    same = torch.equal(out1, out2)
    max_diff = (out1 - out2).abs().max().item()
    same_bmm = torch.equal(out1_bmm, out2_bmm)
    max_diff_bmm = (out1_bmm - out2_bmm).abs().max().item()
    print(
        f"run={run_id} device={device} same={same} max_diff={max_diff} "
        f"same_bmm={same_bmm} max_diff_bmm={max_diff_bmm}"
    )


def main() -> None:
    runs = 8
    device_count = torch.cuda.device_count()
    for run_id in range(runs):
        device_id = run_id % device_count
        _run_once(run_id, device_id)


if __name__ == "__main__":
    main()
