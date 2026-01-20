#!/usr/bin/env python3

import argparse
import glob
import os
import re
from typing import Optional

import torch
from safetensors.torch import load_file
from rich.console import Console
from rich.table import Table

HIST_DIR = "."

FLOAT8_DTYPES = tuple(
    dt
    for dt in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    )
    if dt is not None
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare safetensors dumps per tp rank.")
    parser.add_argument("--dir-a", required=True, help="dir for ckpt A")
    parser.add_argument("--dir-b", required=True, help="dir for ckpt B")
    parser.add_argument("--layer-id", type=int, default=0, help="layer id in file name")
    parser.add_argument(
        "--section",
        choices=("attn0", "attn1", "pagetable_input", "kvcache", "indexer_topk"),
        default="indexer_topk",
        help=(
            "select dump section: "
            "`attn0`(deepseek layer attn0 dump), "
            "`attn1`(deepseek layer attn+mlp out dump), "
            "`pagetable_input`(nsa backend pagetable input dump), "
            "`kvcache`(nsa backend kvcache dump), "
            "`indexer_topk`(nsa fused topk/indexer dump)"
        ),
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--max-mismatch", type=int, default=1)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print extra tensor statistics for debugging",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=16,
        help="number of head/tail samples to print in --debug mode",
    )
    parser.add_argument(
        "--check-ref",
        action="store_true",
        help="for `indexer_topk` dumps: check `topk_sorted` equals `ref_sorted` inside each dump file",
    )
    return parser.parse_args()


def sort_topk(x):
    return torch.sort(x, dim=-1)[0]


def _is_close_mask(a, b, rtol, atol):
    if a.is_floating_point() or b.is_floating_point():
        return torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    return a == b


def show_mismatch_window(a, b, key, rtol, atol, window, max_mismatch, console):
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.dtype in FLOAT8_DTYPES or b.dtype in FLOAT8_DTYPES:
        a = a.to(dtype=torch.float32)
        b = b.to(dtype=torch.float32)
    close_mask = _is_close_mask(a, b, rtol, atol)
    mismatch = ~close_mask
    idxs = torch.nonzero(mismatch).flatten()
    if idxs.numel() == 0:
        return

    for idx in idxs[:max_mismatch].tolist():
        start = max(0, idx - window)
        end = min(a.numel(), idx + window + 1)

        table = Table(title=f"{key} mismatch at {idx} (window {start}:{end})")
        table.add_column("idx")
        table.add_column("a")
        table.add_column("b")
        table.add_column("abs_diff")

        for i in range(start, end):
            av = a[i]
            bv = b[i]
            diff = (av - bv).abs()
            table.add_row(
                str(i),
                f"{av.item():.6g}",
                f"{bv.item():.6g}",
                f"{diff.item():.6g}",
            )
        console.print(table)


def _get_any(ckpt: dict, keys: tuple[str, ...]):
    for k in keys:
        v = ckpt.get(k)
        if v is not None:
            return v
    return None


def _compare_prepare(a: torch.Tensor) -> torch.Tensor:
    if a.dtype in FLOAT8_DTYPES:
        return a.to(dtype=torch.float32)
    return a.to(dtype=torch.float32) if a.is_floating_point() else a


def _tensor_equal_or_allclose(a: torch.Tensor, b: torch.Tensor, args) -> bool:
    if a.is_floating_point() or b.is_floating_point() or a.dtype in FLOAT8_DTYPES or b.dtype in FLOAT8_DTYPES:
        aa = _compare_prepare(a)
        bb = _compare_prepare(b)
        return bool(torch.allclose(aa, bb, rtol=args.rtol, atol=args.atol, equal_nan=True))
    return bool(torch.equal(a, b))


def _iter_common_keys(ckpt_a: dict, ckpt_b: dict) -> list[str]:
    return sorted(set(ckpt_a.keys()) & set(ckpt_b.keys()))


def _compare_topk_like(a: torch.Tensor, b: torch.Tensor, *, key: str, args, console) -> None:
    aa = _prepare_topk(a)
    bb = _prepare_topk(b)
    ok = bool(torch.equal(aa, bb))
    print(f"{key} equal(sorted): {ok}")
    if not ok:
        show_mismatch_window(aa, bb, key, args.rtol, args.atol, args.window, args.max_mismatch, console)


def _compare_logits_valid_if_possible(ckpt_a: dict, ckpt_b: dict, args, console) -> None:
    logits_a = ckpt_a.get("logits")
    logits_b = ckpt_b.get("logits")
    if logits_a is None or logits_b is None:
        return

    row_starts_a = ckpt_a.get("row_starts")
    row_lens_a = ckpt_a.get("row_lens")
    row_starts_b = ckpt_b.get("row_starts")
    row_lens_b = ckpt_b.get("row_lens")

    if (
        row_starts_a is None
        or row_lens_a is None
        or row_starts_b is None
        or row_lens_b is None
        or not torch.equal(row_starts_a, row_starts_b)
        or not torch.equal(row_lens_a, row_lens_b)
    ):
        print("logits_valid allclose: skip (missing or mismatched row_starts/row_lens)")
        return

    ks = row_starts_a.to(dtype=torch.int64)
    lens = row_lens_a.to(dtype=torch.int64)
    kv_dim = int(logits_a.shape[1])
    cols = torch.arange(kv_dim, dtype=torch.int64).reshape(1, -1)
    mask = (cols >= ks.reshape(-1, 1)) & (cols < (ks + lens).reshape(-1, 1))

    a_valid = _compare_prepare(logits_a)[mask]
    b_valid = _compare_prepare(logits_b)[mask]
    logits_close = bool(torch.allclose(a_valid, b_valid, rtol=args.rtol, atol=args.atol))
    print(f"logits_valid allclose: {logits_close}")
    if logits_close:
        return

    diff_mask = ~torch.isclose(
        _compare_prepare(logits_a), _compare_prepare(logits_b), rtol=args.rtol, atol=args.atol
    )
    mismatch = diff_mask & mask
    idx = torch.nonzero(mismatch)
    if idx.numel() == 0:
        return

    r = int(idx[0, 0].item())
    c = int(idx[0, 1].item())
    row_start = int(ks[r].item())
    row_end = int((ks[r] + lens[r]).item())
    start = max(row_start, c - args.window)
    end = min(row_end, c + args.window + 1)
    table = Table(title=f"logits_valid mismatch at row={r} col={c} (window {start}:{end})")
    table.add_column("col")
    table.add_column("a")
    table.add_column("b")
    table.add_column("abs_diff")
    for j in range(start, end):
        av = float(_compare_prepare(logits_a)[r, j].item())
        bv = float(_compare_prepare(logits_b)[r, j].item())
        diff = abs(av - bv)
        table.add_row(str(j), f"{av:.6g}", f"{bv:.6g}", f"{diff:.6g}")
    console.print(table)


def _print_tie_stats(ckpt: dict, *, topk: int, tag: str) -> None:
    tie_eq = ckpt.get("tie_eq_count")
    tie_gt = ckpt.get("tie_gt_count")
    row_lens = ckpt.get("row_lens")

    if tie_eq is None or tie_gt is None or row_lens is None:
        return

    tie_eq = tie_eq.to(dtype=torch.int64).reshape(-1)
    tie_gt = tie_gt.to(dtype=torch.int64).reshape(-1)
    row_lens = row_lens.to(dtype=torch.int64).reshape(-1)

    cutoff_mask = row_lens > int(topk)
    cutoff_rows = int(cutoff_mask.sum().item())
    total_rows = int(row_lens.numel())
    if cutoff_rows == 0:
        print(f"[TIE] {tag}: cutoff_rows=0/{total_rows} (row_len<=topk for all rows)")
        return

    tie_rows = int(((tie_eq > 1) & cutoff_mask).sum().item())
    tie_rows_pct = 100.0 * tie_rows / cutoff_rows

    need_eq = (int(topk) - tie_gt).clamp(min=0)
    ambiguous = (tie_eq - need_eq).clamp(min=0)
    ambig_rows = int(((ambiguous > 0) & cutoff_mask).sum().item())
    ambig_rows_pct = 100.0 * ambig_rows / cutoff_rows

    tie_eq_max = int((tie_eq[cutoff_mask].max().item())) if cutoff_rows else 0
    ambig_max = int((ambiguous[cutoff_mask].max().item())) if cutoff_rows else 0

    print(
        f"[TIE] {tag}: "
        f"tie_rows(eq>1)={tie_rows}/{cutoff_rows} ({tie_rows_pct:.2f}%) "
        f"ambiguous_rows={ambig_rows}/{cutoff_rows} ({ambig_rows_pct:.2f}%) "
        f"max_eq={tie_eq_max} max_ambiguous={ambig_max}"
    )


def _first_ambiguous_row(ckpt: dict, *, topk: int) -> Optional[int]:
    tie_eq = ckpt.get("tie_eq_count")
    tie_gt = ckpt.get("tie_gt_count")
    row_lens = ckpt.get("row_lens")
    if tie_eq is None or tie_gt is None or row_lens is None:
        return None

    tie_eq = tie_eq.to(dtype=torch.int64).reshape(-1)
    tie_gt = tie_gt.to(dtype=torch.int64).reshape(-1)
    row_lens = row_lens.to(dtype=torch.int64).reshape(-1)
    cutoff_mask = row_lens > int(topk)
    need_eq = (int(topk) - tie_gt).clamp(min=0)
    ambiguous = (tie_eq - need_eq) > 0
    idx = torch.nonzero(cutoff_mask & ambiguous).flatten()
    if idx.numel() == 0:
        return None
    return int(idx[0].item())


def _print_boundary_tie_evidence(ckpt: dict, *, topk: int, tag: str) -> None:
    row = _first_ambiguous_row(ckpt, topk=topk)
    if row is None:
        return

    logits = ckpt.get("logits")
    row_starts = ckpt.get("row_starts")
    row_lens = ckpt.get("row_lens")
    tie_eq = ckpt.get("tie_eq_count")
    tie_gt = ckpt.get("tie_gt_count")
    if logits is None or row_starts is None or row_lens is None or tie_eq is None or tie_gt is None:
        return

    base = int(row_starts[row].item())
    length = int(row_lens[row].item())
    if length <= int(topk):
        return

    scores = logits[row, base : base + length].to(dtype=torch.float32)
    sorted_scores, sorted_idx = torch.sort(scores, dim=-1, descending=True, stable=True)

    s_k = float(sorted_scores[int(topk) - 1].item())
    s_k1 = float(sorted_scores[int(topk)].item())
    pos_k = int(sorted_idx[int(topk) - 1].item())
    pos_k1 = int(sorted_idx[int(topk)].item())
    col_k = base + pos_k
    col_k1 = base + pos_k1

    eq = int(tie_eq[row].item())
    gt = int(tie_gt[row].item())
    need_eq = max(0, int(topk) - gt)

    print(
        f"[TIE] {tag}: first_ambiguous_row={row} base={base} len={length} "
        f"gt={gt} eq={eq} need_eq={need_eq} (eq-need_eq={eq - need_eq})"
    )
    print(
        f"[TIE] {tag}: rank_{topk} col={col_k} score={s_k:.6g} | "
        f"rank_{topk+1} col={col_k1} score={s_k1:.6g} (diff={abs(s_k - s_k1):.6g})"
    )


def compare_rank(ckpt_a, ckpt_b, args, console, file_tag):
    keys_a = set(ckpt_a.keys())
    keys_b = set(ckpt_b.keys())
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    if args.debug and (only_a or only_b):
        print(f"[DEBUG] keys only in A: {only_a}")
        print(f"[DEBUG] keys only in B: {only_b}")

    common_keys = _iter_common_keys(ckpt_a, ckpt_b)

    if args.section == "indexer_topk":
        _compare_logits_valid_if_possible(ckpt_a, ckpt_b, args, console)

    for key in common_keys:
        if args.section == "indexer_topk" and key == "logits":
            continue
        if key in ("topk_indices", "topk_result", "topk_sorted", "ref_sorted"):
            _compare_topk_like(ckpt_a[key], ckpt_b[key], key=key, args=args, console=console)
            continue

        a = ckpt_a[key]
        b = ckpt_b[key]
        ok = _tensor_equal_or_allclose(a, b, args)
        print(f"{key} allclose: {ok}")
        if not ok:
            show_mismatch_window(a, b, key, args.rtol, args.atol, args.window, args.max_mismatch, console)

    topk_a = _get_any(ckpt_a, ("topk_indices", "topk_result"))
    topk_b = _get_any(ckpt_b, ("topk_indices", "topk_result"))
    logits_a = ckpt_a.get("logits")
    logits_b = ckpt_b.get("logits")
    return topk_a, topk_b, logits_a, logits_b


def _tp_rank_from_name(name):
    match = re.search(r"_tp(\d+)\.safetensors$", name)
    if match:
        return int(match.group(1))
    return -1


def _prepare_topk(tensor):
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    return sort_topk(tensor)


def _safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _collect_tp_files(dir_path, pattern):
    files = glob.glob(os.path.join(dir_path, pattern))
    tp_to_path = {}
    for path in files:
        name = os.path.basename(path)
        tp_rank = _tp_rank_from_name(name)
        if tp_rank < 0:
            continue
        tp_to_path[tp_rank] = path
    return tp_to_path


def _desc(x):
    if x is None:
        return "None"
    return f"shape={tuple(x.shape)} dtype={x.dtype}"

def _debug_tensor_stats(name: str, t: Optional[torch.Tensor], *, samples: int, kv_dim: Optional[int] = None):
    if t is None:
        print(f"[DEBUG] {name}: None")
        return

    shape = tuple(t.shape)
    dtype = t.dtype
    print(f"[DEBUG] {name}: shape={shape} dtype={dtype}")

    if t.numel() == 0:
        print(f"[DEBUG] {name}: empty")
        return

    if dtype in FLOAT8_DTYPES:
        t = t.to(dtype=torch.float32)
    elif t.is_floating_point():
        t = t.to(dtype=torch.float32)

    flat = t.reshape(-1)
    if t.is_floating_point():
        print(f"[DEBUG] {name}: min={flat.min().item():.6g} max={flat.max().item():.6g}")
    else:
        print(f"[DEBUG] {name}: min={int(flat.min().item())} max={int(flat.max().item())}")

    if t.dtype == torch.int32 or t.dtype == torch.int64:
        num_neg1 = int((flat == -1).sum().item())
        if num_neg1:
            print(f"[DEBUG] {name}: count(-1)={num_neg1}/{flat.numel()}")

    if kv_dim is not None and t.dtype in (torch.int32, torch.int64):
        in_range = (flat >= 0) & (flat < kv_dim)
        frac = float(in_range.to(dtype=torch.float32).mean().item())
        print(f"[DEBUG] {name}: in_range([0,{kv_dim}))={frac:.4f}")

    if t.ndim == 1 and t.dtype in (torch.int32, torch.int64):
        uniq = int(torch.unique(t).numel())
        mono_inc = bool(torch.all(t[1:] >= t[:-1]).item()) if t.numel() > 1 else True
        print(f"[DEBUG] {name}: unique={uniq} monotonic_non_decreasing={mono_inc}")

        ar = torch.arange(t.numel(), dtype=t.dtype, device=t.device)
        is_arange0 = torch.equal(t, ar)
        is_arange1 = torch.equal(t, ar + 1)
        print(f"[DEBUG] {name}: is_arange0={is_arange0} is_arange1={is_arange1}")

    head_n = min(samples, flat.numel())
    tail_n = min(samples, flat.numel())
    head = flat[:head_n].tolist()
    tail = flat[-tail_n:].tolist()
    print(f"[DEBUG] {name}: head={head}")
    print(f"[DEBUG] {name}: tail={tail}")


def main():
    args = parse_args()
    console = Console()
    os.makedirs(HIST_DIR, exist_ok=True)

    patterns = {
        "attn0": f"debug_attn0_tensor_file_{args.layer_id}_tp*.safetensors",
        "attn1": f"debug_attn_mlp_out_tensor_file_{args.layer_id}_tp*.safetensors",
        "pagetable_input": f"debug_attn1_pagetable_input_file_{args.layer_id}_tp*.safetensors",
        "kvcache": f"debug_attn1_kvcache_file_{args.layer_id}_tp*.safetensors",
        "indexer_topk": f"debug_indexer_topk_tensor_file_{args.layer_id}_tp*.safetensors",
    }
    pattern = patterns[args.section]

    print(f"Section: {args.section}")

    print(f"Searching for files with pattern: {pattern}")
    print(f"Dir A: {args.dir_a}")
    print(f"Dir B: {args.dir_b}")

    files_a = _collect_tp_files(args.dir_a, pattern)
    files_b = _collect_tp_files(args.dir_b, pattern)

    print(f"Found {len(files_a)} files in dir A: {list(files_a.keys())}")
    print(f"Found {len(files_b)} files in dir B: {list(files_b.keys())}")

    tp_ranks = sorted(set(files_a) & set(files_b))

    if not tp_ranks:
        console.print("[red]ERROR: No matching tp rank files found in both directories![/red]")
        console.print(f"[yellow]Pattern: {pattern}[/yellow]")
        console.print(f"[yellow]Dir A files: {list(files_a.values())}[/yellow]")
        console.print(f"[yellow]Dir B files: {list(files_b.values())}[/yellow]")
        return

    print(f"Comparing {len(tp_ranks)} tp ranks: {tp_ranks}")
    print()

    for tp_rank in tp_ranks:
        file_a = files_a[tp_rank]
        file_b = files_b[tp_rank]
        name = os.path.basename(file_a)
        console.rule(f"TP Rank {tp_rank}: {name}")
        ckpt_a = load_file(file_a)
        ckpt_b = load_file(file_b)
        topk_a, topk_b, logits_a, logits_b = compare_rank(ckpt_a, ckpt_b, args, console, name)

        print(
            f"TP{tp_rank} shapes: "
            f"A.topk={_desc(topk_a)} A.logits={_desc(logits_a)} | "
            f"B.topk={_desc(topk_b)} B.logits={_desc(logits_b)}"
        )

        if args.debug:
            print(f"[DEBUG] TP{tp_rank} keys A: {sorted(list(ckpt_a.keys()))}")
            print(f"[DEBUG] TP{tp_rank} keys B: {sorted(list(ckpt_b.keys()))}")

            kv_dim_a = int(logits_a.shape[1]) if logits_a is not None and logits_a.ndim >= 2 else None
            kv_dim_b = int(logits_b.shape[1]) if logits_b is not None and logits_b.ndim >= 2 else None
            q_dim_a = int(logits_a.shape[0]) if logits_a is not None and logits_a.ndim >= 1 else None
            q_dim_b = int(logits_b.shape[0]) if logits_b is not None and logits_b.ndim >= 1 else None
            print(f"[DEBUG] TP{tp_rank} dims: A.q={q_dim_a} A.kv={kv_dim_a} | B.q={q_dim_b} B.kv={kv_dim_b}")

            _debug_tensor_stats(f"TP{tp_rank}.A.topk", topk_a, samples=args.debug_samples, kv_dim=kv_dim_a)

            _debug_tensor_stats(f"TP{tp_rank}.B.topk", topk_b, samples=args.debug_samples, kv_dim=kv_dim_b)

        if args.section == "indexer_topk":
            topk_k = None
            topk_tensor = _get_any(ckpt_a, ("topk_result", "topk_indices", "topk_sorted"))
            if topk_tensor is not None and topk_tensor.ndim >= 2:
                topk_k = int(topk_tensor.shape[1])
            if topk_k is not None:
                _print_tie_stats(ckpt_a, topk=topk_k, tag=f"TP{tp_rank} A")
                _print_tie_stats(ckpt_b, topk=topk_k, tag=f"TP{tp_rank} B")
                _print_boundary_tie_evidence(ckpt_a, topk=topk_k, tag=f"TP{tp_rank} A")
                _print_boundary_tie_evidence(ckpt_b, topk=topk_k, tag=f"TP{tp_rank} B")

            a_topk_sorted = ckpt_a.get("topk_sorted")
            a_ref_sorted = ckpt_a.get("ref_sorted")
            b_topk_sorted = ckpt_b.get("topk_sorted")
            b_ref_sorted = ckpt_b.get("ref_sorted")

            if a_topk_sorted is not None and b_topk_sorted is not None:
                ab_equal = torch.equal(a_topk_sorted, b_topk_sorted)
                print(f"TP{tp_rank}: A vs B topk_sorted equal = {ab_equal}")
                if not ab_equal:
                    show_mismatch_window(
                        a_topk_sorted,
                        b_topk_sorted,
                        f"TP{tp_rank}_A_vs_B_topk_sorted",
                        args.rtol,
                        args.atol,
                        args.window,
                        args.max_mismatch,
                        console,
                    )

            if args.check_ref:
                if a_topk_sorted is not None and a_ref_sorted is not None:
                    print(f"TP{tp_rank}: A topk_sorted vs ref_sorted equal = {torch.equal(a_topk_sorted, a_ref_sorted)}")
                if b_topk_sorted is not None and b_ref_sorted is not None:
                    print(f"TP{tp_rank}: B topk_sorted vs ref_sorted equal = {torch.equal(b_topk_sorted, b_ref_sorted)}")

        # Compare A's topk vs B's topk (sorted) for this tp rank, when present.
        if topk_a is not None and topk_b is not None:
            sorted_topk_a = _prepare_topk(topk_a)
            sorted_topk_b = _prepare_topk(topk_b)
            is_equal = torch.equal(sorted_topk_a, sorted_topk_b)
            print(f"TP{tp_rank}: A vs B topk_indices equal(sorted): {is_equal}")
            if not is_equal:
                show_mismatch_window(
                    sorted_topk_a,
                    sorted_topk_b,
                    f"TP{tp_rank}_A_vs_B_topk_indices",
                    args.rtol,
                    args.atol,
                    args.window,
                    args.max_mismatch,
                    console,
                )
if __name__ == "__main__":
    main()
