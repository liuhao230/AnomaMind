#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export anomaly-window samples from raw CSV(s), then cluster with 18-D features + length-stratified KMeans.

Usage::

  python3 -m AnomaMind.dataprep.gen_cluster /path/to/raw_csv_dir -o /path/to/out
  python3 -m AnomaMind.dataprep.gen_cluster -c /path/to/one.csv -o /path/to/out

Defaults live in ``DEFAULT_CONFIG``; ``build_dataset`` calls ``main()`` with ``input_dir``, ``-o``, and ``--ref-segment-size``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from AnomaMind.core.utils import get_column_names  # noqa: E402


@dataclass(frozen=True)
class ClusterPipelineConfig:
    context_ratio: float = 2.0
    single_min_pad: int = 10
    single_max_pad: int = 500
    dir_min_pad: int = 20
    dir_max_pad: int = 250
    single_min_export_points: int = 40
    dir_min_export_points: int = 30
    series_norm: bool = True
    write_sample_images: bool = False
    k_min: int = 3
    k_max: int = 8
    seed: int = 42
    typical_per_cluster: int = 9
    ref_segment_size: int = 100
    min_plot_points: int = 30
    min_cluster_size_for_typical: int = 2
    inner_recluster_reference: bool = True
    reference_inner_k_max: int = 3
    no_cluster_typical: bool = False
    no_progress: bool = False
    length_stratum_e0: int = 10
    length_stratum_e1: int = 30
    length_stratum_k_max: int = 3
    kmeans_ksearch_n_init: int = 30
    kmeans_refine_n_init: int = 45


DEFAULT_CONFIG = ClusterPipelineConfig()


def _tqdm_iterable(seq, **kwargs):
    """Iterate with a tqdm bar when installed; otherwise plain iteration."""
    try:
        from tqdm import tqdm

        return tqdm(seq, **kwargs)
    except Exception:
        return seq


def _tqdm_single_phase(
    desc: str,
    postfix: str,
    show_progress: bool,
    fn: Callable[[], Any],
) -> Any:
    """Wrap a **single** long call (e.g. ``fit`` / ``fit_predict``) with a one-step tqdm + postfix for n/k/n_init."""
    if not show_progress:
        return fn()
    try:
        from tqdm import tqdm

        with tqdm(
            total=1,
            desc=desc,
            unit="run",
            file=sys.stderr,
            leave=True,
            mininterval=0.15,
            dynamic_ncols=True,
        ) as p:
            p.set_postfix_str(postfix, refresh=True)
            out = fn()
            p.update(1)
        return out
    except Exception:
        return fn()


def _label_runs_one_based(labels: np.ndarray) -> List[Tuple[int, int]]:
    """Return closed interval [start, end] on positional indices 0..N-1."""
    n = len(labels)
    if n == 0:
        return []
    y = (labels.astype(int) != 0).astype(int)
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if y[i] != 1:
            i += 1
            continue
        j = i
        while j < n and y[j] == 1:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def _normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip column-name whitespace/BOM for matching ``Label`` / ``Index``."""
    out = df.copy()
    fixed: List[str] = []
    for c in out.columns:
        s = str(c).strip()
        if s.startswith("\ufeff"):
            s = s[1:].lstrip()
        fixed.append(s)
    out.columns = fixed
    return out


def _ensure_index_value_label(df: pd.DataFrame, row_offset: int = 0) -> pd.DataFrame:
    out = _normalize_dataframe_columns(df)
    if "Label" not in out.columns:
        for c in list(out.columns):
            if str(c).lower() == "label":
                out = out.rename(columns={c: "Label"})
                break
    if "Label" not in out.columns:
        raise ValueError("CSV must contain a Label column (or a column named label)")
    if "Index" not in out.columns:
        out.insert(0, "Index", np.arange(row_offset, row_offset + len(out), dtype=int))
    else:
        out["Index"] = out["Index"].astype(int)
    if "Value" not in out.columns and "Data" in out.columns:
        out = out.rename(columns={"Data": "Value"})
    if "Value" not in out.columns:
        _, vc = get_column_names(out)
        out = out.rename(columns={vc: "Value"})
    return out


def _fit_series_norm_params(values: np.ndarray) -> Dict[str, Any]:
    """Whole-series p5–p95 scaling params (preprocess2-compatible); fallback min–max."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mode": "identity", "scale": 1.0}
    p5, p95 = float(np.percentile(x, 5)), float(np.percentile(x, 95))
    denom = p95 - p5
    if denom <= 1e-12:
        xmin, xmax = float(np.min(x)), float(np.max(x))
        scale = max(xmax - xmin, 1e-12)
        return {
            "mode": "min_max_fallback",
            "p05": p5,
            "p95": p95,
            "xmin": xmin,
            "xmax": xmax,
            "scale": scale,
        }
    return {
        "mode": "p05_p95",
        "p05": p5,
        "p95": p95,
        "scale": float(denom + 1e-12),
    }


def _apply_series_norm(values: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if p.get("mode") == "identity":
        return x
    if p["mode"] == "p05_p95":
        return (x - float(p["p05"])) / float(p["scale"])
    # min_max_fallback on full series
    return (x - float(p["xmin"])) / float(p["scale"])


def _norm_detrend_round(raw_1d: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Same as ``preprocess2.process``: p5–p95 (fallback min–max) → ``signal.detrend(linear)`` → round to 6 decimals."""
    x = np.asarray(raw_1d, dtype=float)
    norm_p = _fit_series_norm_params(x)
    xs = _apply_series_norm(x, norm_p)
    xd = signal.detrend(xs, type="linear")
    return np.round(xd, 6), norm_p


def _segment_plot_ylim(ys: np.ndarray) -> Tuple[float, float]:
    """Same as ``preprocess2._segment_plot_ylim``: y-axis range for in-segment curves."""
    y = np.asarray(ys, dtype=float)
    lo, hi = float(np.min(y)), float(np.max(y))
    tol = 1e-9
    span_raw = hi - lo
    span = max(span_raw, tol)

    if hi <= 1.0 + tol:
        ymax = 1.0
        inner = max(hi - lo, tol)
        pad_lo = max(0.02, min(0.12, 0.12 * max(inner, 0.01)))
        if lo >= -tol:
            ymin = min(0.0, lo) - pad_lo
        else:
            margin_lo = max(0.05 * span, 0.03)
            ymin = lo - margin_lo
        if ymin >= ymax - 1e-6:
            ymin = ymax - 0.25
        return ymin, ymax

    margin = 0.06 * span
    if span_raw <= tol:
        margin = max(0.05, 0.05 * max(abs(lo), abs(hi), 1.0))
    return lo - margin, hi + margin


def _plot_sample_png(
    path: Path,
    df: pd.DataFrame,
    a: int,
    b: int,
    *,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tcol, vcol = get_column_names(df)
    xs = df[tcol].astype(float).to_numpy()
    ys = df[vcol].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(10, 2.8), dpi=120)
    ax.plot(xs, ys, color="0.2", lw=0.85)
    m = (df[tcol].astype(int).to_numpy() >= a) & (df[tcol].astype(int).to_numpy() <= b)
    ax.fill_between(xs[m], ys[m], color="tab:red", alpha=0.35, linewidth=0)
    ax.axvspan(float(a), float(b), color="tab:red", alpha=0.12)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("index")
    ax.set_ylabel("Value")
    ymin, ymax = _segment_plot_ylim(ys)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _context_pad(
    anom_len: int,
    *,
    ratio: float,
    min_pad: int,
    max_pad: int,
) -> int:
    p = int(round(ratio * float(anom_len)))
    return max(min_pad, min(max_pad, p))


def _expand_index_window_to_min_len(
    df0: pd.DataFrame,
    ws: int,
    we: int,
    min_len: int,
) -> Tuple[int, int]:
    """Symmetrically expand row window [ws,we] in df0 until length >= min_len or series boundary."""
    n = len(df0)
    if n <= 0:
        return ws, we
    ws = max(0, min(ws, n - 1))
    we = max(ws, min(we, n - 1))
    while we - ws + 1 < min_len:
        grew = False
        if ws > 0:
            ws -= 1
            grew = True
        if we < n - 1 and we - ws + 1 < min_len:
            we += 1
            grew = True
        if not grew:
            break
    return ws, we


def _process_one_raw_csv_path(
    csv_path: Path,
    manifest: Path,
    series_dir: Path,
    img_dir: Optional[Path],
    *,
    context_ratio: float,
    min_pad: int,
    max_pad: int,
    series_norm: bool,
    min_export_points: int,
    write_sample_images: bool,
) -> int:
    """Slice one raw CSV (Index/Value/Label) into anomaly-window samples under ``manifest`` / ``series``. Returns write count."""
    n = 0
    try:
        df0 = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[skip] {csv_path.name}: {e}", file=sys.stderr)
        return 0
    try:
        df0 = _ensure_index_value_label(df0, row_offset=0)
    except Exception as e:
        print(f"[skip] {csv_path.name}: {e}", file=sys.stderr)
        return 0
    norm_p: Dict[str, Any] = {"mode": "identity", "scale": 1.0}
    if series_norm:
        s = pd.to_numeric(df0["Value"], errors="coerce").interpolate(
            method="linear", limit_direction="both"
        )
        s = s.bfill().ffill()
        raw_v = s.to_numpy(dtype=float)
        if not np.any(np.isfinite(raw_v)):
            print(f"[skip] {csv_path.name}: Value all non-numeric", file=sys.stderr)
            return 0
        df0["Value"], norm_p = _norm_detrend_round(raw_v)
    labels = df0["Label"].astype(int).to_numpy()
    runs = _label_runs_one_based(labels)
    if not runs:
        return 0
    stem = csv_path.stem
    for ri, (a, b) in enumerate(runs):
        anom_len = b - a + 1
        pad = _context_pad(anom_len, ratio=context_ratio, min_pad=min_pad, max_pad=max_pad)
        ws = max(0, a - pad)
        we = min(len(df0) - 1, b + pad)
        ws, we = _expand_index_window_to_min_len(df0, ws, we, min_export_points)
        sl = df0.iloc[ws : we + 1].copy()
        sample_id = f"{stem}__anom{ri:03d}_g{a}_{b}_w{int(sl['Index'].iloc[0])}_{int(sl['Index'].iloc[-1])}"
        csv_out = series_dir / f"{sample_id}.csv"
        sl.to_csv(csv_out, index=False)
        if write_sample_images and img_dir is not None:
            png_out = img_dir / f"{sample_id}.png"
            _plot_sample_png(png_out, sl, a, b, title=f"{sample_id}\nGT [{a},{b}]")
        rec = {
            "sample_id": sample_id,
            "mode": "raw",
            "source_csv": str(csv_path.resolve()),
            "segment_dir": None,
            "anomaly_start": a,
            "anomaly_end": b,
            "window_start": int(sl["Index"].iloc[0]),
            "window_end": int(sl["Index"].iloc[-1]),
            "n_window": len(sl),
            "n_anomaly": anom_len,
            "series_norm": bool(series_norm),
        }
        if series_norm and norm_p.get("mode") != "identity":
            rec["series_norm_params"] = norm_p
        with open(manifest, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
    return n


def export_raw_csv_dir(
    data_dir: Path,
    out_dir: Path,
    *,
    context_ratio: float,
    min_pad: int,
    max_pad: int,
    series_norm: bool,
    min_export_points: int,
    write_sample_images: bool,
) -> int:
    n = 0
    manifest = out_dir / "manifest.jsonl"
    series_dir = out_dir / "series"
    img_dir = out_dir / "images"
    series_dir.mkdir(parents=True, exist_ok=True)
    if write_sample_images:
        img_dir.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        manifest.unlink()

    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSV files found: {data_dir}", file=sys.stderr)
        return 1

    for csv_path in csvs:
        n += _process_one_raw_csv_path(
            csv_path,
            manifest,
            series_dir,
            img_dir if write_sample_images else None,
            context_ratio=context_ratio,
            min_pad=min_pad,
            max_pad=max_pad,
            series_norm=series_norm,
            min_export_points=min_export_points,
            write_sample_images=write_sample_images,
        )
    print(f"[raw] wrote {n} samples -> {out_dir}")
    return 0 if n else 1


def export_raw_csv_file(
    csv_path: Path,
    out_dir: Path,
    *,
    context_ratio: float,
    min_pad: int,
    max_pad: int,
    series_norm: bool,
    min_export_points: int,
    write_sample_images: bool,
) -> int:
    """Same rules as ``export_raw_csv_dir`` but for a **single** ``*.csv`` file."""
    csv_path = csv_path.resolve()
    if not csv_path.is_file():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        return 1
    manifest = out_dir / "manifest.jsonl"
    series_dir = out_dir / "series"
    img_dir = out_dir / "images"
    series_dir.mkdir(parents=True, exist_ok=True)
    if write_sample_images:
        img_dir.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        manifest.unlink()

    n = _process_one_raw_csv_path(
        csv_path,
        manifest,
        series_dir,
        img_dir if write_sample_images else None,
        context_ratio=context_ratio,
        min_pad=min_pad,
        max_pad=max_pad,
        series_norm=series_norm,
        min_export_points=min_export_points,
        write_sample_images=write_sample_images,
    )
    print(f"[raw] wrote {n} samples -> {out_dir}")
    return 0 if n else 1


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    x = x.astype(float)
    y = y.astype(float)
    x = x - x.mean()
    denom = float(np.dot(x, x)) + 1e-12
    return float(np.dot(x, y - y.mean()) / denom)


def featurize_one(rec: Dict[str, Any], series_dir: Path) -> np.ndarray:
    """Fixed-length 18-D feature vector; main clustering path uses this + length-stratified KMeans."""
    sid = rec["sample_id"]
    p = series_dir / f"{sid}.csv"
    df = pd.read_csv(p)
    tcol, vcol = get_column_names(df)
    idx = df[tcol].astype(int).to_numpy()
    v = df[vcol].astype(float).to_numpy()
    lab = df["Label"].astype(int).to_numpy() if "Label" in df.columns else np.zeros(len(df), dtype=int)
    a, b = int(rec["anomaly_start"]), int(rec["anomaly_end"])
    mask_a = (idx >= a) & (idx <= b)
    mask_ctx = ~mask_a
    if not np.any(mask_a):
        return np.zeros(18, dtype=np.float64)
    va = v[mask_a]
    mu_ctx = float(np.mean(v[mask_ctx])) if np.any(mask_ctx) else float(np.mean(v))
    sig_ctx = float(np.std(v[mask_ctx])) if np.any(mask_ctx) else float(np.std(v))
    sig_ctx = max(sig_ctx, 1e-9)
    left = (idx < a) & (lab == 0)
    right = (idx > b) & (lab == 0)
    mu_l = float(np.mean(v[left])) if np.any(left) else mu_ctx
    mu_r = float(np.mean(v[right])) if np.any(right) else mu_ctx
    # slopes at boundaries (within window)
    k = min(12, int(np.sum(mask_a)) + 5)
    pos_a = np.flatnonzero(mask_a)[0]
    pos_b = np.flatnonzero(mask_a)[-1]
    lo = max(0, pos_a - k)
    s_lo = _linear_slope(idx[lo : pos_a + 1], v[lo : pos_a + 1]) if pos_a + 1 - lo >= 2 else 0.0
    hi = min(len(v) - 1, pos_b + k)
    s_hi = _linear_slope(idx[pos_b : hi + 1], v[pos_b : hi + 1]) if hi - pos_b >= 2 else 0.0
    dv = np.diff(va)
    fe = [
        np.log1p(float(rec.get("n_anomaly", np.sum(mask_a)))),
        np.log1p(float(len(v))),
        float(np.sum(mask_a)) / max(len(v), 1),
        float(np.mean(va)),
        float(np.std(va)),
        float(np.max(va) - np.min(va)),
        float((np.max(va) - mu_ctx) / sig_ctx),
        float((mu_ctx - np.min(va)) / sig_ctx),
        float(mu_ctx),
        float(sig_ctx),
        float(mu_l - mu_r),
        float(mu_l - float(np.mean(va))),
        float(mu_r - float(np.mean(va))),
        s_lo,
        s_hi,
        float(np.mean(np.abs(np.diff(v[mask_a])))) if np.sum(mask_a) > 1 else 0.0,
        float(np.mean(np.abs(dv))) if len(dv) else 0.0,
        float(np.sum((va - mu_ctx) ** 2) / (len(va) + 1e-9)),
    ]
    return np.asarray(fe, dtype=np.float64)


def _build_featurize_matrix(
    rows: List[Dict[str, Any]],
    series_dir: Path,
    *,
    show_progress: bool,
) -> np.ndarray:
    it = rows
    if show_progress:
        it = _tqdm_iterable(rows, desc="featurize (18d)", unit="sample")
    mats = [featurize_one(r, series_dir) for r in it]
    return np.stack(mats, axis=0)


def _stratum_from_anomaly_len(L: int, e0: int, e1: int) -> int:
    """``[0,e0)`` -> 0, ``[e0,e1)`` -> 1, ``[e1,inf)`` -> 2 (``L`` = closed anomaly interval length)."""
    if L < int(e0):
        return 0
    if L < int(e1):
        return 1
    return 2


def _cluster_kmeans_stratified_by_length(
    X: np.ndarray,
    rows: List[Dict[str, Any]],
    *,
    k_min: int,
    k_max: int,
    length_stratum_k_max: int,
    seed: int,
    show_progress: bool,
    ksearch_n_init: int,
    refine_n_init: int,
    len_e0: int,
    len_e1: int,
) -> Tuple[np.ndarray, int, np.ndarray, float, List[Dict[str, Any]]]:
    """Length-stratified clustering; each stratum runs ``_cluster_kmeans_silhouette``; global ids concatenated.

    Per-stratum k range is intersection of ``[min(k_min, k_cap), min(k_max, k_cap)]`` with sample count;
    ``k_cap = length_stratum_k_max`` (max clusters per stratum).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = X.shape[0]
    lens = _anomaly_length_points(rows).astype(int)
    strata = np.array(
        [_stratum_from_anomaly_len(int(L), len_e0, len_e1) for L in lens],
        dtype=int,
    )
    labels = np.full(n, -1, dtype=int)
    center_rows: List[np.ndarray] = []
    bin_reports: List[Dict[str, Any]] = []
    global_off = 0
    rng = np.random.default_rng(int(seed))
    k_cap = max(1, int(length_stratum_k_max))
    km_layer_max = min(int(k_max), k_cap)
    km_layer_min = min(int(k_min), km_layer_max)

    for s in (0, 1, 2):
        idx = np.flatnonzero(strata == s)
        n_s = int(idx.size)
        if n_s == 0:
            continue
        span = ("[0,{})".format(len_e0), "[{},{})".format(len_e0, len_e1), "[{e1},inf)".format(e1=len_e1))[s]
        X_sub = X[idx].copy()
        tag = f"length-stratum{s} {span}"

        if n_s == 1:
            labels[idx[0]] = global_off
            center_rows.append(X_sub[0].reshape(1, -1))
            bin_reports.append(
                {
                    "stratum": s,
                    "length_span": span,
                    "n": 1,
                    "k": 1,
                    "silhouette": None,
                    "note": "single sample, one cluster",
                }
            )
            global_off += 1
            continue

        kr = _clamp_k_range_stratum(n_s, km_layer_min, km_layer_max)
        if kr is None:
            from sklearn.preprocessing import StandardScaler

            scaler1 = StandardScaler()
            Xss = scaler1.fit_transform(X_sub)
            km1 = KMeans(
                n_clusters=1,
                random_state=int(rng.integers(0, 2**31 - 1)),
                n_init=int(refine_n_init),
            ).fit(Xss)
            lab = np.asarray(km1.labels_, dtype=int)
            for j, ii in enumerate(idx):
                labels[ii] = global_off + int(lab[j])
            center_rows.append(
                scaler1.inverse_transform(km1.cluster_centers_).astype(np.float64)
            )
            bin_reports.append(
                {
                    "stratum": s,
                    "length_span": span,
                    "n": n_s,
                    "k": 1,
                    "silhouette": None,
                    "note": "k range cannot reach >=2, forced k=1",
                }
            )
            global_off += 1
            continue

        lo, hi = kr
        if show_progress:
            print(
                f"[cluster] KMeans {tag}: n={n_s}, k ∈ [{lo},{hi}] "
                f"(stratum k≤{k_cap}, global k∈[{k_min},{k_max}])",
                file=sys.stderr,
            )
        sub_seed = int(rng.integers(0, 2**31 - 1))
        _k, _model, lab_sub, sc_sub, centers_sub = _cluster_kmeans_silhouette(
            X_sub,
            k_min=lo,
            k_max=hi,
            seed=sub_seed,
            show_progress=show_progress,
            ksearch_n_init=ksearch_n_init,
            refine_n_init=refine_n_init,
        )
        k_loc = int(centers_sub.shape[0])
        for j, ii in enumerate(idx):
            labels[ii] = global_off + int(lab_sub[j])
        center_rows.append(np.asarray(centers_sub, dtype=np.float64))
        bin_reports.append(
            {
                "stratum": s,
                "length_span": span,
                "n": n_s,
                "k": k_loc,
                "silhouette": float(sc_sub) if sc_sub == sc_sub else None,
            }
        )
        global_off += k_loc

    assert np.all(labels >= 0)
    centers_all = np.vstack(center_rows) if center_rows else np.zeros((0, X.shape[1]), dtype=np.float64)
    total_k = int(centers_all.shape[0])
    try:
        sil_all = float(silhouette_score(X, labels, metric="euclidean"))
    except Exception:
        sil_all = float("nan")
    return labels, total_k, centers_all, sil_all, bin_reports


def _clamp_k_range_stratum(
    n: int, k_min: int, k_max: int
) -> Optional[Tuple[int, int]]:
    """Return k search range ``(lo,hi)``; ``None`` if only one cluster is possible."""
    if n < 2:
        return None
    hi = min(int(k_max), int(n))
    lo = min(int(k_min), hi)
    if hi < 2:
        return None
    lo = max(2, lo)
    if lo > hi:
        lo = hi
    return lo, hi


def _anomaly_length_points(rows: List[Dict[str, Any]]) -> np.ndarray:
    """Point count of each sample anomaly closed interval on Index (at least 1)."""
    out: List[float] = []
    for r in rows:
        a, b = int(r["anomaly_start"]), int(r["anomaly_end"])
        out.append(float(max(1, b - a + 1)))
    return np.asarray(out, dtype=np.float64)


def _cluster_kmeans_silhouette(
    X: np.ndarray,
    *,
    k_min: int,
    k_max: int,
    seed: int,
    show_progress: bool = True,
    ksearch_n_init: int = 5,
    refine_n_init: int = 15,
) -> Tuple[int, Any, np.ndarray, float, np.ndarray]:
    """KMeans + silhouette on the same shape embedding; lighter ``n_init`` for k search, then full refine."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n = Xs.shape[0]
    hi = min(int(k_max), n - 1)
    lo = max(2, int(k_min))
    if hi < lo:
        lo = min(2, n - 1)
        hi = max(lo, min(lo, n - 1))
    best_k, best_sc, best_km = lo, -1.0, None
    n_k = sum(1 for k in range(lo, hi + 1) if k < n)
    pbar = None
    if show_progress and n_k > 0:
        try:
            from tqdm import tqdm

            pbar = tqdm(
                total=n_k,
                desc="KMeans k search",
                unit="k",
                leave=True,
                file=sys.stderr,
                mininterval=0.25,
                dynamic_ncols=True,
            )
        except Exception:
            pbar = None
    for k in range(lo, hi + 1):
        if k >= n:
            break
        if pbar is not None:
            pbar.set_postfix_str(f"k={k}", refresh=True)
        km = KMeans(
            n_clusters=k,
            random_state=int(seed),
            n_init=int(ksearch_n_init),
        )
        lab = km.fit_predict(Xs)
        if len(np.unique(lab)) < 2:
            if pbar is not None:
                pbar.update(1)
            continue
        try:
            sc = float(silhouette_score(Xs, lab))
        except Exception:
            if pbar is not None:
                pbar.update(1)
            continue
        if sc > best_sc:
            best_sc, best_k, best_km = sc, k, km
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()
    if best_km is None:
        best_k = lo

    def _kmeans_refit():
        return KMeans(
            n_clusters=int(best_k),
            random_state=int(seed),
            n_init=int(refine_n_init),
        ).fit(Xs)

    refit = _tqdm_single_phase(
        "KMeans full refine",
        f"n={n},k={int(best_k)},ni={refine_n_init}",
        show_progress,
        _kmeans_refit,
    )
    lab = np.asarray(refit.labels_, dtype=int)
    try:
        best_sc = float(silhouette_score(Xs, lab))
    except Exception:
        best_sc = float("nan")
    centers = scaler.inverse_transform(refit.cluster_centers_)
    return best_k, refit, lab, float(best_sc), centers


def _safe_image_name(sample_id: str) -> str:
    return sample_id.replace("/", "_").replace("\\", "_")[:180]


def _anomaly_spans_from_frame(
    df: pd.DataFrame,
    tcol: str,
    fallback: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """All anomaly closed intervals in the current frame on Index; fallback when no Label/anomaly."""
    a0, b0 = int(fallback[0]), int(fallback[1])
    if "Label" not in df.columns:
        return [(a0, b0)]
    ic = df[tcol].astype(int).to_numpy()
    lab = df["Label"].astype(int).to_numpy()
    y = (lab != 0).astype(int)
    spans: List[Tuple[int, int]] = []
    n = len(ic)
    i = 0
    while i < n:
        if y[i] != 1:
            i += 1
            continue
        j = i
        while j < n and y[j] == 1:
            j += 1
        spans.append((int(ic[i]), int(ic[j - 1])))
        i = j
    return spans if spans else [(a0, b0)]


def _figsize_for_n_points(n: int, ref_segment_size: int) -> Tuple[float, float]:
    """Linear scaling of ``figsize=(12,3.2)`` vs ``segment_size`` like ``scripts_new/preprocess.py``."""
    ref = max(1, int(ref_segment_size))
    nn = max(1, int(n))
    w = 12.0 * float(nn) / float(ref)
    w = max(w, 2.4)
    return (w, 3.2)


def _mosaic_grid_ncol_nrow(n: int, *, max_cols: int = 14) -> Tuple[int, int]:
    """Subplot grid for ``cluster_overview`` / ``anomaly_reference``: few rows, many columns.

    - ``n<=6``: single row with ``n`` columns;
    - else prefer **2 rows** (``ncol=ceil(n/2)``); add rows only when columns exceed ``max_cols``.
    """
    if n <= 0:
        return 1, 1
    if n <= 6:
        return n, 1
    nrow = 2
    ncol = int(math.ceil(n / nrow))
    while ncol > max_cols:
        nrow += 1
        ncol = int(math.ceil(n / nrow))
    return ncol, nrow


def _maybe_widen_frame_for_plot(
    rec: Dict[str, Any],
    series_dir: Path,
    min_pts: int,
) -> pd.DataFrame:
    """If exported CSV has < min_pts and manifest has source_csv, widen from source table (legacy exports)."""
    p = series_dir / f"{rec['sample_id']}.csv"
    df = pd.read_csv(p)
    if len(df) >= min_pts:
        return df
    src = rec.get("source_csv")
    if not src:
        return df
    fp = Path(str(src))
    if not fp.is_file():
        return df

    df0 = _ensure_index_value_label(pd.read_csv(fp), 0)
    s = (
        pd.to_numeric(df0["Value"], errors="coerce")
        .interpolate(method="linear", limit_direction="both")
        .bfill()
        .ffill()
    )
    raw_v = s.to_numpy(dtype=float)
    df0 = df0.copy()
    if rec.get("series_norm") and rec.get("series_norm_params"):
        xs = _apply_series_norm(raw_v, rec["series_norm_params"])
        df0["Value"] = np.round(signal.detrend(xs, type="linear"), 6)
    else:
        df0["Value"] = raw_v
    ws, we = int(rec["window_start"]), int(rec["window_end"])
    ic0 = df0["Index"].astype(int).to_numpy()
    pos = np.flatnonzero((ic0 >= ws) & (ic0 <= we))
    if pos.size == 0:
        return df
    lo, hi = int(pos[0]), int(pos[-1])
    lo, hi = _expand_index_window_to_min_len(df0, lo, hi, min_pts)
    return df0.iloc[lo : hi + 1].copy()


def _draw_typical_series_on_ax(
    ax: Any,
    rec: Dict[str, Any],
    df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    legend: bool = True,
    tick_fs: int = 8,
    line_lw: float = 0.85,
) -> None:
    """Draw typical-curve content on existing Axes (vector, no PNG raster)."""
    tcol, vcol = get_column_names(df)
    idx = df[tcol].astype(float).to_numpy()
    v = df[vcol].astype(float).to_numpy()
    ax.plot(idx, v, color="0.2", linewidth=line_lw, label="series")
    a, b = int(rec["anomaly_start"]), int(rec["anomaly_end"])
    for s, e in _anomaly_spans_from_frame(df, tcol, (a, b)):
        ax.axvspan(
            float(s),
            float(e),
            facecolor="#4C72B0",
            alpha=0.28,
            edgecolor="#1A3357",
            linewidth=1.0,
            hatch="///",
        )
    ax.set_xlabel("global index", fontsize=tick_fs)
    ax.set_ylabel("Value", fontsize=tick_fs)
    ax.tick_params(axis="both", labelsize=max(6, tick_fs - 1))
    ymin, ymax = _segment_plot_ylim(v)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.25)
    if legend:
        ax.legend(loc="upper right", fontsize=max(5, tick_fs - 2))
    if title:
        ax.set_title(
            title,
            fontsize=tick_fs + 2,
            color="#4b5563",
            pad=5,
            fontweight="500",
        )


def _render_typical_png(
    out_path: Path,
    rec: Dict[str, Any],
    df: pd.DataFrame,
    *,
    ref_segment_size: int,
    title: Optional[str] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(df)
    fw, fh = _figsize_for_n_points(n, ref_segment_size)
    fig, ax = plt.subplots(figsize=(fw, fh), dpi=110)
    _draw_typical_series_on_ax(ax, rec, df, title=title, legend=True, tick_fs=8, line_lw=0.85)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_cluster_overview_mosaic(
    cdir: Path,
    panel_data: List[Tuple[Dict[str, Any], pd.DataFrame]],
) -> Optional[Path]:
    """Lay out multiple typical curves per cluster like ``anomaly_reference.png`` (gridspec + same style)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    n = len(panel_data)
    if n == 0:
        return None

    ncol, nrow = _mosaic_grid_ncol_nrow(n)

    dpi = 165
    cell_w, cell_h = 3.45, 3.15
    fig_w = ncol * cell_w + 0.35
    fig_h = nrow * cell_h + 0.75
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#eceff4")
    gs = gridspec.GridSpec(
        nrow,
        ncol,
        figure=fig,
        hspace=0.38 if nrow > 1 else 0.22,
        wspace=0.26,
        left=0.06,
        right=0.98,
        top=0.94 if nrow == 1 else 0.95,
        bottom=0.07,
    )
    tick_fs = 7 if ncol >= 5 else 8
    for i in range(nrow * ncol):
        ri, ci = divmod(i, ncol)
        ax = fig.add_subplot(gs[ri, ci])
        if i < n:
            rec, df_plot = panel_data[i]
            _draw_typical_series_on_ax(
                ax,
                rec,
                df_plot,
                title=f"Anomaly {i:02d}",
                legend=True,
                tick_fs=tick_fs,
                line_lw=0.85,
            )
            ax.set_facecolor("#ffffff")
        else:
            ax.set_visible(False)
    out_png = cdir / "cluster_overview.png"
    fig.savefig(
        out_png,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        dpi=dpi,
    )
    plt.close(fig)
    return out_png


def export_typical_cluster_images(
    *,
    samples_dir: Path,
    out_dir: Path,
    rows: List[Dict[str, Any]],
    labels: np.ndarray,
    X_embed: np.ndarray,
    cluster_centers: np.ndarray,
    top_per_cluster: int,
    ref_segment_size: int,
    min_plot_points: int,
    show_progress: bool = True,
    min_cluster_size_for_typical: int = 2,
) -> Dict[str, List[Dict[str, Any]]]:
    """Per cluster, export nearest-to-centroid samples in shape space as preprocess-style PNGs (subplot title ``Anomaly {rank:02d}``);
    also write ``cluster_overview.png`` under each ``cluster_XX/``.

    Clusters with size **below** ``min_cluster_size_for_typical`` skip ``cluster_typical`` and ``anomaly_reference`` (caller skips empty ``typical[c]``).
    """
    ser = samples_dir / "series"
    typ_root = out_dir / "cluster_typical"
    if typ_root.exists():
        shutil.rmtree(typ_root)
    typ_root.mkdir(parents=True, exist_ok=True)
    report: Dict[str, List[Dict[str, Any]]] = {}
    k = int(cluster_centers.shape[0])
    c_iter = range(k)
    if show_progress:
        c_iter = _tqdm_iterable(
            c_iter,
            desc="cluster_typical",
            unit="cluster",
            total=k,
            leave=True,
        )
    min_sz = max(1, int(min_cluster_size_for_typical))
    for c in c_iter:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            report[str(c)] = []
            continue
        if len(idx) < min_sz:
            report[str(c)] = []
            continue
        d = np.linalg.norm(X_embed[idx] - cluster_centers[c], axis=1)
        order = idx[np.argsort(d)]
        m = min(int(top_per_cluster), len(order))
        pick = order[:m]
        cdir = typ_root / f"cluster_{c:02d}"
        cdir.mkdir(parents=True, exist_ok=True)
        entries: List[Dict[str, Any]] = []
        mosaic_panels: List[Tuple[Dict[str, Any], pd.DataFrame]] = []
        for rank, j in enumerate(pick):
            rec = rows[int(j)]
            sid = str(rec["sample_id"])
            dst_name = f"{rank:02d}_{_safe_image_name(sid)}.png"
            dst = cdir / dst_name
            dist = float(np.linalg.norm(X_embed[j] - cluster_centers[c]))
            csv_p = ser / f"{sid}.csv"
            err: Optional[str] = None
            if not csv_p.is_file():
                err = f"missing_series_csv:{csv_p}"
                entries.append(
                    {
                        "sample_id": sid,
                        "rank": rank,
                        "dist_to_center_l2": dist,
                        "image": None,
                        "error": err,
                    }
                )
                continue
            try:
                df_plot = _maybe_widen_frame_for_plot(rec, ser, int(min_plot_points))
                _render_typical_png(
                    dst,
                    rec,
                    df_plot,
                    ref_segment_size=int(ref_segment_size),
                    title=f"Anomaly {rank:02d}",
                )
                mosaic_panels.append((rec, df_plot))
                entries.append(
                    {
                        "sample_id": sid,
                        "rank": rank,
                        "dist_to_center_l2": dist,
                        "image": str(dst.name),
                        "n_points_plotted": int(len(df_plot)),
                    }
                )
            except Exception as e:
                entries.append(
                    {
                        "sample_id": sid,
                        "rank": rank,
                        "dist_to_center_l2": dist,
                        "image": None,
                        "error": str(e),
                    }
                )
        _write_cluster_overview_mosaic(cdir, mosaic_panels)
        report[str(c)] = entries
    return report


def _link_or_copy_file(src: Path, dst: Path) -> str:
    """Create a cheap on-disk reference to src at dst.

    Tries hardlink first (best), then symlink, then falls back to copy.
    Returns: one of {"hardlink","symlink","copy"}.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(str(src), str(dst))
        return "hardlink"
    except Exception:
        pass
    try:
        rel = os.path.relpath(str(src), str(dst.parent))
        dst.symlink_to(rel)
        return "symlink"
    except Exception:
        pass
    shutil.copy2(src, dst)
    return "copy"


def export_cluster_member_series(
    *,
    samples_dir: Path,
    out_dir: Path,
    rows: List[Dict[str, Any]],
    labels: np.ndarray,
    X_embed: np.ndarray,
    cluster_centers: np.ndarray,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Export per-cluster member CSVs into out_dir/cluster_members/cluster_XX/.

    Independent of cluster_typical/anomaly_reference so single-sample clusters
    are still represented on disk.
    """
    ser = samples_dir / "series"
    mem_root = out_dir / "cluster_members"
    if mem_root.exists():
        shutil.rmtree(mem_root)
    mem_root.mkdir(parents=True, exist_ok=True)

    k = int(cluster_centers.shape[0])
    c_iter = range(k)
    if show_progress:
        c_iter = _tqdm_iterable(
            c_iter,
            desc="cluster_members",
            unit="cluster",
            total=k,
            leave=True,
        )

    index_lines = ["cluster,rank,sample_id,dist_to_center_l2,link_mode,member_csv,error\n"]
    link_mode_counts: Dict[str, int] = {"hardlink": 0, "symlink": 0, "copy": 0}
    missing_series = 0

    for c in c_iter:
        idx = np.where(labels == c)[0]
        cdir = mem_root / f"cluster_{c:02d}"
        cdir.mkdir(parents=True, exist_ok=True)
        if len(idx) == 0:
            continue
        d = np.linalg.norm(X_embed[idx] - cluster_centers[c], axis=1)
        order = idx[np.argsort(d)]
        for rank, j in enumerate(order):
            rec = rows[int(j)]
            sid = str(rec["sample_id"])
            src = ser / f"{sid}.csv"
            safe = _safe_image_name(sid)
            dst_name = f"{rank:03d}__{safe}.csv"
            dst = cdir / dst_name
            err = ""
            mode = ""
            if not src.is_file():
                missing_series += 1
                err = f"missing_series_csv:{src}"
            else:
                try:
                    mode = _link_or_copy_file(src, dst)
                    link_mode_counts[mode] = int(link_mode_counts.get(mode, 0)) + 1
                except Exception as e:
                    err = str(e)
            dist = float(np.linalg.norm(X_embed[j] - cluster_centers[c]))
            index_lines.append(
                f"{c},{rank},{sid},{dist:.9f},{mode},{dst_name if mode else ''},{err}\n"
            )

    (mem_root / "index.csv").write_text("".join(index_lines), encoding="utf-8")
    return {
        "dir": str(mem_root),
        "index_csv": "cluster_members/index.csv",
        "link_mode_counts": link_mode_counts,
        "missing_series_csv": int(missing_series),
    }


def _pick_inner_recluster_rank0_row(
    X_embed: np.ndarray,
    centers: np.ndarray,
    c: int,
    idx_rows: np.ndarray,
    *,
    seed: int,
    k_inner_max: int = 3,
) -> int:
    """Second KMeans on ``X_embed`` within cluster; pick largest sub-cluster, then point nearest **outer centroid**. Row index space matches ``rows``."""
    from collections import Counter

    from sklearn.cluster import KMeans

    Xi = X_embed[idx_rows]
    n = int(len(idx_rows))
    outer_c = centers[c]
    if n <= 1:
        return int(idx_rows[0])
    if n == 2:
        d0 = np.linalg.norm(Xi - outer_c, axis=1)
        return int(idx_rows[int(np.argmin(d0))])

    k_cap = max(1, int(k_inner_max))
    k_in = min(k_cap, max(2, n // 3))
    k_in = min(k_in, n - 1)
    if k_in < 2:
        k_in = min(2, n - 1) if n > 2 else 1
    if k_in < 2:
        j0 = int(np.argmin(np.linalg.norm(Xi - outer_c, axis=1)))
        return int(idx_rows[j0])

    km = KMeans(
        n_clusters=k_in,
        random_state=int(seed) + 7 * (int(c) + 1),
        n_init=10,
    ).fit(Xi)
    lab = np.asarray(km.labels_, dtype=int)
    cnt = Counter(lab.tolist())
    maj_label = max(cnt.keys(), key=lambda t: cnt[t])
    mask = lab == maj_label
    sub = idx_rows[mask]
    Xsub = X_embed[sub]
    j_local = int(np.argmin(np.linalg.norm(Xsub - outer_c, axis=1)))
    return int(sub[j_local])


def _build_panels_anomaly_reference_outer(
    k: int,
    typical: Dict[str, List[Dict[str, Any]]],
    rows: List[Dict[str, Any]],
    samples_dir: Path,
    min_plot_points: int,
) -> List[Tuple[int, Dict[str, Any], pd.DataFrame]]:
    """Legacy: per cluster, first successfully loaded rank in ``typical[c]`` (rank0 = nearest to outer centroid)."""
    row_map = {str(r["sample_id"]): r for r in rows}
    ser = samples_dir / "series"
    panels: List[Tuple[int, Dict[str, Any], pd.DataFrame]] = []
    for c in range(int(k)):
        ent = typical.get(str(c)) or []
        placed = False
        for e in ent:
            sid = e.get("sample_id")
            if not sid:
                continue
            rec = row_map.get(str(sid))
            if rec is None:
                continue
            csv_p = ser / f"{sid}.csv"
            if not csv_p.is_file():
                continue
            try:
                df_plot = _maybe_widen_frame_for_plot(rec, ser, int(min_plot_points))
                panels.append((c, rec, df_plot))
                placed = True
                break
            except Exception:
                continue
        if not placed:
            continue
    return panels


def _build_panels_anomaly_reference_inner(
    k: int,
    labels: np.ndarray,
    X_embed: np.ndarray,
    centers: np.ndarray,
    rows: List[Dict[str, Any]],
    samples_dir: Path,
    min_plot_points: int,
    min_cluster_size: int,
    seed: int,
    k_inner_max: int,
) -> List[Tuple[int, Dict[str, Any], pd.DataFrame]]:
    """Pick rank0 via inner re-cluster (independent of ``typical`` dir; still skips clusters below ``min_cluster_size``)."""
    ser = samples_dir / "series"
    panels: List[Tuple[int, Dict[str, Any], pd.DataFrame]] = []
    min_sz = max(1, int(min_cluster_size))
    for c in range(int(k)):
        idx = np.where(labels == c)[0]
        if len(idx) < min_sz:
            continue
        j = _pick_inner_recluster_rank0_row(
            X_embed,
            centers,
            c,
            idx,
            seed=seed,
            k_inner_max=k_inner_max,
        )
        rec = rows[int(j)]
        sid = str(rec["sample_id"])
        csv_p = ser / f"{sid}.csv"
        if not csv_p.is_file():
            continue
        try:
            df_plot = _maybe_widen_frame_for_plot(rec, ser, int(min_plot_points))
            panels.append((c, rec, df_plot))
        except Exception:
            continue
    return panels


def _write_anomaly_reference_figure(
    out_png: Path,
    panels: List[Tuple[int, Dict[str, Any], pd.DataFrame]],
) -> bool:
    """Write ``panels`` (one curve per cluster) as a single mosaic PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    n = len(panels)
    if n == 0:
        return False

    ncol, nrow = _mosaic_grid_ncol_nrow(n)

    dpi = 165
    cell_w, cell_h = 3.45, 3.15
    fig_w = ncol * cell_w + 0.35
    fig_h = nrow * cell_h + 0.75
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#eceff4")
    gs = gridspec.GridSpec(
        nrow,
        ncol,
        figure=fig,
        hspace=0.38 if nrow > 1 else 0.22,
        wspace=0.26,
        left=0.06,
        right=0.98,
        top=0.94 if nrow == 1 else 0.95,
        bottom=0.07,
    )
    tick_fs = 7 if ncol >= 5 else 8
    for i in range(nrow * ncol):
        ri, ci = divmod(i, ncol)
        ax = fig.add_subplot(gs[ri, ci])
        if i < n:
            c, rec, df_plot = panels[i]
            _draw_typical_series_on_ax(
                ax,
                rec,
                df_plot,
                title=f"Anomaly {int(c)}",
                legend=True,
                tick_fs=tick_fs,
                line_lw=0.85,
            )
            ax.set_facecolor("#ffffff")
        else:
            ax.set_visible(False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none", dpi=dpi)
    plt.close(fig)
    return True


def write_anomaly_reference_mosaic(
    out_dir: Path,
    k: int,
    typical: Dict[str, List[Dict[str, Any]]],
    rows: List[Dict[str, Any]],
    samples_dir: Path,
    ref_segment_size: int,
    min_plot_points: int,
) -> bool:
    """Vector-redraw each cluster rank0 typical on one Figure (no PNG stitch) to avoid blur/whitespace.

    rank0: first entry per cluster in ``typical`` from ``export_typical_cluster_images`` (nearest outer-centroid L2).
    """
    del ref_segment_size  # signature compat; widening via ``_maybe_widen_frame_for_plot``
    panels = _build_panels_anomaly_reference_outer(
        k, typical, rows, samples_dir, min_plot_points
    )
    return _write_anomaly_reference_figure(out_dir / "anomaly_reference.png", panels)


def write_anomaly_reference_mosaic_inner(
    out_dir: Path,
    k: int,
    labels: np.ndarray,
    X_embed: np.ndarray,
    centers: np.ndarray,
    rows: List[Dict[str, Any]],
    samples_dir: Path,
    ref_segment_size: int,
    min_plot_points: int,
    min_cluster_size: int,
    seed: int,
    k_inner_max: int = 3,
) -> bool:
    """Companion to ``anomaly_reference.png``: inner KMeans then rank0 → ``anomaly_reference_inner.png``."""
    del ref_segment_size
    panels = _build_panels_anomaly_reference_inner(
        k,
        labels,
        X_embed,
        centers,
        rows,
        samples_dir,
        min_plot_points,
        min_cluster_size,
        seed,
        k_inner_max,
    )
    return _write_anomaly_reference_figure(out_dir / "anomaly_reference_inner.png", panels)


def run_cluster_from_dir(samples_dir: Path, out_dir: Path, args: argparse.Namespace) -> int:
    """Cluster ``samples_dir`` (manifest.jsonl / series/) and write results to ``out_dir``."""
    root = samples_dir.resolve()
    out = out_dir.resolve()
    man = root / "manifest.jsonl"
    ser = root / "series"
    if not man.is_file() or not ser.is_dir():
        print(f"Expected {man} and {ser}/", file=sys.stderr)
        return 2
    rows = _read_manifest(man)
    if len(rows) < args.k_min:
        print(f"Sample count {len(rows)} < k_min={args.k_min}", file=sys.stderr)
        return 1

    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("Required: pip install scikit-learn", file=sys.stderr)
        return 1

    show_progress = not bool(getattr(args, "no_progress", False))

    k_max = min(int(args.k_max), len(rows) - 1)
    k_max = max(1, k_max)
    k_min = min(max(2, int(args.k_min)), k_max)

    len_e0 = int(getattr(args, "length_stratum_e0", 10))
    len_e1 = int(getattr(args, "length_stratum_e1", 100))
    length_stratum_k_max = max(1, int(getattr(args, "length_stratum_k_max", 3)))
    X = _build_featurize_matrix(rows, ser, show_progress=show_progress)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    method = "kmeans_featurize18_length_strata"
    best_labels, best_k, centers, best_score, length_bins_report = (
        _cluster_kmeans_stratified_by_length(
            X,
            rows,
            k_min=k_min,
            k_max=k_max,
            length_stratum_k_max=length_stratum_k_max,
            seed=int(args.seed),
            show_progress=show_progress,
            ksearch_n_init=int(getattr(args, "kmeans_ksearch_n_init", 5)),
            refine_n_init=int(getattr(args, "kmeans_refine_n_init", 15)),
            len_e0=len_e0,
            len_e1=len_e1,
        )
    )

    best_labels = np.asarray(best_labels, dtype=int)

    out.mkdir(parents=True, exist_ok=True)
    lines_csv = ["sample_id,cluster,k\n"]
    for r, lb in zip(rows, best_labels):
        lines_csv.append(f"{r['sample_id']},{int(lb)},{best_k}\n")
    (out / "assignments.csv").write_text("".join(lines_csv), encoding="utf-8")

    # Export per-cluster member CSVs (including single-sample clusters) so they
    # are still visible on disk even if excluded from mosaic plots.
    mem_report = None
    if not bool(getattr(args, "no_cluster_members", False)):
        mem_report = export_cluster_member_series(
            samples_dir=root,
            out_dir=out,
            rows=rows,
            labels=best_labels,
            X_embed=X,
            cluster_centers=centers,
            show_progress=show_progress,
        )

    sil_json: Any = best_score
    if isinstance(best_score, float) and best_score != best_score:
        sil_json = None
    summary: Dict[str, Any] = {
        "n_samples": len(rows),
        "k_chosen": int(best_k),
        "silhouette": sil_json,
        "silhouette_note": "Silhouette on merged cluster labels over full corpus (Euclidean)",
        "k_range_searched": [k_min, k_max],
        "length_stratum_k_max": int(length_stratum_k_max),
        "per_layer_effective_k_range": [
            min(k_min, length_stratum_k_max),
            min(k_max, length_stratum_k_max),
        ],
        "cluster_sizes": {str(c): int(np.sum(best_labels == c)) for c in range(best_k)},
        "clustering_method": method,
        "cluster_speed": {
            "kmeans_ksearch_n_init": int(getattr(args, "kmeans_ksearch_n_init", 5)),
            "kmeans_refine_n_init": int(getattr(args, "kmeans_refine_n_init", 15)),
        },
        "featurize18": {
            "source": "featurize_one (per-window stats on series CSV)",
            "dims": 18,
            "length_strata_edges": [len_e0, len_e1],
            "length_strata": length_bins_report,
        },
        "feature_names": [f"feat_{i}" for i in range(18)],
        "samples_dir": str(root),
        "cluster_typical_ref_segment_size": int(args.ref_segment_size),
        "cluster_typical_min_plot_points": max(1, int(args.min_plot_points)),
    }
    if mem_report is not None:
        summary["cluster_members"] = mem_report
    min_typ_sz = max(1, int(getattr(args, "min_cluster_size_for_typical", 2)))
    summary["cluster_typical_min_cluster_size"] = min_typ_sz
    summary["cluster_typical_excluded_cluster_ids"] = [
        int(c)
        for c in range(int(best_k))
        if int(np.sum(best_labels == c)) < min_typ_sz
    ]
    if not args.no_cluster_typical and int(args.typical_per_cluster) > 0:
        typical = export_typical_cluster_images(
            samples_dir=root,
            out_dir=out,
            rows=rows,
            labels=best_labels,
            X_embed=X,
            cluster_centers=centers,
            top_per_cluster=int(args.typical_per_cluster),
            ref_segment_size=int(args.ref_segment_size),
            min_plot_points=max(1, int(args.min_plot_points)),
            show_progress=show_progress,
            min_cluster_size_for_typical=min_typ_sz,
        )
        summary["cluster_typical"] = typical
        summary["cluster_typical_dir"] = str(out / "cluster_typical")
        cov: Dict[str, str] = {}
        for c in range(int(best_k)):
            rel = Path("cluster_typical") / f"cluster_{c:02d}" / "cluster_overview.png"
            if (out / rel).is_file():
                cov[str(c)] = str(rel).replace("\\", "/")
        if cov:
            summary["cluster_overview_pngs"] = cov
        if write_anomaly_reference_mosaic(
            out,
            int(best_k),
            typical,
            rows,
            root,
            int(args.ref_segment_size),
            max(1, int(args.min_plot_points)),
        ):
            summary["anomaly_reference_png"] = str(out / "anomaly_reference.png")
        summary["anomaly_reference_selection_note"] = (
            "anomaly_reference.png: per cluster, nearest to outer KMeans centroid in 18-D feature space (typical rank0); "
            "anomaly_reference_inner.png: after inner KMeans, largest sub-cluster, then nearest to outer centroid."
        )
        if bool(getattr(args, "inner_recluster_reference", True)):
            k_in_max = max(2, int(getattr(args, "reference_inner_k_max", 3)))
            if write_anomaly_reference_mosaic_inner(
                out,
                int(best_k),
                best_labels,
                X,
                centers,
                rows,
                root,
                int(args.ref_segment_size),
                max(1, int(args.min_plot_points)),
                min_typ_sz,
                int(args.seed),
                k_in_max,
            ):
                summary["anomaly_reference_inner_png"] = str(
                    out / "anomaly_reference_inner.png"
                )
                summary["reference_inner_k_max"] = k_in_max
    with open(out / "cluster_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if len(rows) >= 8 and best_k >= 2:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler

        pca = PCA(n_components=2, random_state=args.seed)
        X_vis = StandardScaler().fit_transform(X)
        Z = pca.fit_transform(X_vis)
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=120)
        for c in range(best_k):
            m = best_labels == c
            ax.scatter(Z[m, 0], Z[m, 1], s=22, alpha=0.75, label=f"Anomaly {c}")
        ax.set_title(
            f"PCA of featurize-18 ({method}, k={best_k}, sil≈{best_score:.3f})"
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "pca_scatter.png", bbox_inches="tight")
        plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote:", out / "assignments.csv")
    if not args.no_cluster_typical and int(args.typical_per_cluster) > 0:
        print("wrote:", out / "cluster_typical")
        for c in range(int(best_k)):
            co = out / "cluster_typical" / f"cluster_{c:02d}" / "cluster_overview.png"
            if co.is_file():
                print("wrote:", co)
        ar = out / "anomaly_reference.png"
        if ar.is_file():
            print("wrote:", ar)
        ar_in = out / "anomaly_reference_inner.png"
        if ar_in.is_file():
            print("wrote:", ar_in)
    return 0


def build_parser(defaults: ClusterPipelineConfig = DEFAULT_CONFIG) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Export anomaly windows from raw CSV dir/file and cluster.",
    )
    ap.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Directory of *.csv (Index/Value/Label); mutually exclusive with -c",
    )
    ap.add_argument(
        "-c",
        "--csv",
        dest="csv_file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Single raw CSV path instead of input_dir",
    )
    ap.add_argument("-o", "--output-dir", type=Path, required=True, help="Root output directory for export + clustering")
    ap.add_argument("--context-ratio", type=float, default=defaults.context_ratio, help="Context expansion ratio on both sides of anomaly window.")
    ap.add_argument("--single-min-pad", type=int, default=defaults.single_min_pad, help="Minimum padding in single-file mode.")
    ap.add_argument("--single-max-pad", type=int, default=defaults.single_max_pad, help="Maximum padding in single-file mode.")
    ap.add_argument("--dir-min-pad", type=int, default=defaults.dir_min_pad, help="Minimum padding in directory mode.")
    ap.add_argument("--dir-max-pad", type=int, default=defaults.dir_max_pad, help="Maximum padding in directory mode.")
    ap.add_argument("--single-min-export-points", type=int, default=defaults.single_min_export_points)
    ap.add_argument("--dir-min-export-points", type=int, default=defaults.dir_min_export_points)
    ap.add_argument("--seed", type=int, default=defaults.seed, help="Clustering random seed.")
    ap.add_argument("--k-min", type=int, default=defaults.k_min)
    ap.add_argument("--k-max", type=int, default=defaults.k_max)
    ap.add_argument("--typical-per-cluster", type=int, default=defaults.typical_per_cluster)
    ap.add_argument("--ref-segment-size", type=int, default=defaults.ref_segment_size)
    ap.add_argument("--min-plot-points", type=int, default=defaults.min_plot_points)
    ap.add_argument("--min-cluster-size-for-typical", type=int, default=defaults.min_cluster_size_for_typical)
    ap.add_argument(
        "--no-cluster-members",
        action="store_true",
        default=False,
        help="Skip cluster_members/ export (default exports per-cluster member series CSVs, including singletons).",
    )
    ap.add_argument("--reference-inner-k-max", type=int, default=defaults.reference_inner_k_max)
    ap.add_argument("--length-stratum-e0", type=int, default=defaults.length_stratum_e0)
    ap.add_argument("--length-stratum-e1", type=int, default=defaults.length_stratum_e1)
    ap.add_argument("--length-stratum-k-max", type=int, default=defaults.length_stratum_k_max)
    ap.add_argument("--kmeans-ksearch-n-init", type=int, default=defaults.kmeans_ksearch_n_init)
    ap.add_argument("--kmeans-refine-n-init", type=int, default=defaults.kmeans_refine_n_init)
    ap.add_argument("--write-sample-images", action="store_true", default=defaults.write_sample_images)
    ap.add_argument("--no-series-norm", action="store_true", help="Disable whole-series p5-p95 normalization + detrend.")
    ap.add_argument("--disable-inner-recluster-reference", action="store_true", help="Disable inner re-cluster reference image.")
    ap.add_argument("--no-cluster-typical", action="store_true", default=defaults.no_cluster_typical)
    ap.add_argument("--no-progress", action="store_true", default=defaults.no_progress)
    return ap


def parse_args(
    argv: Optional[Sequence[str]] = None,
    defaults: ClusterPipelineConfig = DEFAULT_CONFIG,
) -> argparse.Namespace:
    return build_parser(defaults).parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ClusterPipelineConfig:
    return ClusterPipelineConfig(
        context_ratio=float(args.context_ratio),
        single_min_pad=int(args.single_min_pad),
        single_max_pad=int(args.single_max_pad),
        dir_min_pad=int(args.dir_min_pad),
        dir_max_pad=int(args.dir_max_pad),
        single_min_export_points=int(args.single_min_export_points),
        dir_min_export_points=int(args.dir_min_export_points),
        series_norm=not bool(args.no_series_norm),
        write_sample_images=bool(args.write_sample_images),
        k_min=int(args.k_min),
        k_max=int(args.k_max),
        seed=int(args.seed),
        typical_per_cluster=int(args.typical_per_cluster),
        ref_segment_size=int(args.ref_segment_size),
        min_plot_points=int(args.min_plot_points),
        min_cluster_size_for_typical=int(args.min_cluster_size_for_typical),
        inner_recluster_reference=not bool(args.disable_inner_recluster_reference),
        reference_inner_k_max=int(args.reference_inner_k_max),
        no_cluster_typical=bool(args.no_cluster_typical),
        no_progress=bool(args.no_progress),
        length_stratum_e0=int(args.length_stratum_e0),
        length_stratum_e1=int(args.length_stratum_e1),
        length_stratum_k_max=int(args.length_stratum_k_max),
        kmeans_ksearch_n_init=int(args.kmeans_ksearch_n_init),
        kmeans_refine_n_init=int(args.kmeans_refine_n_init),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.csv_file is not None and args.input_dir is not None:
        print("Specify only one of: input_dir or -c/--csv", file=sys.stderr)
        return 2
    if args.csv_file is None and args.input_dir is None:
        print("Provide input_dir or -c/--csv for a single CSV", file=sys.stderr)
        return 2

    print(
        "[export] preprocess2-style: p5–p95 scale + linear detrend on full series; mode=raw",
        file=sys.stderr,
    )
    print(
        f"[config] out={out} context_ratio={config.context_ratio} "
        f"k=[{config.k_min},{config.k_max}] seed={config.seed} "
        f"ref_segment_size={config.ref_segment_size}",
        file=sys.stderr,
    )
    if args.csv_file is not None:
        cf = args.csv_file.resolve()
        if not cf.is_file():
            print(f"CSV not found: {cf}", file=sys.stderr)
            return 2
        rc = export_raw_csv_file(
            cf,
            out,
            context_ratio=config.context_ratio,
            min_pad=config.single_min_pad,
            max_pad=config.single_max_pad,
            series_norm=config.series_norm,
            min_export_points=config.single_min_export_points,
            write_sample_images=config.write_sample_images,
        )
    else:
        inp = args.input_dir.resolve()
        if not inp.is_dir():
            print(f"Input directory not found: {inp}", file=sys.stderr)
            return 2
        rc = export_raw_csv_dir(
            inp,
            out,
            context_ratio=config.context_ratio,
            min_pad=config.dir_min_pad,
            max_pad=config.dir_max_pad,
            series_norm=config.series_norm,
            min_export_points=config.dir_min_export_points,
            write_sample_images=config.write_sample_images,
        )
    if rc != 0:
        return rc

    cluster_opts = argparse.Namespace(
        k_min=config.k_min,
        k_max=config.k_max,
        seed=config.seed,
        typical_per_cluster=config.typical_per_cluster,
        ref_segment_size=config.ref_segment_size,
        min_plot_points=config.min_plot_points,
        min_cluster_size_for_typical=config.min_cluster_size_for_typical,
        inner_recluster_reference=config.inner_recluster_reference,
        reference_inner_k_max=config.reference_inner_k_max,
        no_cluster_typical=config.no_cluster_typical,
        no_progress=config.no_progress,
        length_stratum_e0=config.length_stratum_e0,
        length_stratum_e1=config.length_stratum_e1,
        length_stratum_k_max=config.length_stratum_k_max,
        kmeans_ksearch_n_init=config.kmeans_ksearch_n_init,
        kmeans_refine_n_init=config.kmeans_refine_n_init,
    )
    return run_cluster_from_dir(out, out, cluster_opts)


if __name__ == "__main__":
    raise SystemExit(main())
