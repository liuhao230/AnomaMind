"""
Preprocess time series data: normalize, segment, and write per-segment CSV/metadata.

Normalization: whole-series **5th/95th percentile** linear scale toward [0,1] via (x - p5) / (p95 - p5),
with min–max fallback; then linear detrend (not p10/p90 or other percentile schemes).

Segment preview PNGs use dpi=165, matching cluster ``anomaly_reference.png``, to reduce blur in ``segment_final`` stitching.
"""
import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

# Preview / composite PNG styling (fixed; not on CLI)
PREVIEW_FIGSIZE = (12.0, 3.2)
PREVIEW_DPI = 165
PREVIEW_LINEWIDTH = 0.85
PREVIEW_GRID_ALPHA = 0.25
PREVIEW_TITLE_FONTSIZE = 9
PREVIEW_LEGEND_FONTSIZE = 7
PREVIEW_NORMALIZED_UPPER_BOUND = 1.0
SEGMENT_FINAL_GAP_PX = 10
SEGMENT_FINAL_BACKGROUND_RGB = (236, 239, 244)

DETREND_TYPE = "linear"


@dataclass(frozen=True)
class PreprocessConfig:
    segment_size: int = 100
    sample_ratio: float = 1.0
    anomaly_ratio: float = 0.7
    max_if_normal: int = 0
    random_seed: int = 42
    normalization_lower_percentile: float = 5.0
    normalization_upper_percentile: float = 95.0
    normalization_eps: float = 1e-12
    detrend_enabled: bool = True
    round_decimals: int = 6


DEFAULT_CONFIG = PreprocessConfig()


def _build_parser(defaults: PreprocessConfig = DEFAULT_CONFIG) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess time series CSV files: normalize, segment, and write segment folders."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Single time-series CSV file or directory of *.csv files",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output folder for processed samples",
    )
    parser.add_argument(
        "--segment-size",
        "--segment_size",
        type=int,
        default=defaults.segment_size,
        help="Segment size (default: 100)",
    )
    parser.add_argument(
        "--sample-ratio",
        "--sample_ratio",
        type=float,
        default=defaults.sample_ratio,
    )
    parser.add_argument(
        "--anomaly-ratio",
        "--anomaly_ratio",
        type=float,
        default=defaults.anomaly_ratio,
    )
    parser.add_argument(
        "--max-if-normal",
        "--max_if_normal",
        type=int,
        default=defaults.max_if_normal,
        help=(
            "When no anomaly segments exist after slicing (A==0), cap normal segments kept. "
            "0 = skip the series (default); -1 = no cap."
        ),
    )
    parser.add_argument(
        "--dataset_type",
        "--type",
        dest="dataset_type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume: if _global/data/*__full.csv exists, reload Value/Label and skip raw CSV normalization;"
            "if masks/*_sample.csv exists, reuse Sampled=1 segments without resampling;"
            "only fill missing meta / preview images."
        ),
    )
    parser.add_argument(
        "--no-detrend",
        action="store_true",
        help="Disable linear detrending after normalization.",
    )
    parser.add_argument(
        "--anomaly-reference-png",
        type=Path,
        default=None,
        help=(
            "Optional cluster anomaly_reference.png; when present, also write segment_final.png "
            "(reference mosaic on top, segment_preview_plain below, same width)."
        ),
    )
    return parser


def parse_args(
    argv: Optional[Sequence[str]] = None,
    defaults: PreprocessConfig = DEFAULT_CONFIG,
) -> argparse.Namespace:
    parser = _build_parser(defaults)
    args = parser.parse_args(argv)
    if args.anomaly_reference_png is None:
        env_ref = os.environ.get("ANOMALY_REFERENCE_PNG", "").strip()
        if env_ref:
            args.anomaly_reference_png = Path(env_ref)
    return args


def config_from_args(args: argparse.Namespace) -> PreprocessConfig:
    return PreprocessConfig(
        segment_size=int(args.segment_size),
        sample_ratio=float(args.sample_ratio),
        anomaly_ratio=float(args.anomaly_ratio),
        max_if_normal=int(args.max_if_normal),
        detrend_enabled=not bool(args.no_detrend),
    )


def validate_config(config: PreprocessConfig) -> None:
    if config.segment_size <= 0:
        raise ValueError("segment_size must be positive")
    if not 0 <= config.sample_ratio <= 1:
        raise ValueError("sample_ratio must be in [0, 1]")
    if not 0 <= config.anomaly_ratio <= 1:
        raise ValueError("anomaly_ratio must be in [0, 1]")
    if int(config.max_if_normal) < -1:
        raise ValueError("max_if_normal must be >= -1")
    if config.normalization_lower_percentile >= config.normalization_upper_percentile:
        raise ValueError("normalization_lower_percentile must be smaller than normalization_upper_percentile")
    if config.normalization_eps <= 0:
        raise ValueError("normalization_eps must be positive")
    if config.round_decimals < 0:
        raise ValueError("round_decimals must be >= 0")


def _label_spans_in_slice(labels: np.ndarray, base: int) -> list[tuple[int, int]]:
    """labels is in-segment 0/1 array; return global closed interval [s,e]."""
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(labels)
    while i < n:
        if int(labels[i]) != 1:
            i += 1
            continue
        j = i
        while j < n and int(labels[j]) == 1:
            j += 1
        spans.append((base + i, base + j - 1))
        i = j
    return spans


def _segment_plot_ylim(ys: np.ndarray, *, normalized_upper_bound: float) -> tuple[float, float]:
    """Y-axis range for in-segment curves.

    When peaks stay <= 1, **fix ymax at 1** (consistent with p5–p95 normalization),
    avoiding adaptive scaling when detrend pushes lo slightly below 0.

    If values exceed 1, use data min/max with padding.
    """
    y = np.asarray(ys, dtype=float)
    lo, hi = float(np.min(y)), float(np.max(y))
    tol = 1e-9
    span_raw = hi - lo
    span = max(span_raw, tol)

    if hi <= normalized_upper_bound + tol:
        ymax = normalized_upper_bound
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


def _write_segment_preview_pngs(
    sample_folder: Path,
    *,
    values: np.ndarray,
    labels: np.ndarray,
    start: int,
    end: int,
    series_name: str,
    seg_idx: int,
) -> None:
    """Write two images per segment dir: Label=1 highlight / plain curve only."""
    ys = values[start:end].astype(float)
    xs = np.arange(start, end, dtype=int)
    lab = labels[start:end].astype(int)

    def _one_fig(*, with_labels: bool, out_name: str, title_suffix: str) -> None:
        fig, ax = plt.subplots(figsize=PREVIEW_FIGSIZE, dpi=PREVIEW_DPI)
        ax.plot(xs, ys, color="0.2", linewidth=PREVIEW_LINEWIDTH, label="normalized+detrend")
        if with_labels:
            for s, e in _label_spans_in_slice(lab, start):
                ax.axvspan(
                    s,
                    e,
                    facecolor="#4C72B0",
                    alpha=0.28,
                    edgecolor="#1A3357",
                    linewidth=1.0,
                    hatch="///",
                )
            ax.set_title(
                f"{series_name}  seg{seg_idx:03d}  [{start}, {end})  {title_suffix}",
                fontsize=PREVIEW_TITLE_FONTSIZE,
            )
        # plain image: no title, for Locator multimodal steps
        ax.set_xlabel("global index")
        ax.set_ylabel("Value")
        ymin, ymax = _segment_plot_ylim(ys, normalized_upper_bound=PREVIEW_NORMALIZED_UPPER_BOUND)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=PREVIEW_GRID_ALPHA)
        ax.legend(loc="upper right", fontsize=PREVIEW_LEGEND_FONTSIZE)
        fig.tight_layout()
        fig.savefig(
            sample_folder / out_name,
            bbox_inches="tight",
            dpi=PREVIEW_DPI,
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)

    _one_fig(with_labels=True, out_name="segment_preview_labeled.png", title_suffix="(Label=1 shaded)")
    _one_fig(with_labels=False, out_name="segment_preview_plain.png", title_suffix="")


def _maybe_write_segment_final_png(sample_folder: Path, ref_png: Path) -> None:
    """Write ``segment_final.png``: reference on top, ``segment_preview_plain`` below, widths aligned."""
    plain_path = sample_folder / "segment_preview_plain.png"
    out_path = sample_folder / "segment_final.png"
    if not plain_path.is_file() or not ref_png.is_file():
        return
    try:
        from PIL import Image
    except ImportError:
        print(
            "[warn] Pillow not installed; cannot build segment_final.png — pip install pillow",
            flush=True,
        )
        return
    gap = SEGMENT_FINAL_GAP_PX
    bg_rgb = SEGMENT_FINAL_BACKGROUND_RGB
    im_plain = Image.open(plain_path).convert("RGBA")
    im_ref = Image.open(ref_png).convert("RGBA")
    w = im_plain.width
    if w < 1:
        return
    new_ref_h = max(1, int(round(im_ref.height * (w / float(im_ref.width)))))
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS  # type: ignore[attr-defined]
    im_ref_s = im_ref.resize((w, new_ref_h), resample)
    total_h = im_ref_s.height + gap + im_plain.height
    canvas = Image.new("RGB", (w, total_h), bg_rgb)
    canvas.paste(im_ref_s.convert("RGB"), (0, 0))
    canvas.paste(im_plain.convert("RGB"), (0, im_ref_s.height + gap))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", optimize=True)


def extract_sequence_id(name: str) -> str:
    """Extract sequence ID from filename."""
    parts = name.split("_")
    if len(parts) > 0:
        try:
            int(parts[0])
            return parts[0]
        except ValueError:
            pass
    return name


def _global_stats_from_series(x: np.ndarray, norm_info: dict) -> dict:
    diff_x = np.diff(x)
    return {
        **norm_info,
        "mean": float(np.round(np.mean(x), 6)),
        "std": float(np.round(np.std(x), 6)),
        "min": float(np.round(np.min(x), 6)),
        "max": float(np.round(np.max(x), 6)),
        "median": float(np.round(np.median(x), 6)),
        "25th_percentile": float(np.round(np.percentile(x, 25), 6)),
        "75th_percentile": float(np.round(np.percentile(x, 75), 6)),
        "90th_percentile": float(np.round(np.percentile(x, 90), 6)),
        "95th_percentile": float(np.round(np.percentile(x, 95), 6)),
        "99th_percentile": float(np.round(np.percentile(x, 99), 6)),
        "data_points": len(x),
        "iqr": float(np.round(np.percentile(x, 75) - np.percentile(x, 25), 6)),
        "diff_mean": float(np.round(np.mean(diff_x), 6)) if len(diff_x) > 0 else 0.0,
        "diff_std": float(np.round(np.std(diff_x), 6)) if len(diff_x) > 0 else 0.0,
        "note": "Statistics on final series x after norm (see norm_mode) + linear detrend. Label unchanged.",
    }


def process(
    csv_path: Path,
    output_dir: Path,
    dataset_type: str,
    config: PreprocessConfig = DEFAULT_CONFIG,
    *,
    resume: bool = False,
    anomaly_reference_png: Optional[Path] = None,
) -> None:
    validate_config(config)
    name = csv_path.stem
    sequence_id = extract_sequence_id(name)
    g_data_dir = output_dir / "_global" / "data"
    g_stat_dir = output_dir / "_global" / "statistics"
    g_data_dir.mkdir(parents=True, exist_ok=True)
    g_stat_dir.mkdir(parents=True, exist_ok=True)
    full_csv = g_data_dir / f"{name}__full.csv"
    global_stats_path = g_stat_dir / f"{name}_global_statistics.json"
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"{sequence_id}_sample.csv"

    loaded_from_full = False
    if resume and full_csv.is_file():
        df_full = pd.read_csv(full_csv)
        if {"Index", "Value", "Label"}.issubset(df_full.columns):
            data = df_full["Value"].astype(float).to_numpy()
            labels = df_full["Label"].astype(int).to_numpy()
            N = len(data)
            loaded_from_full = True
            print(f"[resume] {csv_path.name}: load existing {full_csv.name} (N={N}), skip normalize")
        else:
            print(f"[resume] {csv_path.name}: {full_csv.name} columns invalid, fall back to full preprocess")

    if not loaded_from_full:
        df = pd.read_csv(csv_path)
        if not {"Data", "Label"}.issubset(df.columns):
            print(f"Skipping {csv_path.name} (missing required columns)")
            return

        # Impute missing / non-finite values so min-max + detrend (lstsq) never sees NaN/Inf
        data_series = pd.to_numeric(df["Data"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        data_series = data_series.interpolate(method="linear", limit_direction="both")
        data_series = data_series.bfill().ffill()
        if data_series.isna().any():
            fill = 0.0 if data_series.isna().all() else float(data_series.dropna().median())
            data_series = data_series.fillna(fill)

        # Normalize (5th–95th percentile range, robust to outliers) + linear detrend
        x_raw = data_series.values.astype(float)
        p_low = float(np.percentile(x_raw, config.normalization_lower_percentile))
        p_high = float(np.percentile(x_raw, config.normalization_upper_percentile))
        denom = p_high - p_low
        if denom <= config.normalization_eps:
            xmin, xmax = float(np.min(x_raw)), float(np.max(x_raw))
            x_scaled = (x_raw - xmin) / (xmax - xmin + config.normalization_eps)
            norm_info: dict = {
                "norm_mode": "min_max_fallback",
                "reason": (
                    f"p{config.normalization_upper_percentile:g} - "
                    f"p{config.normalization_lower_percentile:g} <= normalization_eps"
                ),
                "xmin": xmin,
                "xmax": xmax,
            }
        else:
            x_scaled = (x_raw - p_low) / (denom + config.normalization_eps)
            norm_info = {
                "norm_mode": (
                    f"p{config.normalization_lower_percentile:05.2f}_"
                    f"p{config.normalization_upper_percentile:05.2f}"
                ),
                "lower_percentile": float(config.normalization_lower_percentile),
                "upper_percentile": float(config.normalization_upper_percentile),
                "p_low": p_low,
                "p_high": p_high,
                "scale": float(denom),
            }
        x = (
            signal.detrend(x_scaled, type=DETREND_TYPE)
            if config.detrend_enabled
            else x_scaled
        )
        df["Data"] = np.round(x, config.round_decimals)

        global_stats = _global_stats_from_series(x, norm_info)
        labels = df["Label"].values.astype(int)
        data = df["Data"].values.astype(float)
        N = len(data)

        if not full_csv.exists():
            pd.DataFrame(
                {
                    "Index": np.arange(N, dtype=int),
                    "Value": data.astype(float),
                    "Label": labels.astype(int),
                }
            ).to_csv(full_csv, index=False)

        if not global_stats_path.exists():
            try:
                with open(global_stats_path, "w", encoding="utf-8") as f:
                    json.dump(global_stats, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] failed to write global statistics for {csv_path.name}: {e}")

    # Same as legacy pandas logic: row position is global index (0..N-1)
    anomalies = np.flatnonzero(labels == 1)
    num_segments = N // config.segment_size

    if loaded_from_full:
        if not global_stats_path.exists():
            x2 = data.astype(float)
            norm_info2: dict = {"norm_mode": "unknown_resumed_from_full_csv"}
            gs = _global_stats_from_series(x2, norm_info2)
            try:
                with open(global_stats_path, "w", encoding="utf-8") as f:
                    json.dump(gs, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] failed to write global statistics for {csv_path.name}: {e}")

    segment_indices: list[int] = []
    reused_mask = False
    if resume and mask_path.is_file():
        try:
            mdf = pd.read_csv(mask_path)
            if "Sampled" in mdf.columns and len(mdf) == N:
                sampled = mdf["Sampled"].to_numpy(dtype=int)
                seg_set: set[int] = set()
                for i in range(N):
                    if int(sampled[i]) == 1:
                        seg_set.add(i // config.segment_size)
                if seg_set:
                    segment_indices = sorted(seg_set)
                    reused_mask = True
                    print(
                        f"[resume] {csv_path.name}: reuse segment set from {mask_path.name} "
                        f"({len(segment_indices)} segments)"
                    )
        except Exception as e:
            print(f"[resume] {csv_path.name}: failed to read mask ({e}), will re-sample")

    if not reused_mask:
        rng = random.Random(config.random_seed)

        anomaly_seg_indices: list = []
        normal_seg_indices: list = []
        for seg_idx in range(num_segments):
            seg_start = seg_idx * config.segment_size
            seg_end = min(seg_start + config.segment_size, N)
            if np.any((anomalies >= seg_start) & (anomalies < seg_end)):
                anomaly_seg_indices.append(seg_idx)
            else:
                normal_seg_indices.append(seg_idx)

        A = len(anomaly_seg_indices)
        B = len(normal_seg_indices)
        r = config.anomaly_ratio

        T_req = max(1, int(num_segments * config.sample_ratio))
        T_req = min(T_req, num_segments)

        def _feasible_segment_count(T: int) -> bool:
            if T < 1 or T > A + B:
                return False
            if r <= 0:
                return True
            lo = int(np.ceil(T * r))
            hi = min(A, T)
            need_a = max(lo, T - B)
            return need_a <= hi

        segment_indices = []

        if r <= 0:
            pool = list(range(num_segments))
            segment_indices = rng.sample(pool, min(T_req, len(pool)))
        elif A == 0:
            n_keep = min(T_req, len(normal_seg_indices))
            cap = int(getattr(config, "max_if_normal", -1))
            if cap >= 0:
                n_keep = min(n_keep, cap)
            if n_keep <= 0:
                print(
                    f"[{csv_path.name}] no anomaly windows; cannot reach anomaly_ratio={r}, "
                    f"max_if_normal={cap} -> skip series."
                )
                return
            segment_indices = rng.sample(normal_seg_indices, n_keep)
            print(
                f"[{csv_path.name}] no anomaly windows; cannot reach anomaly_ratio={r}, "
                f"outputting {len(segment_indices)} normal segments only (target T_req={T_req}, max_if_normal={cap})."
            )
        else:
            T = T_req
            while T >= 1 and not _feasible_segment_count(T):
                T -= 1
            if T < 1:
                segment_indices = []
            else:
                if T < T_req:
                    print(
                        f"[{csv_path.name}] to satisfy anomaly_ratio>={r} and anomaly/normal pool limits, "
                        f"segment count {T_req} -> {T}."
                    )
                lo_a = int(np.ceil(T * r)) if r > 0 else 0
                a_low = max(lo_a, T - B)
                a_high = min(A, T)
                a_pick = a_low
                if a_high > a_low:
                    a_pick = rng.randint(a_low, a_high)
                b_pick = T - a_pick
                chosen_anom = rng.sample(anomaly_seg_indices, a_pick)
                chosen_norm = (
                    rng.sample(normal_seg_indices, b_pick) if b_pick > 0 else []
                )
                segment_indices = chosen_anom + chosen_norm

        segment_indices = sorted(set(segment_indices))

        if not segment_indices:
            return

        full_mask = np.zeros(N, dtype=int)
        for seg_idx in segment_indices:
            start = seg_idx * config.segment_size
            end = min(start + config.segment_size, N)
            full_mask[start:end] = 1

        mask_df = pd.DataFrame({"Index": np.arange(N), "Sampled": full_mask})
        mask_df.to_csv(mask_path, index=False)
    elif not segment_indices:
        return

    # Process each sampled segment
    for seg_idx in tqdm(segment_indices, desc=f"Segments {name}", ncols=100):
        start = seg_idx * config.segment_size
        end = min(start + config.segment_size, N)

        sample_folder = output_dir / f"{name}_seg{seg_idx:03d}"
        meta_path = sample_folder / "meta_info.json"
        if meta_path.is_file():
            p_plain = sample_folder / "segment_preview_plain.png"
            p_lab = sample_folder / "segment_preview_labeled.png"
            if not (p_plain.is_file() and p_lab.is_file()):
                _write_segment_preview_pngs(
                    sample_folder,
                    values=data,
                    labels=labels,
                    start=start,
                    end=end,
                    series_name=name,
                    seg_idx=seg_idx,
                )
            if anomaly_reference_png is not None:
                _maybe_write_segment_final_png(
                    sample_folder,
                    anomaly_reference_png.resolve(),
                )
            continue

        sample_folder.mkdir(parents=True, exist_ok=True)

        # end is exclusive like pandas iloc[start:end]; ground_truth derived from sliced Label at runtime
        meta = {
            "type": str(dataset_type).strip().upper(),
            "dataset_root": "..",
            "series_stem": name,
            "index": {"start": int(start), "end": int(end)},
            "global_data_file": f"_global/data/{name}__full.csv",
            "global_statistics_file": f"_global/statistics/{name}_global_statistics.json",
        }
        with open(sample_folder / "meta_info.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        _write_segment_preview_pngs(
            sample_folder,
            values=data,
            labels=labels,
            start=start,
            end=end,
            series_name=name,
            seg_idx=seg_idx,
        )
        if anomaly_reference_png is not None:
            _maybe_write_segment_final_png(
                sample_folder,
                anomaly_reference_png.resolve(),
            )

    print(
        f"Done -> {csv_path.name} "
        f"(total points {N}, sampled segments {len(segment_indices)}/{num_segments})"
    )


def main():
    args = parse_args()
    config = config_from_args(args)
    validate_config(config)
    input_path = args.input_path.resolve()
    output_dir = args.output_dir.resolve()

    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise SystemExit(f"When input is a file it must be .csv: {input_path}")
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.csv"))
    else:
        raise SystemExit(f"Input is neither a CSV file nor a directory: {input_path}")
    print(f"Found {len(files)} CSV file(s), processing...")
    print(
        "[config]",
        f"segment_size={config.segment_size}",
        f"sample_ratio={config.sample_ratio}",
        f"anomaly_ratio={config.anomaly_ratio}",
        f"max_if_normal={config.max_if_normal}",
        f"detrend_enabled={config.detrend_enabled}",
        f"resume={bool(args.resume)}",
        sep=" ",
    )
    ref_arg: Optional[Path] = args.anomaly_reference_png
    if ref_arg is not None:
        ref_arg = ref_arg.resolve()
        if not ref_arg.is_file():
            print(f"[warn] --anomaly-reference-png is not a valid file, ignored: {ref_arg}", flush=True)
            ref_arg = None
    for f in files:
        process(
            f,
            output_dir,
            args.dataset_type,
            config=config,
            resume=args.resume,
            anomaly_reference_png=ref_arg,
        )
    print("All done!")


if __name__ == "__main__":
    main()
