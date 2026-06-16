from __future__ import annotations

"""
Build one dataset root from raw CSV splits into cluster and processed outputs.

Example::

  python3 -m AnomaMind.dataprep.build_dataset \
      data/KPI \
      --include-train-off \
      --train-off-output-name train_off

Expected dataset root layout includes paths such as::

  data/KPI/KPI.yaml
  data/KPI/raw/in-series
  data/KPI/raw/cross-series

The pipeline writes ``raw/train``, ``raw/test``, ``cluster/``, and ``processed/``.
"""

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from scipy import signal

from AnomaMind.dataprep import gen_cluster
from AnomaMind.dataprep import preprocess


@dataclass(frozen=True)
class PipelineConfig:
    segment_size: int
    train_anomaly_ratio: float
    test_anomaly_ratio: float
    dataset_type: str


@dataclass(frozen=True)
class RawSplit:
    train_series: list[Path]
    test_series: list[Path]


def compute_offset_windows(*, length: int, segment_size: int, offset: int) -> list[tuple[int, int]]:
    if int(segment_size) <= 0:
        raise ValueError("segment_size must be positive")
    if int(offset) <= 0:
        raise ValueError("offset must be positive")
    if int(offset) >= int(segment_size):
        raise ValueError("offset must be smaller than segment_size")
    if int(length) < int(segment_size) + int(offset):
        return []
    return [
        (int(start), int(start + segment_size))
        for start in range(int(offset), int(length) - int(segment_size) + 1, int(segment_size))
    ]


def format_offset_segment_dir_name(series_name: str, *, offset: int, seg_idx: int) -> str:
    return f"{series_name}_off{int(offset):04d}_seg{int(seg_idx):03d}"


def _reset_csv_dir(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    for path in dir_path.glob("*.csv"):
        path.unlink()


def _reset_dir(dir_path: Path) -> None:
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def _resolve_yaml_path(dataset_root: Path) -> Path:
    exact = dataset_root / f"{dataset_root.name}.yaml"
    if exact.is_file():
        return exact
    yaml_files = sorted(dataset_root.glob("*.yaml"))
    if len(yaml_files) == 1:
        return yaml_files[0]
    raise FileNotFoundError(f"cannot resolve dataset yaml under: {dataset_root}")


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            out[key] = None
            continue
        try:
            if any(ch in value for ch in [".", "e", "E"]):
                out[key] = float(value)
            else:
                out[key] = int(value)
            continue
        except ValueError:
            pass
        if value.lower() in {"true", "false"}:
            out[key] = value.lower() == "true"
        else:
            out[key] = value.strip("'\"")
    return out


def _load_yaml_dict(yaml_path: Path) -> dict[str, Any]:
    text = yaml_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return _parse_simple_yaml(text)


def load_pipeline_config(dataset_root: Path) -> PipelineConfig:
    yaml_path = _resolve_yaml_path(dataset_root)
    raw = _load_yaml_dict(yaml_path)
    segment_size = int(raw.get("segment_size", 1000))
    train_anomaly_ratio = float(raw.get("processed_train_anomaly_ratio", 0.7))
    test_anomaly_ratio = float(raw.get("processed_test_anomaly_ratio", 0.4))
    return PipelineConfig(
        segment_size=segment_size,
        train_anomaly_ratio=train_anomaly_ratio,
        test_anomaly_ratio=test_anomaly_ratio,
        dataset_type=yaml_path.stem.upper(),
    )


def build_raw_split_from_inputs(dataset_root: Path) -> RawSplit:
    raw_dir = dataset_root / "raw"
    in_series = sorted((raw_dir / "in-series").glob("*.csv"))
    cross_series = sorted((raw_dir / "cross-series").glob("*.csv"))
    train_series = sorted(in_series + cross_series, key=lambda p: p.name)
    test_series = sorted(in_series, key=lambda p: p.name)
    return RawSplit(train_series=train_series, test_series=test_series)


def materialize_raw_split(
    *,
    split: RawSplit,
    train_dir: Path,
    test_dir: Path,
) -> None:
    _reset_csv_dir(train_dir)
    _reset_csv_dir(test_dir)
    test_names = {p.name for p in split.test_series}
    for src in split.train_series:
        if src.name in test_names:
            df = pd.read_csv(src)
            midpoint = len(df) // 2
            df.iloc[:midpoint].to_csv(train_dir / src.name, index=False)
            df.iloc[midpoint:].to_csv(test_dir / src.name, index=False)
        else:
            _link_or_copy(src, train_dir / src.name)


def run_gen_cluster(*, input_dir: Path, output_dir: Path, ref_segment_size: int) -> None:
    _reset_dir(output_dir)
    rc = gen_cluster.main(
        [
            str(input_dir),
            "-o",
            str(output_dir),
            "--ref-segment-size",
            str(ref_segment_size),
        ]
    )
    if rc != 0:
        raise RuntimeError(f"gen_cluster failed with exit code {rc}")


def run_preprocess(
    *,
    input_path: Path,
    output_dir: Path,
    dataset_type: str,
    segment_size: int,
    anomaly_ratio: float,
    anomaly_reference_png: Path | None,
) -> None:
    _reset_dir(output_dir)
    config = preprocess.PreprocessConfig(
        segment_size=int(segment_size),
        anomaly_ratio=float(anomaly_ratio),
    )
    preprocess.validate_config(config)
    files = (
        [input_path]
        if input_path.is_file()
        else sorted(p for p in input_path.glob("*.csv") if p.is_file())
    )
    ref = anomaly_reference_png if anomaly_reference_png and anomaly_reference_png.is_file() else None
    for csv_file in files:
        preprocess.process(
            csv_file,
            output_dir,
            dataset_type,
            config=config,
            resume=False,
            anomaly_reference_png=ref,
        )


def _resolve_cluster_reference(cluster_dir: Path) -> Path | None:
    inner = cluster_dir / "anomaly_reference_inner.png"
    if inner.is_file():
        return inner
    base = cluster_dir / "anomaly_reference.png"
    if base.is_file():
        return base
    return None


def _prepare_full_series(
    *,
    csv_path: Path,
    output_dir: Path,
    source_processed_train_dir: Path | None,
    config: preprocess.PreprocessConfig,
) -> tuple[pd.Series, pd.Series]:
    name = csv_path.stem
    g_data_dir = output_dir / "_global" / "data"
    g_stat_dir = output_dir / "_global" / "statistics"
    g_data_dir.mkdir(parents=True, exist_ok=True)
    g_stat_dir.mkdir(parents=True, exist_ok=True)
    full_csv = g_data_dir / f"{name}__full.csv"
    global_stats_path = g_stat_dir / f"{name}_global_statistics.json"

    if source_processed_train_dir is not None:
        src_full = source_processed_train_dir / "_global" / "data" / f"{name}__full.csv"
        src_stats = source_processed_train_dir / "_global" / "statistics" / f"{name}_global_statistics.json"
        if src_full.is_file() and not full_csv.exists():
            _link_or_copy(src_full, full_csv)
        if src_stats.is_file() and not global_stats_path.exists():
            _link_or_copy(src_stats, global_stats_path)

    if full_csv.is_file():
        df_full = pd.read_csv(full_csv)
        return df_full["Value"].astype(float), df_full["Label"].astype(int)

    df = pd.read_csv(csv_path)
    if {"Value", "Label"}.issubset(df.columns):
        values = pd.to_numeric(df["Value"], errors="coerce").replace([float("inf"), float("-inf")], pd.NA)
        values = values.interpolate(method="linear", limit_direction="both").bfill().ffill()
        if values.isna().any():
            fill = 0.0 if values.isna().all() else float(values.dropna().median())
            values = values.fillna(fill)
        values = preprocess.np.round(values.to_numpy(dtype=float), config.round_decimals)
        labels = df["Label"].astype(int).to_numpy()
        norm_info: dict[str, object] = {"norm_mode": "pre_normalized_value_column"}
    elif {"Data", "Label"}.issubset(df.columns):
        data_series = pd.to_numeric(df["Data"], errors="coerce").replace([float("inf"), float("-inf")], pd.NA)
        data_series = data_series.interpolate(method="linear", limit_direction="both")
        data_series = data_series.bfill().ffill()
        if data_series.isna().any():
            fill = 0.0 if data_series.isna().all() else float(data_series.dropna().median())
            data_series = data_series.fillna(fill)

        x_raw = data_series.to_numpy(dtype=float)
        p_low = float(preprocess.np.percentile(x_raw, config.normalization_lower_percentile))
        p_high = float(preprocess.np.percentile(x_raw, config.normalization_upper_percentile))
        denom = p_high - p_low
        if denom <= config.normalization_eps:
            xmin, xmax = float(preprocess.np.min(x_raw)), float(preprocess.np.max(x_raw))
            x_scaled = (x_raw - xmin) / (xmax - xmin + config.normalization_eps)
            norm_info = {
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

        x = signal.detrend(x_scaled, type=preprocess.DETREND_TYPE) if config.detrend_enabled else x_scaled
        values = preprocess.np.round(x, config.round_decimals)
        labels = df["Label"].astype(int).to_numpy()
    else:
        raise ValueError(f"missing required columns in {csv_path}")

    pd.DataFrame(
        {
            "Index": preprocess.np.arange(len(values), dtype=int),
            "Value": values.astype(float),
            "Label": labels.astype(int),
        }
    ).to_csv(full_csv, index=False)

    global_stats = preprocess._global_stats_from_series(values, norm_info)
    with open(global_stats_path, "w", encoding="utf-8") as f:
        json.dump(global_stats, f, ensure_ascii=False, indent=2)

    df_full = pd.read_csv(full_csv)
    return df_full["Value"].astype(float), df_full["Label"].astype(int)


def _sample_window_indices(
    *,
    labels: preprocess.np.ndarray,
    windows: Sequence[tuple[int, int]],
    config: preprocess.PreprocessConfig,
) -> list[int]:
    anomalies = preprocess.np.flatnonzero(labels == 1)
    rng = random.Random(config.random_seed)

    anomaly_window_indices: list[int] = []
    normal_window_indices: list[int] = []
    for idx, (start, end) in enumerate(windows):
        if preprocess.np.any((anomalies >= start) & (anomalies < end)):
            anomaly_window_indices.append(idx)
        else:
            normal_window_indices.append(idx)

    total_windows = len(windows)
    if total_windows == 0:
        return []

    anomaly_count = len(anomaly_window_indices)
    normal_count = len(normal_window_indices)
    ratio = config.anomaly_ratio
    target = max(1, int(total_windows * config.sample_ratio))
    target = min(target, total_windows)

    def _feasible_window_count(count: int) -> bool:
        if count < 1 or count > anomaly_count + normal_count:
            return False
        if ratio <= 0:
            return True
        lo = int(preprocess.np.ceil(count * ratio))
        hi = min(anomaly_count, count)
        need_anomaly = max(lo, count - normal_count)
        return need_anomaly <= hi

    if ratio <= 0:
        return sorted(rng.sample(list(range(total_windows)), min(target, total_windows)))

    if anomaly_count == 0:
        keep = min(target, len(normal_window_indices))
        cap = int(getattr(config, "max_if_normal", -1))
        if cap >= 0:
            keep = min(keep, cap)
        if keep <= 0:
            return []
        return sorted(rng.sample(normal_window_indices, keep))

    chosen_target = target
    while chosen_target >= 1 and not _feasible_window_count(chosen_target):
        chosen_target -= 1
    if chosen_target < 1:
        return []

    lo_anomaly = int(preprocess.np.ceil(chosen_target * ratio)) if ratio > 0 else 0
    anomaly_low = max(lo_anomaly, chosen_target - normal_count)
    anomaly_high = min(anomaly_count, chosen_target)
    anomaly_pick = anomaly_low
    if anomaly_high > anomaly_low:
        anomaly_pick = rng.randint(anomaly_low, anomaly_high)
    normal_pick = chosen_target - anomaly_pick
    chosen_anomaly = rng.sample(anomaly_window_indices, anomaly_pick)
    chosen_normal = rng.sample(normal_window_indices, normal_pick) if normal_pick > 0 else []
    return sorted(chosen_anomaly + chosen_normal)


def process_offset_series(
    *,
    csv_path: Path,
    output_dir: Path,
    dataset_type: str,
    config: preprocess.PreprocessConfig,
    offset: int,
    source_processed_train_dir: Path | None,
    anomaly_reference_png: Path | None,
) -> int:
    values, labels = _prepare_full_series(
        csv_path=csv_path,
        output_dir=output_dir,
        source_processed_train_dir=source_processed_train_dir,
        config=config,
    )
    data = values.to_numpy(dtype=float)
    label_array = labels.to_numpy(dtype=int)
    windows = compute_offset_windows(length=len(data), segment_size=config.segment_size, offset=offset)
    chosen_indices = _sample_window_indices(labels=label_array, windows=windows, config=config)
    if not chosen_indices:
        return 0

    sequence_id = preprocess.extract_sequence_id(csv_path.stem)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"{sequence_id}_sample.csv"
    full_mask = preprocess.np.zeros(len(data), dtype=int)
    for idx in chosen_indices:
        start, end = windows[idx]
        full_mask[start:end] = 1
    pd.DataFrame({"Index": preprocess.np.arange(len(data)), "Sampled": full_mask}).to_csv(mask_path, index=False)

    built = 0
    for idx in chosen_indices:
        start, end = windows[idx]
        sample_folder = output_dir / format_offset_segment_dir_name(csv_path.stem, offset=offset, seg_idx=idx)
        sample_folder.mkdir(parents=True, exist_ok=True)
        meta = {
            "type": str(dataset_type).strip().upper(),
            "dataset_root": "..",
            "series_stem": csv_path.stem,
            "window_offset": int(offset),
            "window_index": int(idx),
            "index": {"start": int(start), "end": int(end)},
            "global_data_file": f"_global/data/{csv_path.stem}__full.csv",
            "global_statistics_file": f"_global/statistics/{csv_path.stem}_global_statistics.json",
        }
        with open(sample_folder / "meta_info.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        preprocess._write_segment_preview_pngs(
            sample_folder,
            values=data,
            labels=label_array,
            start=start,
            end=end,
            series_name=csv_path.stem,
            seg_idx=idx,
        )
        if anomaly_reference_png is not None and anomaly_reference_png.is_file():
            preprocess._maybe_write_segment_final_png(
                sample_folder,
                anomaly_reference_png.resolve(),
            )
        built += 1
    return built


def run_train_off_pipeline(
    dataset_root: Path,
    *,
    output_name: str = "train_off",
    offset: int | None = None,
    sample_ratio: float = 1.0,
) -> Path:
    dataset_root = dataset_root.resolve()
    pipeline = load_pipeline_config(dataset_root)
    train_dir = dataset_root / "raw" / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"missing train dir: {train_dir}")

    output_dir = dataset_root / "processed" / output_name
    _reset_dir(output_dir)
    config = preprocess.PreprocessConfig(
        segment_size=int(pipeline.segment_size),
        sample_ratio=float(sample_ratio),
        anomaly_ratio=float(pipeline.train_anomaly_ratio),
    )
    preprocess.validate_config(config)

    chosen_offset = int(offset) if offset is not None else int(config.segment_size // 2)
    if chosen_offset <= 0:
        raise ValueError("computed offset must be positive")

    source_processed_train_dir = dataset_root / "processed" / "train"
    if not source_processed_train_dir.is_dir():
        source_processed_train_dir = None

    cluster_dir = dataset_root / "cluster"
    anomaly_reference_png = _resolve_cluster_reference(cluster_dir)

    total_built = 0
    for csv_path in sorted(p for p in train_dir.glob("*.csv") if p.is_file()):
        total_built += process_offset_series(
            csv_path=csv_path,
            output_dir=output_dir,
            dataset_type=pipeline.dataset_type,
            config=config,
            offset=chosen_offset,
            source_processed_train_dir=source_processed_train_dir,
            anomaly_reference_png=anomaly_reference_png,
        )
    print(
        f"[build_data.train_off] dataset_root={dataset_root} output_dir={output_dir} "
        f"offset={chosen_offset} built_segments={total_built}"
    )
    return output_dir


def run_pipeline(
    dataset_root: Path,
    *,
    include_train_off: bool = True,
    train_off_output_name: str = "train_off",
    train_off_offset: int | None = None,
    train_off_sample_ratio: float = 1.0,
) -> None:
    dataset_root = dataset_root.resolve()
    config = load_pipeline_config(dataset_root)

    raw_dir = dataset_root / "raw"
    split = build_raw_split_from_inputs(dataset_root)
    materialize_raw_split(
        split=split,
        train_dir=raw_dir / "train",
        test_dir=raw_dir / "test",
    )

    cluster_dir = dataset_root / "cluster"
    run_gen_cluster(
        input_dir=raw_dir / "train",
        output_dir=cluster_dir,
        ref_segment_size=config.segment_size,
    )
    ref_png = _resolve_cluster_reference(cluster_dir)

    processed_dir = dataset_root / "processed"
    run_preprocess(
        input_path=raw_dir / "train",
        output_dir=processed_dir / "train",
        dataset_type=config.dataset_type,
        segment_size=config.segment_size,
        anomaly_ratio=config.train_anomaly_ratio,
        anomaly_reference_png=ref_png,
    )
    run_preprocess(
        input_path=raw_dir / "test",
        output_dir=processed_dir / "test",
        dataset_type=config.dataset_type,
        segment_size=config.segment_size,
        anomaly_ratio=config.test_anomaly_ratio,
        anomaly_reference_png=ref_png,
    )
    if include_train_off:
        run_train_off_pipeline(
            dataset_root,
            output_name=str(train_off_output_name),
            offset=train_off_offset,
            sample_ratio=float(train_off_sample_ratio),
        )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Build raw/train,test -> cluster -> processed from one dataset root. "
            "Expected layout: <root>/<NAME>.yaml, <root>/raw/in-series, <root>/raw/cross-series."
        ),
    )
    ap.add_argument(
        "dataset_root",
        type=Path,
        help="Dataset root, e.g. data/WSD",
    )
    ap.add_argument(
        "--include-train-off",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also build processed/train_off after processed/train and processed/test. Default: enabled.",
    )
    ap.add_argument(
        "--train-off-output-name",
        default="train_off",
        help="Processed output subdir name for offset windows.",
    )
    ap.add_argument(
        "--train-off-offset",
        type=int,
        default=None,
        help="Window offset for train_off. Default: segment_size // 2.",
    )
    ap.add_argument(
        "--train-off-sample-ratio",
        type=float,
        default=1.0,
        help="Sample ratio applied to offset windows.",
    )
    return ap


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.dataset_root.resolve()
    run_pipeline(
        root,
        include_train_off=bool(args.include_train_off),
        train_off_output_name=str(args.train_off_output_name),
        train_off_offset=args.train_off_offset,
        train_off_sample_ratio=float(args.train_off_sample_ratio),
    )
    print(f"dataset_root: {root}")
    print(f"cluster: {root / 'cluster'}")
    print(f"processed_train: {root / 'processed' / 'train'}")
    print(f"processed_test: {root / 'processed' / 'test'}")
    if bool(args.include_train_off):
        print(f"processed_train_off: {root / 'processed' / str(args.train_off_output_name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
