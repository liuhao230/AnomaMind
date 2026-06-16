#!/usr/bin/env python3
"""Batch inference for processed anomaly segments using the AnomaAgent workflow.

Typical usage in the ``detector`` conda env::

  python3 -m AnomaMind.infer.infer \
      /path/data/KPI/processed/test_all \
      --output-dir /path/data/KPI/result/test/API \
      --workers 2 \
      --max-retries 3

Input can be either:
- a processed dataset directory (optionally recursive) whose segment folders contain ``meta_info.json``; or
- a single processed segment directory containing ``meta_info.json``.

Another common evaluation-style run::

  python3 -m AnomaMind.infer.infer \
      /path/data/WSD/processed/test_all \
      --output-dir /path/data/WSD/result/test_all/anomamind_infer \
      --workers 8 \
      --recursive

Each segment produces ``<segment>.json`` and, unless disabled, the same PNG artifacts and
``_smoke_summary.json`` contract used by the previous smoke runner.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _env_dir() -> Path:
    return _repo_root() / "env"


def _load_dotenv_file(path: Path, *, override: bool) -> None:
    """
    Minimal .env loader.

    - Supports KEY=VALUE lines
    - Ignores blank lines and comments starting with '#'
    - Does NOT require python-dotenv
    - By default, does not override already-set environment variables
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if not override and os.getenv(key) is not None:
            continue
        os.environ[key] = value


def _autoload_env_files(*, override: bool = False) -> None:
    for name in ("actor.env", "detector.env", "locator.env", "evaluator.env"):
        p = _env_dir() / name
        if p.is_file():
            _load_dotenv_file(p, override=override)


@dataclass(frozen=True)
class SegmentTask:
    index: int
    seg_dir: Path
    stem: str
    skip_existing_ok: bool


@dataclass(frozen=True)
class InferConfig:
    output_dir: Path
    max_turns: int
    enable_reflection: bool
    no_print_messages: bool
    no_debug_artifacts: bool
    workers: int
    max_retries: int
    retry_delay: float
    show_progress: bool
    force: bool


def _gv(d: dict, key: str, default=None):
    v = d.get(key)
    return v if v not in (None, "") else default


def _build_llm_config(
    env_map: dict,
    *,
    model_keys: list[str],
    api_key_keys: list[str] | None = None,
    base_url_keys: list[str] | None = None,
    temperature_keys: list[str] | None = None,
    max_tokens_keys: list[str] | None = None,
    default_model: str = "gpt-4o-mini",
    default_base_url: str = "http://localhost:8000/v1",
    default_temperature: float = 0.2,
    default_max_tokens: int | None = None,
):
    from AnomaMind.core.llm_config import LLMConfig

    def _first(keys: list[str] | None, default=None):
        if not keys:
            return default
        for key in keys:
            value = _gv(env_map, key)
            if value not in (None, ""):
                return value
        return default

    max_tokens_value = _first(max_tokens_keys)
    if max_tokens_value is not None and str(max_tokens_value).strip().isdigit():
        max_tokens = int(max_tokens_value)
    else:
        max_tokens = default_max_tokens

    return LLMConfig(
        model_name=str(_first(model_keys, default_model)),
        api_key=_first(api_key_keys or ["LLM_API_KEY"]),
        base_url=str(_first(base_url_keys or ["LLM_BASE_URL"], default_base_url)),
        temperature=float(_first(temperature_keys, default_temperature) or default_temperature),
        max_tokens=max_tokens,
    )


def discover_segment_dirs(
    input_path: Path,
    *,
    recursive: bool = False,
    max_depth: int = 6,
) -> list[Path]:
    """Discover processed segment directories under input_path.

    A segment directory is identified by containing a ``meta_info.json`` file.

    If input_path itself is a segment dir, returns [input_path].
    Otherwise:
    - non-recursive mode: search only direct child dirs (old behavior)
    - recursive mode: search descendants up to max_depth
    """
    root = input_path.resolve()
    if not root.is_dir():
        return []
    if (root / "meta_info.json").is_file():
        return [root]
    found: list[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "meta_info.json").is_file():
            found.append(p)
    if found or not recursive:
        return found

    base_depth = len(root.parts)
    dedup: dict[str, Path] = {}
    for meta in root.rglob("meta_info.json"):
        seg_dir = meta.parent
        depth = len(seg_dir.parts) - base_depth
        if int(max_depth) >= 0 and depth > int(max_depth):
            continue
        dedup[str(seg_dir.resolve())] = seg_dir.resolve()
    return [dedup[k] for k in sorted(dedup.keys())]


def _exit_code_from_smoke_json(data: dict) -> int:
    """0=success, 2=failure; keep parity with the historical smoke exit semantics."""
    fo = data.get("final_output")
    if data.get("state_has_error"):
        return 2
    if not isinstance(fo, dict):
        return 2
    if fo.get("error"):
        return 2
    return 0


def _smoke_segment_done_ok(output_dir: Path, stem: str) -> bool:
    json_path = output_dir / f"{stem}.json"
    if not json_path.is_file():
        return False
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return _exit_code_from_smoke_json(data) == 0


def _summary_row_from_resume_json(output_dir: Path, stem: str, seg_dir: Path) -> dict[str, Any]:
    json_path = output_dir / f"{stem}.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    se = data.get("smoke_point_eval") or {}
    rc = _exit_code_from_smoke_json(data)
    return {
        "segment": stem,
        "segment_dir": str(seg_dir.resolve()),
        "exit_code": rc,
        "tp_points": int(se.get("tp_points", 0)),
        "fp_points": int(se.get("fp_points", 0)),
        "fn_points": int(se.get("fn_points", 0)),
        "gt_point_count": int(se.get("gt_point_count", 0)),
        "pred_point_count": int(se.get("pred_point_count", 0)),
        "attempts": 0,
        "skipped_resume_ok": True,
    }


def build_segment_tasks(
    segments: Iterable[Path],
    *,
    output_dir: Path,
    force: bool,
) -> list[SegmentTask]:
    tasks: list[SegmentTask] = []
    for index, seg_dir in enumerate(sorted(Path(p).resolve() for p in segments)):
        stem = seg_dir.name
        tasks.append(
            SegmentTask(
                index=index,
                seg_dir=seg_dir,
                stem=stem,
                skip_existing_ok=(not force and _smoke_segment_done_ok(output_dir, stem)),
            )
        )
    return tasks


def _iter_progress(items: Iterable[Any], *, total: int, enabled: bool, desc: str):
    if not enabled:
        for item in items:
            yield item
        return
    try:
        from tqdm import tqdm

        yield from tqdm(items, total=total, desc=desc, dynamic_ncols=True)
    except Exception:
        for item in items:
            yield item


def _write_smoke_gt_pred_png(
    plot_path: Path,
    seg_name: str,
    pack: dict[str, Any],
    gt_intervals: List[List[int]],
    pred_intervals: List[list[Any]],
    smoke_eval: dict[str, int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from AnomaMind.core.utils import get_column_names

    df = pack["segment_data"]
    tcol, vcol = get_column_names(df)
    xs = df[tcol].astype(float).to_numpy()
    ys = df[vcol].astype(float).to_numpy()
    dpi = 165
    fig, ax = plt.subplots(figsize=(12, 3.2), dpi=dpi)
    ax.plot(xs, ys, color="0.2", linewidth=0.85, label="series")
    for a, b in gt_intervals:
        try:
            aa, bb = int(a), int(b)
        except (TypeError, ValueError):
            continue
        if aa > bb:
            aa, bb = bb, aa
        ax.axvspan(float(aa), float(bb), color="tab:green", alpha=0.28)
    for it in pred_intervals:
        if not isinstance(it, (list, tuple)) or len(it) < 1:
            continue
        try:
            if len(it) == 1:
                aa = bb = int(it[0])
            else:
                aa, bb = int(it[0]), int(it[1])
        except (TypeError, ValueError):
            continue
        if aa > bb:
            aa, bb = bb, aa
        ax.axvspan(float(aa), float(bb), color="tab:red", alpha=0.22)
    ax.legend(
        handles=[
            Patch(facecolor="tab:green", alpha=0.35, edgecolor="none", label="GT (Label=1)"),
            Patch(facecolor="tab:red", alpha=0.28, edgecolor="none", label="Pred (final_output)"),
        ],
        loc="upper right",
        fontsize=8,
    )
    ax.set_title(
        f"{seg_name}  TP={smoke_eval['tp_points']} FP={smoke_eval['fp_points']} FN={smoke_eval['fn_points']}"
    )
    ax.set_xlabel("global index")
    ax.set_ylabel(str(vcol))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def _copy_segment_final_to_output(seg_dir: Path, output_dir: Path, stem: str) -> None:
    src = seg_dir / "segment_final.png"
    if not src.is_file():
        return
    dst = output_dir / f"{stem}_segment_final.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _load_llm_configs(*, load_evaluator: bool = True):
    try:
        from dotenv import dotenv_values
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Required: pip install python-dotenv") from exc

    env_dir = _env_dir()
    det = dotenv_values(env_dir / "detector.env") or {}
    loc = dotenv_values(env_dir / "locator.env") or {}

    locator_cfg = _build_llm_config(
        loc,
        model_keys=["LOCATOR_VISION_MODEL", "AUX_LLM_MODEL"],
        api_key_keys=["LOCATOR_API_KEY", "LLM_API_KEY"],
        base_url_keys=["LOCATOR_BASE_URL", "LLM_BASE_URL"],
        temperature_keys=["LOCATOR_TEMPERATURE"],
        max_tokens_keys=["LOCATOR_MAX_TOKENS"],
        default_model="gpt-4o-mini",
        default_base_url="http://localhost:8000/v1",
        default_temperature=0.2,
        default_max_tokens=None,
    )
    actor_cfg = _build_llm_config(
        det,
        model_keys=["ACTOR_MODEL", "AUX_LLM_MODEL"],
        api_key_keys=["ACTOR_API_KEY", "LLM_API_KEY"],
        base_url_keys=["ACTOR_BASE_URL", "LLM_BASE_URL"],
        temperature_keys=["ACTOR_TEMPERATURE"],
        max_tokens_keys=["ACTOR_MAX_TOKENS"],
        default_model="gpt-4o-mini",
        default_base_url="http://localhost:8000/v1",
        default_temperature=0.2,
        default_max_tokens=4000,
    )
    detector_cfg = _build_llm_config(
        det,
        model_keys=["DETECTOR_MODEL", "ACTOR_MODEL", "AUX_LLM_MODEL"],
        api_key_keys=["DETECTOR_API_KEY", "ACTOR_API_KEY", "LLM_API_KEY"],
        base_url_keys=["DETECTOR_BASE_URL", "ACTOR_BASE_URL", "LLM_BASE_URL"],
        temperature_keys=["DETECTOR_TEMPERATURE", "ACTOR_TEMPERATURE"],
        max_tokens_keys=["DETECTOR_MAX_TOKENS", "ACTOR_MAX_TOKENS"],
        default_model="gpt-4o-mini",
        default_base_url="http://localhost:8000/v1",
        default_temperature=0.2,
        default_max_tokens=4000,
    )
    evaluator_cfg = None
    if load_evaluator:
        ev = dotenv_values(env_dir / "evaluator.env") or {}
        evaluator_cfg = _build_llm_config(
            ev,
            model_keys=["EVALUATOR_MODEL", "AUX_LLM_MODEL"],
            api_key_keys=["EVALUATOR_API_KEY", "LLM_API_KEY"],
            base_url_keys=["EVALUATOR_BASE_URL", "LLM_BASE_URL"],
            temperature_keys=["EVALUATOR_TEMPERATURE"],
            max_tokens_keys=["EVALUATOR_MAX_TOKENS"],
            default_model="gpt-4o-mini",
            default_base_url="http://localhost:8000/v1",
            default_temperature=0.2,
            default_max_tokens=4000,
        )
    return locator_cfg, actor_cfg, detector_cfg, evaluator_cfg


def _make_workflow_factory(*, max_turns: int, enable_reflection: bool) -> Callable[[], Any]:
    from AnomaMind.workflows.AnomaAgent import AnomaAgent

    locator_cfg, actor_cfg, detector_cfg, evaluator_cfg = _load_llm_configs(
        load_evaluator=enable_reflection
    )

    def _factory():
        agent = AnomaAgent(
            max_turns=max_turns,
            locator_config=locator_cfg,
            actor_config=actor_cfg,
            detector_config=detector_cfg,
            evaluator_config=evaluator_cfg,
            enable_reflection=enable_reflection,
        )
        return agent.build_workflow()

    return _factory


def _write_error_payload(task: SegmentTask, output_dir: Path, error_message: str) -> None:
    payload = {
        "segment_dir": str(task.seg_dir.resolve()),
        "state_has_error": True,
        "final_output": {
            "error": True,
            "error_message": str(error_message),
            "anomaly_intervals": [],
        },
        "visual_description": "",
        "visual_localization": [],
        "visual_localization_items": [],
        "actor_messages": [],
        "detector_messages": [],
        "iteration_count": 0,
        "redo": False,
        "suggestion": "",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{task.stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_segment_once(
    task: SegmentTask,
    *,
    workflow_factory: Callable[[], Any],
    config: InferConfig,
) -> tuple[int, dict[str, Any]]:
    from AnomaMind.core.reward import intervals_to_set
    from AnomaMind.core.utils import load_segment_data
    from AnomaMind.helpers.message_io import export_messages

    pack = load_segment_data(task.seg_dir)
    state = {
        "data": pack["segment_data"],
        "type": pack["type"],
        "path": pack["path"],
        "actor_messages": [],
        "detector_messages": [],
        "iteration_count": 0,
        "redo": False,
        "suggestion": "",
        "has_error": False,
        "error_message": "",
        "final_output": {},
    }

    workflow = workflow_factory()
    out = workflow.invoke(state, {"recursion_limit": 200})
    fo = out.get("final_output") or {}
    actor_messages_export = export_messages(out.get("actor_messages") or [])
    detector_messages_export = export_messages(out.get("detector_messages") or [])
    messages_export = actor_messages_export + detector_messages_export

    gt_intervals: List[List[int]] = list(pack.get("ground_truth") or [])
    pred_raw = fo.get("anomaly_intervals")
    pred_intervals: List[list[Any]] = pred_raw if isinstance(pred_raw, list) else []
    gt_set = intervals_to_set(gt_intervals)
    pred_set = intervals_to_set(pred_intervals)
    smoke_eval = {
        "tp_points": len(gt_set & pred_set),
        "fp_points": len(pred_set - gt_set),
        "fn_points": len(gt_set - pred_set),
        "gt_point_count": len(gt_set),
        "pred_point_count": len(pred_set),
    }

    payload: dict[str, Any] = {
        "segment_dir": str(task.seg_dir),
        "state_has_error": bool(out.get("has_error")),
        "final_output": fo,
        "visual_description": out.get("visual_description", ""),
        "visual_localization": out.get("visual_localization", []),
        "visual_localization_items": out.get("visual_localization_items", []),
        "actor_messages": actor_messages_export,
        "detector_messages": detector_messages_export,
        "messages": messages_export,
        "iteration_count": int(out.get("iteration_count") or 0),
        "redo": bool(out.get("redo")),
        "suggestion": str(out.get("suggestion") or ""),
    }
    if not config.no_debug_artifacts:
        payload["ground_truth_intervals"] = gt_intervals
        payload["smoke_point_eval"] = smoke_eval
        labeled = task.seg_dir / "segment_preview_labeled.png"
        if labeled.is_file():
            payload["segment_labeled_png"] = str(labeled.resolve())
        sf = task.seg_dir / "segment_final.png"
        if sf.is_file():
            payload["segment_final_png"] = str(sf.resolve())

    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = config.output_dir / f"{task.stem}.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if not config.no_debug_artifacts:
        p_gt_pred = config.output_dir / f"{task.stem}_smoke_gt_pred.png"
        _write_smoke_gt_pred_png(p_gt_pred, task.stem, pack, gt_intervals, pred_intervals, smoke_eval)
        _copy_segment_final_to_output(task.seg_dir, config.output_dir, task.stem)

    if not config.no_print_messages:
        print(f"[segment {task.stem}] messages={len(messages_export)}")

    rc = 0 if not fo.get("error") and not out.get("has_error") else 2
    row: dict[str, Any] = {
        "segment": task.stem,
        "segment_dir": str(task.seg_dir),
        "exit_code": rc,
        **smoke_eval,
    }
    if rc != 0:
        row["error"] = str(fo.get("error_message") or out.get("error_message") or "unknown error")
    return rc, row


def run_task_with_retries(
    task: SegmentTask,
    *,
    run_once: Callable[[SegmentTask], tuple[int, dict[str, Any]]],
    max_retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    attempts = 0
    last_row: Optional[dict[str, Any]] = None
    last_error: Optional[str] = None

    while attempts < max(1, int(max_retries)):
        attempts += 1
        try:
            rc, row = run_once(task)
            last_row = row
            if rc == 0:
                return {"task": task, "rc": rc, "row": row, "attempts": attempts, "error": None}
            last_error = str(row.get("error") or f"non-zero exit_code={rc}")
        except Exception as exc:
            last_error = str(exc)
        if attempts < max_retries and retry_delay > 0:
            time.sleep(float(retry_delay))

    if last_row is None:
        last_row = {
            "segment": task.stem,
            "segment_dir": str(task.seg_dir.resolve()),
            "exit_code": 2,
            "error": last_error or "unknown error",
            "tp_points": 0,
            "fp_points": 0,
            "fn_points": 0,
            "gt_point_count": 0,
            "pred_point_count": 0,
        }
    return {
        "task": task,
        "rc": int(last_row.get("exit_code", 2)),
        "row": last_row,
        "attempts": attempts,
        "error": last_error,
    }


def _execute_tasks(
    tasks: list[SegmentTask],
    *,
    workflow_factory: Callable[[], Any],
    config: InferConfig,
) -> dict[int, dict[str, Any]]:
    rows_by_index: dict[int, dict[str, Any]] = {}

    def _run_once(task: SegmentTask) -> tuple[int, dict[str, Any]]:
        return _run_segment_once(task, workflow_factory=workflow_factory, config=config)

    pending = [task for task in tasks if not task.skip_existing_ok]
    for task in tasks:
        if task.skip_existing_ok:
            rows_by_index[task.index] = _summary_row_from_resume_json(config.output_dir, task.stem, task.seg_dir)

    if not pending:
        return rows_by_index

    if config.workers <= 1:
        iterator = _iter_progress(pending, total=len(pending), enabled=config.show_progress, desc="infer")
        for task in iterator:
            result = run_task_with_retries(
                task,
                run_once=_run_once,
                max_retries=config.max_retries,
                retry_delay=config.retry_delay,
            )
            if result["rc"] != 0 and result["error"]:
                _write_error_payload(task, config.output_dir, str(result["error"]))
            row = dict(result["row"])
            row["attempts"] = int(result["attempts"])
            row["skipped_resume_ok"] = False
            rows_by_index[task.index] = row
        return rows_by_index

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        future_map: dict[Future[dict[str, Any]], SegmentTask] = {
            executor.submit(
                run_task_with_retries,
                task,
                run_once=_run_once,
                max_retries=config.max_retries,
                retry_delay=config.retry_delay,
            ): task
            for task in pending
        }
        completion_iter = _iter_progress(
            as_completed(future_map),
            total=len(future_map),
            enabled=config.show_progress,
            desc="infer",
        )
        for future in completion_iter:
            task = future_map[future]
            result = future.result()
            if result["rc"] != 0 and result["error"]:
                _write_error_payload(task, config.output_dir, str(result["error"]))
            row = dict(result["row"])
            row["attempts"] = int(result["attempts"])
            row["skipped_resume_ok"] = False
            rows_by_index[task.index] = row
    return rows_by_index


def _sorted_summary_rows(rows_by_index: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [rows_by_index[i] for i in sorted(rows_by_index)]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch AnomaAgent inference: run Locator+Detector on segments under INPUT; write to --output-dir",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_path", type=Path, help="Dataset directory (e.g. data/KPI/train_processed) or a single segment directory")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively find segment dirs under input_path (must contain meta_info.json).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Max recursion depth from input_path (default 6); -1 = unlimited.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Output dir for per-segment json/png and _smoke_summary.json",
    )
    parser.add_argument("--max-turns", type=int, default=6, help="Detector max_turns per segment (default 6)")
    parser.add_argument(
        "--no-reflection",
        action="store_true",
        help="Disable evaluator redo loop (use for SFT/RL data generation; reflection fields stay empty).",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker count (default 1)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max attempts per segment after failure (default 3)")
    parser.add_argument("--retry-delay", type=float, default=0.0, help="Seconds to wait before retry (default 0)")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument(
        "--no-print-messages",
        action="store_true",
        help="Do not print per-segment message summaries (JSON still written)",
    )
    parser.add_argument(
        "--no-debug-artifacts",
        action="store_true",
        help="Skip PNG and ground_truth / smoke_point_eval in JSON",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore successful outputs and rerun all segments (default: skip ok, retry failures)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    # Auto-load local .env files so users don't need to `source` manually.
    # Environment variables already set in the shell take precedence.
    _autoload_env_files(override=False)

    args = parse_args(argv)

    input_path = args.input_path.resolve()
    output_dir = args.output_dir.resolve()
    if not input_path.exists():
        print(f"INPUT not found: {input_path}", file=sys.stderr)
        return 1

    segments = discover_segment_dirs(
        input_path,
        recursive=bool(args.recursive),
        max_depth=int(args.max_depth),
    )
    if not segments and not (input_path / "meta_info.json").is_file() and not bool(args.recursive):
        # Auto-fallback: if non-recursive found nothing, try a bounded recursive search once.
        segments = discover_segment_dirs(input_path, recursive=True, max_depth=int(args.max_depth))
    if not segments:
        print(f"No segments found (need meta_info.json in subdirs, or INPUT is a segment dir): {input_path}", file=sys.stderr)
        return 1

    workers = max(1, int(args.workers))
    no_print_messages = bool(args.no_print_messages) or workers > 1
    if workers > 1 and not args.no_print_messages:
        print("[info] Suppressing message console output in parallel mode; JSON still complete.", file=sys.stderr)

    config = InferConfig(
        output_dir=output_dir,
        max_turns=int(args.max_turns),
        enable_reflection=not bool(args.no_reflection),
        no_print_messages=no_print_messages,
        no_debug_artifacts=bool(args.no_debug_artifacts),
        workers=workers,
        max_retries=max(1, int(args.max_retries)),
        retry_delay=max(0.0, float(args.retry_delay)),
        show_progress=not bool(args.no_progress),
        force=bool(args.force),
    )

    tasks = build_segment_tasks(segments, output_dir=output_dir, force=config.force)
    if not config.force:
        print(
            f"[resume] output-dir={output_dir} "
            f"(ok segments skipped; failures retried; --force disables)",
            file=sys.stderr,
        )

    workflow_factory = _make_workflow_factory(
        max_turns=config.max_turns,
        enable_reflection=config.enable_reflection,
    )
    rows_by_index = _execute_tasks(tasks, workflow_factory=workflow_factory, config=config)
    summary_rows = _sorted_summary_rows(rows_by_index)

    worst_rc = max((int(row.get("exit_code", 2)) for row in summary_rows), default=0)
    skipped_resume_ok = sum(1 for row in summary_rows if row.get("skipped_resume_ok"))
    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "max_turns": config.max_turns,
        "enable_reflection": config.enable_reflection,
        "workers": config.workers,
        "max_retries": config.max_retries,
        "retry_delay": config.retry_delay,
        "segment_count": len(tasks),
        "segments": summary_rows,
        "worst_exit_code": worst_rc,
        "skipped_resume_ok": skipped_resume_ok,
        "resume": not config.force,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== done ===")
    print("wrote:", summary_path.resolve())
    print("worst_exit_code:", worst_rc, "(0=all ok, 2=any failure)")
    return worst_rc


if __name__ == "__main__":
    raise SystemExit(main())
