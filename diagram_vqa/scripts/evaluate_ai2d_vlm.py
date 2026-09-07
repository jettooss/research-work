from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent
from vqa_retrieval.ai2d_hybrid import (  # noqa: E402
    load_manifest_hybrid,
    load_split_payload,
    resolve_sample_file_paths,
    select_samples_for_split,
)
from vqa_retrieval.ai2d_vlm import (  # noqa: E402
    predict_ai2d_sample_with_vlm,
    summarize_ai2d_vlm_predictions,
)
from vqa_retrieval.public_vqa_metrics import evaluate_public_vqa_rows, write_public_vqa_report  # noqa: E402


def _load_jsonl_by_sample_id(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id:
            out[sample_id] = row
    return out


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen-style VLM direct/rerank baselines on AI2D hybrid splits."
    )
    parser.add_argument("--manifest", type=Path, default=EXTERNAL_ROOT / "ai2d" / "prepared_v2" / "manifest_hybrid.jsonl")
    parser.add_argument("--split-json", type=Path, default=EXTERNAL_ROOT / "ai2d" / "prepared_v2" / "split_hybrid.json")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "ai2d_vlm_direct")
    parser.add_argument("--mode", choices=["direct", "rerank"], default="direct")
    parser.add_argument("--hybrid-predictions", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=EXTERNAL_ROOT / "models" / "Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--device-mode", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-ocr-lines", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ai2d_vlm_config(args)


def run_ai2d_vlm_config(args: argparse.Namespace) -> dict[str, Any]:
    samples = select_samples_for_split(
        resolve_sample_file_paths(
            load_manifest_hybrid(args.manifest),
            roots=[Path.cwd(), ROOT, EXTERNAL_ROOT],
        ),
        split_name=args.split,
        split_payload=load_split_payload(args.split_json),
    )
    if args.max_samples is not None and args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise SystemExit(f"No samples for split={args.split}")

    hybrid_by_id = _load_jsonl_by_sample_id(args.hybrid_predictions)

    from vqa_retrieval.vlm_service import QwenVlmRunner  # Imported lazily to keep unit tests light.

    runner = QwenVlmRunner(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        use_4bit=args.use_4bit,
        device_mode=args.device_mode,
    )

    predictions = []
    for idx, sample in enumerate(samples, start=1):
        helper = hybrid_by_id.get(sample.sample_id)
        pred = predict_ai2d_sample_with_vlm(
            sample=sample,
            generator=runner.generate,
            mode=args.mode,
            hybrid_prediction=helper,
            max_ocr_lines=args.max_ocr_lines,
            max_new_tokens=args.max_new_tokens,
        )
        predictions.append(pred)
        print(
            f"[{idx}/{len(samples)}] {sample.sample_id} "
            f"pred={pred.pred_option_idx} gold={pred.gold_option_idx} correct={pred.is_correct}"
        )

    rows = [pred.to_dict() for pred in predictions]
    summary = summarize_ai2d_vlm_predictions(predictions)
    public_metrics, public_results = evaluate_public_vqa_rows(rows, dataset_name="ai2d")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(rows, args.output_dir / f"{args.split}_{args.mode}_predictions.jsonl")
    (args.output_dir / f"{args.split}_{args.mode}_metrics.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "split": args.split,
                "summary": summary,
                "public_vqa": public_metrics,
                "config": {
                    "manifest": str(args.manifest),
                    "split_json": str(args.split_json),
                    "model_path": str(args.model_path),
                    "adapter_path": None if args.adapter_path is None else str(args.adapter_path),
                    "hybrid_predictions": None
                    if args.hybrid_predictions is None
                    else str(args.hybrid_predictions),
                    "max_samples": args.max_samples,
                    "max_ocr_lines": args.max_ocr_lines,
                    "max_new_tokens": args.max_new_tokens,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_public_vqa_report(
        metrics=public_metrics,
        sample_results=public_results,
        output_dir=args.output_dir,
        prefix=f"public_vqa_{args.split}_{args.mode}",
    )
    print(f"[INFO] accuracy={summary['accuracy']:.4f}")
    print(f"[INFO] artifacts saved under: {args.output_dir}")
    return {"summary": summary, "public_vqa": public_metrics, "predictions": rows}


if __name__ == "__main__":
    main()
