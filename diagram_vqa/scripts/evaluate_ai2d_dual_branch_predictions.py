from __future__ import annotations

import argparse
import json
from pathlib import Path


SETUP_CELLS = (1, 4, 5, 7, 10, 12, 14, 16)


def _exec_notebook_setup(notebook_path: Path) -> dict[str, object]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {
        "__name__": "__dual_branch_evaluation__",
        "display": lambda *_args, **_kwargs: None,
    }

    for cell_index in SETUP_CELLS:
        source = "".join(notebook["cells"][cell_index].get("source", []))
        if cell_index == 10:
            source = source.split("preview_features =", maxsplit=1)[0]
        elif cell_index == 14:
            source = source.split("train_loader =", maxsplit=1)[0]
        exec(compile(source, f"{notebook_path}#cell-{cell_index}", "exec"), namespace)
    return namespace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path(
            "notebooks/experiments/ai2d/"
            "ai2d_sam2_dinov2_dual_branch_evidence_graph.ipynb"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/ai2d/sam2_dinov2_dual_branch_evidence_graph/"
            "seed42_full/checkpoint_best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/ai2d_top_models_100_seed_analysis/dual_branch"),
    )
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    namespace = _exec_notebook_setup(args.notebook.resolve())
    torch = namespace["torch"]
    data_loader = namespace["DataLoader"]
    dataset_type = namespace["AI2DQuestions"]
    collate_questions = namespace["collate_questions"]
    move_batch = namespace["move_batch"]
    model = namespace["model"]
    device = namespace["DEVICE"]

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.eval()

    rows = list(namespace["test_rows"])[: args.max_samples]
    loader = data_loader(
        dataset_type(rows),
        batch_size=int(namespace["EVAL_BATCH_SIZE"]),
        shuffle=False,
        collate_fn=collate_questions,
    )

    predictions: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            raw_rows = batch["rows"]
            moved = move_batch(batch)
            result = model(
                moved["nodes"],
                moved["mask"],
                moved["questions"],
                moved["options"],
                moved["option_mask"],
            )
            predicted_indices = result["vqa_logits"].argmax(1).cpu().tolist()
            for row, predicted_index in zip(raw_rows, predicted_indices, strict=True):
                gold_index = int(row["correct_option_idx"])
                is_correct = int(predicted_index == gold_index)
                predictions.append(
                    {
                        "sample_id": row["sample_id"],
                        "question_id": row["sample_id"],
                        "image_id": str(row["image_id"]),
                        "question": row["question"],
                        "pred_answer": str(row["options"][predicted_index]),
                        "gold_answers": [str(row["options"][gold_index])],
                        "score": float(is_correct),
                        "exact": float(is_correct),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    accuracy = sum(float(row["score"]) for row in predictions) / max(1, len(predictions))
    metrics = {
        "model": "Dual-Branch Evidence Graph",
        "samples": len(predictions),
        "accuracy": accuracy,
        "checkpoint": str(args.checkpoint.resolve()),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
