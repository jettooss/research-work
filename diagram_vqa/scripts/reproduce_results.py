"""Recompute published tables from versioned evidence; no models or datasets needed.

This verifies saved measurements and their aggregation, not fresh model inference.
Run from any working directory. --check also detects drift in the two Markdown tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports/reproducibility/evidence.json"
START = "<!-- BEGIN GENERATED RETRIEVAL TABLE -->"
END = "<!-- END GENERATED RETRIEVAL TABLE -->"
DATASET_LABELS = {"ai2d": "AI2D", "docvqa": "DocVQA", "infographicvqa": "InfographicVQA"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def content_digest(value):
    """JSON identity independent of line endings/indentation across Git checkouts."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def check_identity(metrics, item, dataset, architecture):
    if metrics.get("dataset") != dataset or metrics.get("seed") != item["seed"]:
        raise ValueError(f"Run identity mismatch: {item['metrics_path']}")
    expected_model = {"hybrid_v4_multipos_training": "hybrid_gatv2_knn"}.get(architecture, architecture)
    if metrics.get("architecture", metrics.get("model")) not in {architecture, expected_model}:
        raise ValueError(f"Model identity mismatch: {item['metrics_path']}")
    if content_digest(metrics) != item["content_sha256"]:
        raise ValueError(f"Evidence file changed: {item['metrics_path']}")


def percentage_stats(values):
    return {"mean": statistics.mean(values) * 100,
            "sd": statistics.pstdev(values) * 100 if len(values) > 1 else None}


def validate_retrieval(value):
    for direction in ("question_to_document", "document_to_question"):
        scores = [value[direction]["recall_at_k"][str(k)] for k in (1, 5, 10)]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores) or scores != sorted(scores):
            raise ValueError(f"Invalid or nonmonotonic retrieval: {direction}")
    for k in ("1", "5", "10"):
        expected = sum(value[d]["recall_at_k"][k] for d in
                       ("question_to_document", "document_to_question")) / 2
        if not math.isclose(expected, value["mean_recall_at_k"][k], abs_tol=1e-12):
            raise ValueError(f"Invalid bidirectional mean at K={k}")


def compute(evidence_path=EVIDENCE, root=ROOT):
    evidence = read_json(evidence_path)
    accuracy = []
    for group in evidence["accuracy"]:
        if group["dataset"] != "ai2d" or sorted(item["seed"] for item in group["runs"]) != [42, 43, 44]:
            raise ValueError("Incomplete, duplicated or non-AI2D accuracy group")
        values = []
        for item in group["runs"]:
            value = read_json(root / item["metrics_path"])
            check_identity(value, item, group["dataset"], group["architecture"])
            if value.get("status") not in {"available_full", "completed_full", "completed_early_stopped"}:
                raise ValueError(f"Not a complete accuracy run: {item['metrics_path']}")
            test = value.get("test", value)
            if test.get("split", "test") != "test" or test.get("num_samples", 3088) != 3088:
                raise ValueError(f"Not the complete AI2D test evaluation: {item['metrics_path']}")
            for field in item["score_field"].split("."):
                value = value[field]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid accuracy: {item['metrics_path']}")
            values.append(value)
        stats = percentage_stats(values)
        accuracy.append({"model": group["name"], "seeds": [r["seed"] for r in group["runs"]],
                         "percent_by_seed": [v * 100 for v in values], "mean": stats["mean"],
                         "std_ddof0": stats["sd"],
                         "sources": ["diagram_vqa/" + r["metrics_path"] for r in group["runs"]]})
    retrieval = []
    for group in evidence["retrieval"]:
        seeds = [item["seed"] for item in group["runs"]]
        expected_seeds = [42, 43, 44]
        if sorted(seeds) != expected_seeds:
            raise ValueError(f"Incomplete or duplicated seeds: {group['dataset']}/{group['architecture']}")
        measurements = []
        for item in group["runs"]:
            metrics = read_json(root / item["metrics_path"])
            check_identity(metrics, item, group["dataset"], group["architecture"])
            if (metrics["dataset"], metrics["seed"], metrics["num_samples"]) != (
                group["dataset"], item["seed"], group["num_samples"]
            ):
                raise ValueError(f"Run identity or candidate count mismatch: {item['metrics_path']}")
            # Older CLIP/Hybrid metrics omitted the document count. The evidence
            # manifest records the observed input count without inventing a field
            # in those original measurements.
            if "num_documents" in metrics and metrics["num_documents"] != group["num_documents"]:
                raise ValueError(f"Candidate count mismatch: {item['metrics_path']}")
            if metrics.get("split", "test") != "test":
                raise ValueError(f"Not a test measurement: {item['metrics_path']}")
            if metrics.get("status") not in {"available_full", "completed_full", "evaluated_full_test"}:
                raise ValueError(f"Not a complete evaluation: {item['metrics_path']}")
            validate_retrieval(metrics["retrieval"])
            measurements.append(metrics["retrieval"]["mean_recall_at_k"])
        retrieval.append({"dataset": group["dataset"], "architecture": group["architecture"],
                          "name": group["name"], "seeds": seeds, "n": len(seeds),
                          "percent": {k: percentage_stats([m[k] for m in measurements]) for k in ("1", "5", "10")}})
    return {"schema_version": 1, "evidence": "reports/reproducibility/evidence.json",
            "protocol": "bidirectional_mean_unique_documents_v1", "sd_ddof": 0,
            "verification_scope": "Aggregation of saved full-test measurements, not fresh inference or training.",
            "accuracy": accuracy, "retrieval": retrieval}


def table(result):
    rows = ["| Dataset | Model | R@1, % | R@5, % | R@10, % | n |",
            "|---|---|---:|---:|---:|---:|"]
    for group in result["retrieval"]:
        values = []
        for k in ("1", "5", "10"):
            stat = group["percent"][k]
            values.append(f"{stat['mean']:.2f}" + (f" ± {stat['sd']:.2f}" if stat['sd'] is not None else ""))
        rows.append(f"| {DATASET_LABELS[group['dataset']]} | {group['name']} | " + " | ".join(values) + f" | {group['n']} |")
    return "\n".join(rows)


def replace_table(text, rendered):
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("Expected one pair of generated retrieval table markers")
    before, rest = text.split(START)
    _, after = rest.split(END)
    return before + START + "\n" + rendered + "\n" + END + after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Fail on stale versioned JSON or Markdown tables.")
    mode.add_argument("--write", action="store_true", help="Update generated JSON and marked tables.")
    args = parser.parse_args()
    result = compute()
    rendered = table(result)
    snapshot = ROOT / "docs/readme_metrics_verification.json"
    expected_json = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    targets = [(snapshot, expected_json)]
    for path in (ROOT.parent / "README.md", ROOT / "docs/results.md"):
        targets.append((path, replace_table(path.read_text(encoding="utf-8"), rendered)))
    stale = [str(path.relative_to(ROOT.parent)) for path, content in targets
             if path.read_text(encoding="utf-8") != content]
    if args.write:
        for path, content in targets:
            path.write_text(content, encoding="utf-8")
    elif args.check and stale:
        parser.exit(1, "Stale generated results: " + ", ".join(stale) + "\nRun reproduce_results.py --write.\n")
    print(json.dumps({"accuracy_groups": len(result["accuracy"]), "retrieval_groups": len(result["retrieval"]),
                      "saved_full_test_runs": sum(g["n"] for g in result["retrieval"]),
                      "check": "passed" if args.check else "written" if args.write else "computed"}))
    if not args.check:
        print(rendered)


if __name__ == "__main__":
    main()
