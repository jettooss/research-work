from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _stable_question_suffix(question: str) -> str:
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()
    return digest[:12]


def make_question_text(question: str) -> str:
    return f"Question: {str(question).strip()}"


def make_option_text(
    question: str,
    option: str,
    short_description: Optional[str] = None,
    use_caption_context: bool = False,
) -> str:
    text = f"Question: {str(question).strip()} Option: {str(option).strip()}."
    if use_caption_context and short_description:
        ctx = str(short_description).strip()
        if ctx:
            text = f"{text} Context: {ctx}"
    return text


@dataclass(frozen=True)
class Ai2dHybridSample:
    sample_id: str
    image_id: str
    split: str
    image_path: str
    ocr_v2_path: Optional[str]
    short_description: Optional[str]
    question: str
    options: tuple[str, ...]
    correct_option_idx: int
    correct_option_text: str
    text_q_only: str
    text_q_plus_correct: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["options"] = list(self.options)
        return payload

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "Ai2dHybridSample":
        options = tuple(str(x) for x in payload.get("options", []) if str(x).strip())
        correct_idx = int(payload.get("correct_option_idx", 0))
        if options:
            correct_idx = max(0, min(correct_idx, len(options) - 1))
        else:
            correct_idx = 0
        correct_text = str(payload.get("correct_option_text", "")).strip()
        if options and not correct_text:
            correct_text = options[correct_idx]
        return Ai2dHybridSample(
            sample_id=str(payload.get("sample_id", "")),
            image_id=str(payload.get("image_id", "")),
            split=str(payload.get("split", "all")),
            image_path=str(payload.get("image_path", "")),
            ocr_v2_path=payload.get("ocr_v2_path"),
            short_description=payload.get("short_description"),
            question=str(payload.get("question", "")),
            options=options,
            correct_option_idx=correct_idx,
            correct_option_text=correct_text,
            text_q_only=str(payload.get("text_q_only", "")),
            text_q_plus_correct=str(payload.get("text_q_plus_correct", "")),
        )


def read_test_ids_csv(path: str | Path) -> set[str]:
    path = Path(path)
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if token:
            out.add(token)
    return out


def _load_caption_short_description(caption_path: Path) -> Optional[str]:
    if not caption_path.exists():
        return None
    payload = json.loads(caption_path.read_text(encoding="utf-8"))
    value = str(payload.get("short_description", "")).strip()
    return value or None


def build_hybrid_samples_from_ai2d(
    ai2d_root: str | Path,
    prepared_root: str | Path,
) -> list[Ai2dHybridSample]:
    ai2d_root = Path(ai2d_root)
    prepared_root = Path(prepared_root)
    images_dir = ai2d_root / "images"
    questions_dir = ai2d_root / "questions"
    ocr_dir = prepared_root / "ocr_v2"
    caption_dir = prepared_root / "caption_v1"

    out: list[Ai2dHybridSample] = []
    for q_path in sorted(questions_dir.glob("*.json")):
        payload = json.loads(q_path.read_text(encoding="utf-8"))
        image_name = str(payload.get("imageName", "")).strip()
        if not image_name:
            continue

        image_path = images_dir / image_name
        if not image_path.exists():
            continue

        image_id = Path(image_name).stem
        ocr_path = ocr_dir / f"{image_id}.ocr.json"
        caption_path = caption_dir / f"{image_id}.caption.json"
        short_description = _load_caption_short_description(caption_path)

        questions = payload.get("questions", {})
        for question_text, q_data in questions.items():
            question = str(question_text).strip()
            answers_raw = q_data.get("answerTexts", []) or []
            options = tuple(str(answer).strip() for answer in answers_raw if str(answer).strip())
            if not options:
                continue

            try:
                correct_idx = int(q_data.get("correctAnswer", 0))
            except (TypeError, ValueError):
                correct_idx = 0
            if correct_idx < 0 or correct_idx >= len(options):
                correct_idx = 0

            question_id = str(q_data.get("questionId", "")).strip()
            suffix = question_id or _stable_question_suffix(question)
            sample_id = f"{image_name}:{suffix}"
            correct_text = options[correct_idx]
            q_only = make_question_text(question)
            q_plus_correct = f"{q_only} Correct answer: {correct_text}."

            out.append(
                Ai2dHybridSample(
                    sample_id=sample_id,
                    image_id=image_id,
                    split="all",
                    image_path=str(image_path),
                    ocr_v2_path=str(ocr_path) if ocr_path.exists() else None,
                    short_description=short_description,
                    question=question,
                    options=options,
                    correct_option_idx=correct_idx,
                    correct_option_text=correct_text,
                    text_q_only=q_only,
                    text_q_plus_correct=q_plus_correct,
                )
            )
    return out


def create_image_level_splits(
    image_ids: Iterable[str],
    test_ids: Iterable[str],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    unique_image_ids = sorted({str(x).strip() for x in image_ids if str(x).strip()})
    requested_test = {str(x).strip() for x in test_ids if str(x).strip()}
    image_id_set = set(unique_image_ids)

    # AI2D explicitly contains images without question files.  Official test
    # IDs for those images still belong to the document split and must not be
    # reported as missing data merely because the question manifest has no row
    # for them.
    document_only_test_ids = sorted(requested_test - image_id_set)
    test_image_ids = sorted(requested_test)

    remainder = [x for x in unique_image_ids if x not in requested_test]
    rng = random.Random(seed)
    rng.shuffle(remainder)

    if not remainder or val_ratio <= 0:
        val_count = 0
    else:
        val_count = int(len(remainder) * float(val_ratio))
        if len(remainder) > 1:
            val_count = max(1, min(len(remainder) - 1, val_count))
        else:
            val_count = 0

    val_image_ids = sorted(remainder[:val_count])
    train_image_ids = sorted(remainder[val_count:])

    return {
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "num_images": len(image_id_set | requested_test),
        "num_question_images": len(unique_image_ids),
        "train_image_ids": train_image_ids,
        "val_image_ids": val_image_ids,
        "test_image_ids": test_image_ids,
        "document_only_test_ids": document_only_test_ids,
        "missing_test_ids": [],
    }


def assign_splits_to_samples(
    samples: Sequence[Ai2dHybridSample],
    split_payload: dict[str, Any],
) -> list[Ai2dHybridSample]:
    train_ids = set(split_payload.get("train_image_ids", []))
    val_ids = set(split_payload.get("val_image_ids", []))
    test_ids = set(split_payload.get("test_image_ids", []))

    out: list[Ai2dHybridSample] = []
    for sample in samples:
        if sample.image_id in test_ids:
            split = "test"
        elif sample.image_id in val_ids:
            split = "val"
        elif sample.image_id in train_ids:
            split = "train"
        else:
            split = "train"
        out.append(replace(sample, split=split))
    return out


def load_manifest_hybrid(path: str | Path) -> list[Ai2dHybridSample]:
    path = Path(path)
    out: list[Ai2dHybridSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        out.append(Ai2dHybridSample.from_dict(payload))
    return out


def write_manifest_hybrid(samples: Sequence[Ai2dHybridSample], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    return output_path


def load_split_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_split_payload(payload: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def select_samples_for_split(
    samples: Sequence[Ai2dHybridSample],
    split_name: str,
    split_payload: Optional[dict[str, Any]] = None,
) -> list[Ai2dHybridSample]:
    split_name = str(split_name).strip().lower()
    if split_payload is None:
        return [sample for sample in samples if sample.split.lower() == split_name]

    image_ids = set(split_payload.get(f"{split_name}_image_ids", []))
    return [sample for sample in samples if sample.image_id in image_ids]


def resolve_sample_file_paths(
    samples: Sequence[Ai2dHybridSample],
    roots: Sequence[str | Path],
) -> list[Ai2dHybridSample]:
    root_paths = [Path(root) for root in roots]

    def _resolve(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute() and path.exists():
            return str(path)
        if path.exists():
            return str(path)
        for root in root_paths:
            candidate = root / path
            if candidate.exists():
                return str(candidate)
        return str(path)

    return [
        replace(
            sample,
            image_path=str(_resolve(sample.image_path) or sample.image_path),
            ocr_v2_path=_resolve(sample.ocr_v2_path),
        )
        for sample in samples
    ]


class Ai2dHybridDataset(Dataset):
    def __init__(self, samples: Sequence[Ai2dHybridSample]) -> None:
        self._samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Ai2dHybridSample:
        return self._samples[idx]


def make_hybrid_collate_fn(use_caption_context: bool = False):
    def _collate(batch: Sequence[Ai2dHybridSample]) -> dict[str, Any]:
        if not batch:
            raise ValueError("batch must be non-empty")

        batch_size = len(batch)
        max_options = max(len(sample.options) for sample in batch)
        option_mask = torch.zeros((batch_size, max_options), dtype=torch.bool)
        correct_indices = torch.zeros((batch_size,), dtype=torch.long)
        option_texts: list[list[str]] = []

        for row_idx, sample in enumerate(batch):
            row_texts: list[str] = []
            for col_idx, option in enumerate(sample.options):
                option_mask[row_idx, col_idx] = True
                row_texts.append(
                    make_option_text(
                        question=sample.question,
                        option=option,
                        short_description=sample.short_description,
                        use_caption_context=use_caption_context,
                    )
                )
            while len(row_texts) < max_options:
                row_texts.append("")
            option_texts.append(row_texts)
            correct_indices[row_idx] = int(sample.correct_option_idx)

        return {
            "sample_ids": [sample.sample_id for sample in batch],
            "image_ids": [sample.image_id for sample in batch],
            "splits": [sample.split for sample in batch],
            "image_paths": [sample.image_path for sample in batch],
            "ocr_paths": [sample.ocr_v2_path for sample in batch],
            "questions": [sample.question for sample in batch],
            "question_texts": [sample.text_q_only for sample in batch],
            "short_descriptions": [sample.short_description for sample in batch],
            "options": [list(sample.options) for sample in batch],
            "option_texts": option_texts,
            "option_mask": option_mask,
            "correct_indices": correct_indices,
            "correct_texts": [sample.correct_option_text for sample in batch],
        }

    return _collate


def flatten_option_texts(
    option_texts: Sequence[Sequence[str]],
    option_mask: torch.Tensor,
) -> tuple[list[str], list[tuple[int, int]]]:
    flat_texts: list[str] = []
    flat_positions: list[tuple[int, int]] = []
    for row_idx, row in enumerate(option_texts):
        for col_idx, text in enumerate(row):
            if bool(option_mask[row_idx, col_idx]):
                flat_texts.append(str(text))
                flat_positions.append((row_idx, col_idx))
    return flat_texts, flat_positions


def encode_text_batch(
    texts: Sequence[str],
    text_encoder,
    cache=None,
    normalize: bool = False,
) -> torch.Tensor:
    if cache is not None and hasattr(cache, "get_text_batch"):
        return cache.get_text_batch(list(texts), text_encoder, normalize=normalize)
    with torch.no_grad():
        embs = text_encoder.encode(list(texts), convert_to_tensor=True, normalize_embeddings=normalize)
    return embs.detach().cpu() if isinstance(embs, torch.Tensor) else torch.tensor(embs, dtype=torch.float32)


def compute_option_logits(
    z_img: torch.Tensor,
    option_texts: Sequence[Sequence[str]],
    option_mask: torch.Tensor,
    text_encoder,
    text_proj,
    cache=None,
    temperature: float = 0.07,
) -> torch.Tensor:
    if z_img.dim() != 2:
        raise ValueError(f"z_img must be 2D [B, D], got shape {tuple(z_img.shape)}")

    batch_size, _ = z_img.shape
    max_options = option_mask.shape[1]
    logits = torch.full(
        (batch_size, max_options),
        fill_value=-1e9,
        dtype=z_img.dtype,
        device=z_img.device,
    )

    flat_texts, flat_positions = flatten_option_texts(option_texts, option_mask)
    if not flat_texts:
        return logits

    text_emb = encode_text_batch(flat_texts, text_encoder=text_encoder, cache=cache, normalize=False).to(z_img.device)
    z_opt = F.normalize(text_proj(text_emb), dim=1)

    for emb_idx, (row_idx, col_idx) in enumerate(flat_positions):
        logits[row_idx, col_idx] = torch.dot(z_img[row_idx], z_opt[emb_idx]) / temperature
    return logits


def positive_mask_from_group_ids(group_ids: Sequence[str], device: Optional[torch.device] = None) -> torch.Tensor:
    ids = [str(x) for x in group_ids]
    if not ids:
        raise ValueError("group_ids must be non-empty")
    return torch.tensor(
        [[left == right for right in ids] for left in ids],
        dtype=torch.bool,
        device=device,
    )


def _multi_positive_ce(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != positive_mask.shape:
        raise ValueError(
            f"positive_mask must have shape {tuple(logits.shape)}, got {tuple(positive_mask.shape)}"
        )
    positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool)
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("each row must contain at least one positive")

    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_probs = torch.logsumexp(
        log_probs.masked_fill(~positive_mask, torch.finfo(logits.dtype).min),
        dim=1,
    )
    return -positive_log_probs.mean()


def contrastive_loss(
    z_img: torch.Tensor,
    z_txt: torch.Tensor,
    temperature: float = 0.07,
    positive_mask: Optional[torch.Tensor] = None,
    group_ids: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    logits = (z_img @ z_txt.t()) / temperature
    if positive_mask is None:
        if group_ids is not None:
            positive_mask = positive_mask_from_group_ids(group_ids, device=logits.device)
        else:
            labels = torch.arange(logits.size(0), device=logits.device)
            positive_mask = F.one_hot(labels, num_classes=logits.size(1)).bool()
    else:
        positive_mask = positive_mask.to(logits.device)
    return (
        _multi_positive_ce(logits, positive_mask)
        + _multi_positive_ce(logits.t(), positive_mask.t())
    ) / 2.0


def recall_at_k_torch(
    sim: torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
    positive_mask: Optional[torch.Tensor] = None,
) -> dict[int, float]:
    if positive_mask is None:
        gt = torch.arange(sim.size(0), device=sim.device).unsqueeze(1)
    else:
        if positive_mask.shape != sim.shape:
            raise ValueError(
                f"positive_mask must have shape {tuple(sim.shape)}, got {tuple(positive_mask.shape)}"
            )
        positive_mask = positive_mask.to(device=sim.device, dtype=torch.bool)
        if not bool(positive_mask.any(dim=1).all()):
            raise ValueError("each row must contain at least one positive")

    ranks = sim.argsort(dim=1, descending=True)
    out: dict[int, float] = {}
    for k in ks:
        k_eff = min(int(k), sim.size(0))
        if positive_mask is None:
            hits = (ranks[:, :k_eff] == gt).any(dim=1)
        else:
            hits = positive_mask.gather(dim=1, index=ranks[:, :k_eff]).any(dim=1)
        out[int(k)] = float(hits.float().mean().item())
    return out


def retrieval_metrics_from_embeddings(
    z_img: torch.Tensor,
    z_txt: torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
    image_ids: Optional[Sequence[str]] = None,
    positive_mask: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    sim = z_img @ z_txt.t()
    if positive_mask is None and image_ids is not None:
        positive_mask = positive_mask_from_group_ids(image_ids, device=sim.device)
    if positive_mask is not None:
        positive_mask = positive_mask.to(device=sim.device, dtype=torch.bool)
    i2t = recall_at_k_torch(sim, ks=ks, positive_mask=positive_mask)
    t2i = recall_at_k_torch(
        sim.t(),
        ks=ks,
        positive_mask=positive_mask.t() if positive_mask is not None else None,
    )
    mean = {int(k): 0.5 * (i2t[int(k)] + t2i[int(k)]) for k in ks}
    return {"i2t": i2t, "t2i": t2i, "mean": mean, "sim": sim}


def vqa_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())
