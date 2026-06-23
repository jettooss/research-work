import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Sequence


@dataclass(frozen=True)
class RetrievalSample:
    dataset_name: str
    split: str
    image_path: str
    question: str
    answers: tuple[str, ...]
    text: str
    sample_id: str
    ocr_path: Optional[str] = None


class BaseRetrievalDataset(Sequence[RetrievalSample]):
    dataset_name = "base"

    def __init__(self) -> None:
        self._samples: List[RetrievalSample] = []

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> RetrievalSample:
        return self._samples[idx]

    def __iter__(self) -> Iterator[RetrievalSample]:
        return iter(self._samples)

    @staticmethod
    def _build_text(question: str, answers: Iterable[str], text_mode: str) -> str:
        answers_list = [a for a in answers if a is not None]
        if text_mode == "q_only":
            return f"Question: {question}"
        if text_mode == "q+answers":
            first = answers_list[0] if answers_list else ""
            return f"Question: {question} Correct answer: {first}."
        if text_mode == "q+all_answers":
            joined = "; ".join(str(a) for a in answers_list)
            return f"Question: {question} Answers: {joined}."
        raise ValueError(f"Unsupported text_mode: {text_mode}")

    @staticmethod
    def open_image_rgb(image_path: str) -> Any:
        from PIL import Image

        return Image.open(image_path).convert("RGB")

    def summary(self, limit: int = 3) -> dict:
        existing = sum(1 for s in self._samples if Path(s.image_path).exists())
        return {
            "dataset": self.dataset_name,
            "samples": len(self._samples),
            "images_exist": existing,
            "missing_images": len(self._samples) - existing,
            "preview_ids": [s.sample_id for s in self._samples[:limit]],
        }


class Ai2dRetrievalDataset(BaseRetrievalDataset):
    dataset_name = "ai2d"

    def __init__(self, root_dir: str | Path, text_mode: str = "q+answers") -> None:
        super().__init__()
        root = Path(root_dir)
        images_dir = root / "images"
        questions_dir = root / "questions"

        for q_path in sorted(questions_dir.glob("*.json")):
            with q_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            image_name = payload.get("imageName")
            if not image_name:
                continue

            image_path = images_dir / image_name
            if not image_path.exists():
                continue

            questions = payload.get("questions", {})
            for q_text, q_data in questions.items():
                answers = tuple(q_data.get("answerTexts", []) or ())
                if not answers:
                    continue
                qid = str(q_data.get("questionId", ""))
                sample_id = f"{image_name}:{qid or hash(q_text)}"
                self._samples.append(
                    RetrievalSample(
                        dataset_name=self.dataset_name,
                        split="all",
                        image_path=str(image_path),
                        question=str(q_text),
                        answers=answers,
                        text=self._build_text(str(q_text), answers, text_mode),
                        sample_id=sample_id,
                    )
                )


class DocVQARetrievalDataset(BaseRetrievalDataset):
    dataset_name = "docvqa"

    def __init__(self, root_dir: str | Path, split: str = "train", text_mode: str = "q+answers") -> None:
        super().__init__()
        root = Path(root_dir) / split
        ann_path = root / f"{split}_v1.0.json"
        with ann_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        for item in payload.get("data", []):
            rel_image = str(item.get("image", "")).strip()
            if not rel_image:
                continue
            image_path = root / rel_image
            if not image_path.exists():
                alt = root / "documents" / Path(rel_image).name
                image_path = alt if alt.exists() else image_path

            question = str(item.get("question", "")).strip()
            answers = tuple(item.get("answers", []) or ())
            qid = str(item.get("questionId", ""))

            ocr_json = root / "ocr_results" / (Path(rel_image).stem + ".json")
            ocr_path = str(ocr_json) if ocr_json.exists() else None

            self._samples.append(
                RetrievalSample(
                    dataset_name=self.dataset_name,
                    split=split,
                    image_path=str(image_path),
                    question=question,
                    answers=answers,
                    text=self._build_text(question, answers, "q_only" if not answers else text_mode),
                    sample_id=qid or f"{split}:{len(self._samples)}",
                    ocr_path=ocr_path,
                )
            )


class InfographicVQARetrievalDataset(BaseRetrievalDataset):
    dataset_name = "infographicvqa"

    def __init__(self, root_dir: str | Path, split: str = "train", text_mode: str = "q+answers") -> None:
        super().__init__()
        root = Path(root_dir) / "InfographicVQA"
        ann_path = root / "infographicsvqa_qas" / f"infographicsVQA_{split}_v1.0.json"
        images_dir = root / "infographicsvqa_images"
        with ann_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        for item in payload.get("data", []):
            image_name = str(item.get("image_local_name", "")).strip()
            if not image_name:
                continue
            image_path = images_dir / image_name
            question = str(item.get("question", "")).strip()
            answers = tuple(item.get("answers", []) or ())
            qid = str(item.get("questionId", ""))
            self._samples.append(
                RetrievalSample(
                    dataset_name=self.dataset_name,
                    split=split,
                    image_path=str(image_path),
                    question=question,
                    answers=answers,
                    text=self._build_text(question, answers, "q_only" if not answers else text_mode),
                    sample_id=qid or f"{split}:{len(self._samples)}",
                )
            )


def dataset_from_name(name: str, root_dir: str | Path, split: Optional[str] = None) -> BaseRetrievalDataset:
    key = name.strip().lower()
    if key == "ai2d":
        return Ai2dRetrievalDataset(root_dir)
    if key == "docvqa":
        return DocVQARetrievalDataset(root_dir, split=split or "train")
    if key == "infographicvqa":
        return InfographicVQARetrievalDataset(root_dir, split=split or "train")
    raise ValueError(f"Unknown dataset: {name}")
