import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent
NB_PATH = EXTERNAL_ROOT / "i2d_gnn_encoder_ocr_knn.ipynb"
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image
from torch.utils.data import Dataset

from vqa_retrieval.datasets import InfographicVQARetrievalDataset


def load_notebook_namespace():
    nb = json.load(open(NB_PATH, "r", encoding="utf-8"))
    ns = {"__name__": "__main__"}
    skip_cells = {14, 17}  # demo and hardcoded train launch
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code" or idx in skip_cells:
            continue
        code = "".join(cell.get("source", []))
        exec(compile(code, f"<nb_cell_{idx}>", "exec"), ns)
    return ns


class InfographicAsAi2dDataset(Dataset):
    def __init__(self, root_dir: str | Path, text_mode: str = "q+correct") -> None:
        self.base = InfographicVQARetrievalDataset(root_dir, split="train", text_mode="q+answers")
        self.samples = [(s.image_path, s.text) for s in self.base]
        self.text_mode = text_mode
        print(f"[InfographicVQA] Loaded {len(self.samples)} (image, text) pairs")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, text = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return image, text, img_path


def main() -> None:
    ns = load_notebook_namespace()
    ns["Ai2dClipDataset"] = InfographicAsAi2dDataset
    train = ns["train"]

    print("[RUN] Starting full training on InfographicVQA (100 epochs)")
    gnn, text_proj, featurizer = train(
        data_root=EXTERNAL_ROOT / "infographicvqa",
        device="cuda",
        batch_size=4,
        epochs=100,
        lr=2e-4,
        show_demo=False,
        eval_max_items=None,
        use_cache=True,
        cache_dir=EXTERNAL_ROOT / "infographicvqa" / "InfographicVQA" / "_cache_features_graph_retrieval",
        extract_min_area=300,
        extract_max_nodes=80,
        extract_ocr_conf=60,
        extract_knn_k=4,
        use_attn_pool=True,
    )

    out_dir = ROOT / "runs" / "infographic_100e"
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(gnn.state_dict(), out_dir / "gnn_final.pt")
    torch.save(text_proj.state_dict(), out_dir / "text_proj_final.pt")
    with (out_dir / "run_info.txt").open("w", encoding="utf-8") as f:
        f.write("dataset=infographicvqa\n")
        f.write("epochs=100\n")
        f.write("batch_size=4\n")
        f.write(f"vision_model={featurizer.vision_model_name}\n")
        f.write(f"text_model={featurizer.text_model_name}\n")
    print(f"[RUN] Saved final weights to {out_dir}")


if __name__ == "__main__":
    main()
