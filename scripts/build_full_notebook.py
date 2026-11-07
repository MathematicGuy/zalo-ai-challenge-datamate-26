"""Build one notebook that runs the whole workflow (01 EDA -> 02 inference -> 03 evaluation -> 04 training data -> 05 LoRA training)
from the five notebooks, so it never drifts from them.

    .venv/bin/python scripts/build_full_notebook.py                                   # notebooks/00_full_pipeline.ipynb (no outputs)
    .venv/bin/python scripts/build_full_notebook.py --from-dir ~/projects/datamate_notebooks_outputs \
        --out ~/projects/datamate_notebooks_outputs/00_full_pipeline.ipynb            # the same cells with each part's executed outputs

One setup and one import cell at the top; each part keeps its intro, its parameters cell and its cells; the per-part package install
(`install("llm")`, `install("train")`, a no-op in a local checkout) becomes the part's first code cell. Headings move down one level."""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parents[1]
PARTS = ["01_eda", "02_inference", "03_evaluation", "04_training_data", "05_train_lora"]
INTRO = """# DataMate · toàn bộ quy trình (01 → 05)

Một notebook chạy lần lượt cả năm bước của baseline Track 2 (Zalo AI Challenge 2026, DataMate), ghép tự động từ các notebook
`notebooks/01_eda` … `05_train_lora` bằng `scripts/build_full_notebook.py`:

| Phần | Việc | Phần cứng |
|---|---|---|
| 01 · EDA | đọc dữ liệu BTC: shop, tệp, loại đáp án, dạng câu hỏi, rulebook | CPU |
| 02 · Suy luận | pipeline `datamate` trên một split, ghi `submission.jsonl` | GPU bf16 từ 24 GB |
| 03 · Đánh giá | điểm một lượt chạy train, nhóm lỗi, ngưỡng bỏ phiếu; so các bản nộp public | CPU |
| 04 · Dữ liệu train LoRA | dòng SFT từ lượt chạy train có nhãn, ghi chú trước SQL | CPU, GPU cho bước ghi chú |
| 05 · Train LoRA | QLoRA trên Qwen3.5-9B bằng Unsloth | GPU bf16 từ 24 GB |

Mỗi phần có ô tham số riêng ngay đầu phần (giá trị mặc định là cấu hình của bản đã nộp). Chạy từ trên xuống; phần nào không cần có thể bỏ qua.
Trên Kaggle, phần 05 cài stack train khác với vLLM của 02 / 04: chạy 05 trong một phiên riêng (khởi động lại, chạy ô setup rồi tới phần 05)."""


def demote(md: str) -> str:
    return re.sub(r"^(#{1,5}) ", lambda m: "#" + m.group(1) + " ", md, flags=re.M)


def install_kind(setup_source: str) -> str:
    return re.search(r'install\("(\w+)"\)', setup_source).group(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", type=Path, default=REPO / "notebooks", help="where the five notebooks are read (executed copies keep outputs)")
    ap.add_argument("--out", type=Path, default=REPO / "notebooks/00_full_pipeline.ipynb")
    a = ap.parse_args()
    books = {name: nbformat.read(a.from_dir / f"{name}.ipynb", as_version=4) for name in PARTS}
    first = books[PARTS[0]]
    full = nbformat.v4.new_notebook(metadata=copy.deepcopy(first.metadata))
    full.cells.append(nbformat.v4.new_markdown_cell(INTRO))
    setup, imports = copy.deepcopy(first.cells[1]), copy.deepcopy(first.cells[2])
    setup.source = setup.source.replace('install("base")', 'install("base")            # each part below installs what it needs (Kaggle only)')
    full.cells += [setup, imports]
    for name in PARTS:
        nb = books[name]
        intro = copy.deepcopy(nb.cells[0])
        intro.source = demote(intro.source)
        full.cells.append(intro)
        kind = install_kind(nb.cells[1].source)
        if kind != "base":
            note = "  # Kaggle: a separate session for this part (the training stack differs from vLLM)" if kind == "train" else ""
            full.cells.append(nbformat.v4.new_code_cell(f'install("{kind}"){note}'))
        for cell in nb.cells[3:]:
            cell = copy.deepcopy(cell)
            if cell.cell_type == "markdown":
                cell.source = demote(cell.source)
            full.cells.append(cell)
    count = 0
    for cell in full.cells:
        cell.pop("id", None)
        if cell.cell_type == "code":
            if cell.get("outputs"):
                count += 1
                cell.execution_count = count
                for out in cell.outputs:
                    if "execution_count" in out:
                        out["execution_count"] = count
            else:
                cell.execution_count = None
    nbformat.validator.normalize(full)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(full, a.out)
    print(f"{len(full.cells)} cells -> {a.out}")


if __name__ == "__main__":
    main()
