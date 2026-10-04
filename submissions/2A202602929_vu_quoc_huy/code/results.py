"""Write the required seven-sheet experiment workbook from real run records."""
from __future__ import annotations

from copy import copy
from pathlib import Path

import pandas as pd

SHEET_COLUMNS = {
    "Backbones": [
        "exp_id", "backbone", "pretrained_tag", "params_m", "gmacs", "img_size",
        "epochs", "seed", "best_epoch", "macro_f1_val", "top1_val",
        "chinee_recall_val", "snake_recall_val", "train_seconds_per_epoch",
        "latency_batch1_ms", "notes",
    ],
    "Training": [
        "exp_id", "backbone", "axis", "change_from_T00", "seed", "macro_f1_val",
        "top1_val", "chinee_recall_val", "snake_recall_val",
        "delta_macro_f1_vs_T00", "rare_class_f1", "notes",
    ],
    "Inference": [
        "exp_id", "method", "checkpoint", "views_or_models", "macro_f1_val",
        "top1_val", "ece_val", "p50_ms_batch1", "p95_ms_batch1", "p99_ms_batch1",
        "images_per_s", "relative_cost_vs_I00", "notes",
    ],
    "Final": [
        "exp_id", "configuration", "seed", "macro_f1_val", "macro_f1_test",
        "top1_test", "ece_test", "mean_std_summary",
    ],
    "PerClass": [
        "configuration", "class", "n_test", "precision", "recall", "f1",
    ],
    "Latency": [
        "configuration", "gpu", "dtype", "batch", "img_size", "bn_fused",
        "p50_ms", "p95_ms", "p99_ms", "images_per_s", "torch_version",
        "preprocessing_included",
    ],
    "Summary": [
        "rank", "exp_id", "configuration", "macro_f1_val", "top1_val",
        "params_m", "gmacs", "p95_ms_batch1", "relative_cost", "notes",
    ],
}


def write_results_xlsx(
    path,
    backbones=None,
    training=None,
    inference=None,
    final=None,
    per_class=None,
    latency=None,
    summary=None,
):
    """Write all required sheets; rows should come from actual experiments."""
    tables = {
        "Backbones": backbones,
        "Training": training,
        "Inference": inference,
        "Final": final,
        "PerClass": per_class,
        "Latency": latency,
        "Summary": summary,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, columns in SHEET_COLUMNS.items():
            rows = tables[sheet]
            frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows or [])
            for column in columns:
                if column not in frame:
                    frame[column] = pd.Series(dtype="object")
            frame = frame[columns]
            frame.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cells in ws.columns:
                letter = cells[0].column_letter
                max_length = max(
                    (len(str(cell.value)) for cell in cells if cell.value is not None),
                    default=10,
                )
                ws.column_dimensions[letter].width = min(max(12, max_length + 2), 42)
            for cell in ws[1]:
                header_font = copy(cell.font)
                header_font.bold = True
                cell.font = header_font
    return path
