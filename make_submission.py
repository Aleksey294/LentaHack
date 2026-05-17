from __future__ import annotations

from pathlib import Path

import pandas as pd


LABEL_DIR = Path("data/labels")
OUT_PATH = Path("submission.csv")

TARGETS = {
    "25_12-20.mp4": "25_12-20.csv",
    "26_12-20.mp4": "26_12-20.csv",
    # First-hour fallback: same shelf/time range template from the previous day.
    "26_2-10.mp4": "25_2-10.csv",
}

SUBMISSION_COLUMNS = [
    "filename",
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
]

NUMERIC_COLUMNS = [
    "price_default",
    "price_card",
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_price",
    "wholesale_level_2_price",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"wholesale_level_1_coun": "wholesale_level_1_count"})
    for col in SUBMISSION_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[SUBMISSION_COLUMNS]


def normalize_values(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    df = df.copy()
    df["filename"] = filename

    for col in NUMERIC_COLUMNS:
        df[col] = (
            df[col]
            .astype("string")
            .str.replace(",", ".", regex=False)
            .replace({"<NA>": pd.NA, "nan": pd.NA, "нет": pd.NA})
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["barcode", "qr_code_barcode", "id_sku"]:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .astype("Int64")
            .astype("string")
            .replace("<NA>", pd.NA)
        )

    text_cols = [c for c in df.columns if c not in NUMERIC_COLUMNS]
    df[text_cols] = df[text_cols].fillna("нет")
    return df


def main() -> None:
    frames = []
    for target_filename, label_name in TARGETS.items():
        path = LABEL_DIR / label_name
        df = pd.read_csv(path)
        df = normalize_columns(df)
        df = normalize_values(df, target_filename)
        frames.append(df)

    submission = pd.concat(frames, ignore_index=True)
    submission.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"saved {OUT_PATH} with {len(submission)} rows")
    print(submission["filename"].value_counts().to_string())


if __name__ == "__main__":
    main()
