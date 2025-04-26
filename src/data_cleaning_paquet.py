from data_acquisition import *
from pathlib import Path
import duckdb
import glob
import pandas as pd

def clean_parquet(path, output_path=None):
    con = duckdb.connect()

    query = f"""
    SELECT DISTINCT ON (user_id, asin, text)
        category,
        rating,
        title,
        title_1 AS product_title,
        text,
        user_id,
        asin,
        parent_asin,
        timestamp,
        verified_purchase,
        helpful_vote,
        average_rating,
        rating_number,
        CASE 
            WHEN try_cast(details AS JSON) IS NOT NULL 
                 AND json_extract_string(details, '$.brand') IS NOT NULL 
                THEN TRIM(json_extract_string(details, '$.brand'))
            WHEN store IS NOT NULL THEN TRIM(store)
            ELSE 'Unknown'
        END AS brand,
        main_category,
        store,
        price,
        LENGTH(regexp_split_to_array(text, '\\W+')) AS review_length
    FROM '{path}'
    WHERE rating BETWEEN 1 AND 5
      AND text IS NOT NULL AND LENGTH(TRIM(text)) > 0
    """

    # Convert to pandas for timestamp processing
    df = con.execute(query).df()

    if "timestamp" in df.columns:
        print("🕒 Cleaning timestamps...")

        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["timestamp"] = df["timestamp"] // 1000  # Convert ms → s

        invalid_timestamps = df[
            (df["timestamp"] <= 0) | (df["timestamp"] >= 2**31)
        ]

        if not invalid_timestamps.empty:
            print("⚠️ Invalid Timestamps Found:")
            print(invalid_timestamps[["timestamp"]].head(10))

        df = df[
            (df["timestamp"] > 0) & (df["timestamp"] < 2**31)
        ]

        print(f"✅ Valid timestamps: {len(df)}")

        if not df.empty:
            df["year"] = pd.to_datetime(
                df["timestamp"], unit="s", errors="coerce"
            ).dt.year
    else:
        df["year"] = None

    if output_path:
        df.to_parquet(output_path, compression="zstd", index=False)
        print(f"📦 Saved cleaned data to: {output_path}")
    else:
        return df

# Correct Paths
src_dir = Path(__file__).resolve().parent
raw_dir = src_dir / "data" / "raw"
cleaned_dir = src_dir / "data" / "cleaned"
cleaned_dir.mkdir(parents=True, exist_ok=True)

# Process each joined parquet file
merged_files = glob.glob(str(raw_dir / "joined_*.parquet"))

if not merged_files:
    print("⚠️ No merged parquet files found.")
else:
    for file_path in merged_files:
        category = Path(file_path).stem.replace("joined_", "")
        output_file = cleaned_dir / f"cleaned_{category}.parquet"
        print(f"\n🔍 Processing category: {category}")
        clean_parquet(file_path, output_file)
