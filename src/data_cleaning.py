from data_acquisition import *
import pandas as pd
import re
import unicodedata

def clean_dataframe(reviews_df, meta_df):
    """
    Cleans and merges review and metadata DataFrames.
    """

    # Normalize and clean 'parent_asin' columns
    for df in [reviews_df, meta_df]:
        df["parent_asin"] = df["parent_asin"].astype(str).str.strip()
        df["parent_asin"] = df["parent_asin"].apply(lambda x: unicodedata.normalize("NFKC", x))

    # Merge DataFrames on 'parent_asin'
    merged_df = reviews_df.merge(meta_df, on="parent_asin", how="inner")
    
    # Data Cleaning
    merged_df = merged_df[merged_df['rating'].between(1, 5)]  # Keep valid ratings
    
    merged_df.dropna(subset=['text'], inplace=True)  # Remove empty reviews
    
    merged_df['store'] = merged_df['store'].fillna("Unknown")  # Fill missing brands
    
    # Remove duplicate reviews
    merged_df.drop_duplicates(subset=['user_id', 'asin', 'text'], keep='first', inplace=True)
    
    # Add review length feature
    merged_df['review_length'] = merged_df['text'].apply(lambda x: len(re.findall(r'\w+', str(x))))

    # Check if 'timestamp' exists in merged_df
    if "timestamp" in merged_df.columns:

        # Convert timestamp to numeric
        merged_df["timestamp"] = pd.to_numeric(merged_df["timestamp"], errors="coerce")

        # Convert milliseconds to seconds
        merged_df["timestamp"] = merged_df["timestamp"] // 1000  # Divide by 1000 to get seconds

        # Identify invalid timestamps (still ensuring reasonable Unix time range)
        invalid_timestamps = merged_df[
            (merged_df["timestamp"] <= 0) | (merged_df["timestamp"] >= 2**31)
        ]
    
        if not invalid_timestamps.empty:
            print("\n=== Invalid Timestamps Found ===")
            print(invalid_timestamps[["timestamp"]].head(10)) 

        # Filter valid timestamps
        merged_df = merged_df[
            (merged_df["timestamp"] > 0) & (merged_df["timestamp"] < 2**31)
        ]
    
        print("\n=== Valid Timestamp Count ===", len(merged_df))

        # Extract year from valid timestamps
        if not merged_df.empty:
            merged_df["year"] = pd.to_datetime(
                merged_df["timestamp"], unit="s", errors="coerce"
            ).dt.year
        
        else:
            print("No valid timestamps left after filtering. Skipping year extraction.")

    else:
        print("No timestamp column found.")

    return merged_df

def inspect_dataframe(df):
    """
    Displays various details about the given DataFrame.
    """
    print("=== DataFrame Info ===")
    print(df.info(), "\n")
    
    print("=== First 5 Rows ===")
    print(df.head(), "\n")
    
    print("=== Shape (Rows, Columns) ===", df.shape, "\n")
    
    print("=== Data Types ===")
    print(df.dtypes, "\n")
    
    print("=== Summary Statistics ===")
    print(df.describe(), "\n")
    
    print("=== Missing Values ===")
    print(df.isnull().sum(), "\n")
    
    print("=== Duplicate Rows ===", df.duplicated().sum(), "\n")
    
    print("=== Column Names ===", df.columns.tolist(), "\n")
    
    print("=== Random Sample ===")
    print(df.sample(5), "\n")