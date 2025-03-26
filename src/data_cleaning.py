from data_acquisition import *
import pandas as pd
import re

def clean_dataframe(reviews_df, meta_df):
    # Debugging: Check the first few rows and shape of both DataFrames
    print("Reviews DataFrame Shape:", reviews_df.shape)
    print("Meta DataFrame Shape:", meta_df.shape)
    
    print("Reviews DataFrame Sample:\n", reviews_df.head())
    print("Meta DataFrame Sample:\n", meta_df.head())
    
    # Merge on parent_asin
    merged_df = reviews_df.merge(meta_df, on='parent_asin', how='left')
    
    # Debugging: Check the result of the merge
    print("Merged DataFrame Shape:", merged_df.shape)
    print("Merged DataFrame Sample:\n", merged_df.head())
    
    # Handle Invalid / Missing Values
    merged_df = merged_df[merged_df['rating'].between(1, 5)]  # Keep ratings in [1,5]
    merged_df = merged_df.dropna(subset=['text'])  # Drop rows with empty review text
    
    # Fill missing brand info
    merged_df['brand'] = merged_df['store'].fillna("Unknown")
    
    # Remove Duplicates
    merged_df = merged_df.drop_duplicates(subset=['user_id', 'asin', 'text'], keep='first')
    
    # Derived Columns
    merged_df['review_length'] = merged_df['text'].apply(lambda x: len(re.findall(r'\w+', str(x))))  # Count words
    
    # Convert timestamp safely
    if 'timestamp' in merged_df.columns:
        merged_df['timestamp'] = pd.to_numeric(merged_df['timestamp'], errors='coerce')  # Convert to numeric
        merged_df = merged_df[(merged_df['timestamp'] > 0) & (merged_df['timestamp'] < 2**31)]  # Remove out-of-bounds values
        merged_df['year'] = pd.to_datetime(merged_df['timestamp'], unit='s', errors='coerce').dt.year
    
    return merged_df



def inspect_dataframe(df):
    print("=== DataFrame Info ===")
    print(df.info(), "\n")  # Displays data types, non-null counts, and memory usage

    print("=== First 5 Rows ===")
    print(df.head(), "\n")
    
    print("=== Shape (Rows, Columns) ===")
    print(df.shape, "\n")
    
    print("=== Data Types ===")
    print(df.dtypes, "\n")
    
    print("=== Summary Statistics (Numerical Columns) ===")
    print(df.describe(), "\n")
    
    print("=== Missing Values Count ===")
    print(df.isnull().sum(), "\n")
    
    print("=== Duplicate Rows Count ===")
    print(df.duplicated().sum(), "\n")
    
    print("=== Column Names ===")
    print(df.columns.tolist(), "\n")
    
    print("=== Random Sample (5 Rows) ===")
    print(df.sample(5), "\n")
