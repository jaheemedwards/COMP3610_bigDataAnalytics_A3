# === Import necessary modules from each project component ===
from initialize import *
from data_acquisition import *
from data_cleaning import *

# === Define directory and filenames for raw data ===
data_dir = 'data/raw'
reviews_filename = 'reviews'
meta_filename = 'meta'

# === Step 0: Initialize the project (create folders, etc.) ===
initialize_project()

# === Flag to toggle between using saved data or freshly downloaded data ===
use_saved_data = False

if not use_saved_data:
    # === Step 1: Download datasets (if not already downloaded) ===
    download_successful = download_amazon_reviews()

    if download_successful:
        print("\n✅ Download successful.")
        # === Step 2: Load raw review and metadata ===
        raw_reviews_df, raw_meta_df = load_amazon_reviews()

        # Optional: Save the newly downloaded data
        save_dataframe(raw_reviews_df, data_dir, reviews_filename)
        save_dataframe(raw_meta_df, data_dir, meta_filename)
    else:
        print("❌ Download failed. Cannot load datasets.")
        exit()

    reviews_df = raw_reviews_df
    meta_df = raw_meta_df

else:
    print("\n📂 Loading previously saved raw data...")
    # === Load previously saved raw DataFrames ===
    reviews_df = load_dataframe(data_dir, reviews_filename)
    meta_df = load_dataframe(data_dir, meta_filename)

# === Inspect raw dataframes ===
inspect_dataframes(reviews_df, meta_df)

# === Clean and merge review and metadata into one unified DataFrame ===
print("\n>>> 🔄 Cleaning DataFrame... Please wait.\n")
merged_df = clean_dataframe(reviews_df, meta_df)

# === Inspect the cleaned and merged DataFrame ===
inspect_dataframe(merged_df)

# === Save the unified dataset to the processed data directory ===
save_dataframe(merged_df, 'data/processed', 'unified_data')
