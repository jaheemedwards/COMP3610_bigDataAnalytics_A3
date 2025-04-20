from data_acquisition import *
from data_cleaning import *
from eda import *
from binary_sentiment import *
from recommender_system import *
from clustering import *


'''
# Step 1: Download datasets (if not already downloaded)
#download_successful = download_amazon_reviews()
download_successful = True

if download_successful:
    # Step 2: Load reviews and meta data after successful download
    raw_reviews_df, raw_meta_df = load_amazon_reviews()
else:
    print("Download failed. Cannot load datasets.")

inspect_dataframes(raw_reviews_df, raw_meta_df)
'''
# Define directory
data_dir = 'data/raw'
reviews_filename = 'reviews'
meta_filename = 'meta'

'''
save_dataframe(raw_reviews_df, data_dir, reviews_filename)
save_dataframe(raw_meta_df, data_dir, meta_filename)
'''
print("&&&&&&&&&&&  cleaning dataframe  &&&&&&&&&&&")

# Load the previously saved DataFrames
reviews_df = load_dataframe(data_dir, reviews_filename)
meta_df = load_dataframe(data_dir, meta_filename)

merged_df = clean_dataframe(reviews_df, meta_df)

inspect_dataframe(merged_df)

#save_dataframe(merged_df, 'data\processed', 'unified_data')

print("&&&&&&&&&&&  running eda  &&&&&&&&&&&")

run_eda(merged_df)

print("&&&&&&&&&&&  running sentiment analysis  &&&&&&&&&&&")

run_sentiment_analysis(merged_df)

