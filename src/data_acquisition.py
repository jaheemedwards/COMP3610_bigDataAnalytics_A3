from bigdata_a3_utils import *

# Define the base save path for storing the dataset
base_save_path = Path(r"data/raw")

# Call the function to download all categories
download_all_amazon_reviews(
    base_save_path=base_save_path,
    compress=True,  # Set to True if you want the downloaded datasets to be compressed
    compression_format="gz",  # You can change this to "bz2" or "xz" based on preference
    compression_level=6  # Adjust compression level (1-9) based on speed/size preference
)