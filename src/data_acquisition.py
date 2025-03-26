from bigdata_a3_utils import *

def get_amazon_reviews_dataset():
    """
    Downloads, extracts, and loads all Amazon review datasets into a dictionary.
    
    Returns:
        dict: A dictionary where keys are dataset categories and values are loaded data.
    """
    base_save_path = Path("data/raw")
    '''
    # Download datasets (if not already downloaded)
    download_all_amazon_reviews(
        base_save_path=base_save_path,
        compress=True,
        compression_format="gz",
        compression_level=6
    )
    '''
    # Find all .tar.gz files
    compressed_files = list(base_save_path.glob("*.tar.gz"))
    
    # Load datasets into a dictionary
    datasets = {
        file.stem.replace("raw_review_", ""): load_compressed_dataset(str(file)) 
        for file in compressed_files
    }
    
    return datasets