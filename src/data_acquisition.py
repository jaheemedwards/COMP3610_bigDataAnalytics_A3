from bigdata_a3_utils import *
import pandas as pd
import os

base_save_path = 'data/raw'

def download_amazon_reviews():
    """
    Downloads the Amazon review datasets for multiple categories without requiring parameters.
    
    Returns:
        bool: True if download was successful, False otherwise.
    """
    
    try:
        # Download datasets for the specified categories
        download_all_amazon_reviews(
            base_save_path=base_save_path,
            categories=VALID_CATEGORIES,  # Passing the valid categories from the external file
            compress=True,
            compression_format="gz",
            compression_level=6
        )
        return True
    except Exception as e:
        print(f"Error during download: {e}")
        return False

def load_amazon_reviews():
    """
    Loads Amazon review datasets and meta datasets into separate DataFrames without requiring parameters.
    
    Returns:
        tuple: Two pandas DataFrames, one for reviews and one for meta data.
    """

    review_dfs = []
    meta_dfs = []

    for category in VALID_CATEGORIES:  # Using categories from the external file
        try:
            # Load the review dataset
            review_compressed_path = Path(base_save_path) / f"raw_review_{category}.tar.gz"
            review_dataset = load_compressed_dataset(review_compressed_path)
            review_df = pd.DataFrame(review_dataset["full"])  # Access the 'full' split
            review_dfs.append(review_df)

            # Load the meta dataset
            meta_compressed_path = Path(base_save_path) / f"raw_meta_{category}.tar.gz"
            meta_dataset = load_compressed_dataset(meta_compressed_path)
            meta_df = pd.DataFrame(meta_dataset["full"])  # Access the 'full' split
            meta_dfs.append(meta_df)

        except Exception as e:
            print(f"Error loading category {category}: {e}")
            continue  # Skip the current category if there's an error

    # Concatenate all review dataframes into one dataframe
    all_reviews_df = pd.concat(review_dfs, ignore_index=True) if review_dfs else pd.DataFrame()

    # Concatenate all meta dataframes into one dataframe
    all_meta_df = pd.concat(meta_dfs, ignore_index=True) if meta_dfs else pd.DataFrame()

    return all_reviews_df, all_meta_df

def save_dataframe(df, save_dir, filename):
    """
    Saves the given DataFrame to a CSV file in the specified directory.
    
    :param df: DataFrame to save
    :param save_dir: Directory where the file should be saved
    :param filename: Name of the CSV file (without extension)
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    save_path = os.path.join(save_dir, f"{filename}.csv")
    
    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")

def load_dataframe(save_dir, filename):
    """
    Loads a CSV file into a DataFrame from the specified directory with dtype handling.
    
    :param save_dir: Directory where the file is located
    :param filename: Name of the CSV file (without extension)
    :return: Loaded DataFrame
    """
    file_path = os.path.join(save_dir, f"{filename}.csv")
    
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, low_memory=False)  # Avoid DtypeWarning
            print(f"Loaded DataFrame from {file_path}")
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    else:
        print(f"File not found: {file_path}")
        return None


def inspect_dataframes(reviews_df, meta_df):
    """
    Inspects the reviews and meta DataFrames, providing a summary of each.
    
    Args:
        reviews_df (pd.DataFrame): DataFrame containing review data.
        meta_df (pd.DataFrame): DataFrame containing meta data.
    
    Returns:
        None: Prints various summary information for both DataFrames.
    """
    # Inspect reviews DataFrame
    print("=== Reviews DataFrame ===")
    
    # View the first few rows
    print("\nFirst 5 rows:")
    print(reviews_df.head())
    
    # View the shape (rows, columns)
    print("\nShape (rows, columns):")
    print(reviews_df.shape)
    
    # Get a summary of the DataFrame
    print("\nSummary of DataFrame:")
    print(reviews_df.info())
    
    # Get basic statistics (numerical columns)
    print("\nSummary Statistics (Numerical):")
    print(reviews_df.describe())
    
    # Check for missing values
    print("\nMissing Values Count:")
    print(reviews_df.isnull().sum())
    
    # List all column names
    print("\nColumn Names:")
    print(reviews_df.columns)
    
    # Check for duplicates
    #print("\nDuplicate Rows Count:")
    #print(reviews_df.duplicated().sum())
    
    # Sample random rows
    print("\nRandom Sample (5 rows):")
    print(reviews_df.sample(5))

    # Inspect meta DataFrame
    print("\n=== Meta DataFrame ===")
    
    # View the first few rows
    print("\nFirst 5 rows:")
    print(meta_df.head())
    
    # View the shape (rows, columns)
    print("\nShape (rows, columns):")
    print(meta_df.shape)
    
    # Get a summary of the DataFrame
    print("\nSummary of DataFrame:")
    print(meta_df.info())
    
    # Get basic statistics (numerical columns)
    print("\nSummary Statistics (Numerical):")
    print(meta_df.describe())
    
    # Check for missing values
    print("\nMissing Values Count:")
    print(meta_df.isnull().sum())
    
    # List all column names
    print("\nColumn Names:")
    print(meta_df.columns)
    
    # Check for duplicates
    #print("\nDuplicate Rows Count:")
    #print(meta_df.duplicated().sum())
    
    # Sample random rows
    print("\nRandom Sample (5 rows):")
    print(meta_df.sample(5))
