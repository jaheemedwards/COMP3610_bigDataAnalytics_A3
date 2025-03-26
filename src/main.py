from data_acquisition import *
from data_cleaning import *
from eda import *
from binary_sentiment import *
from recommender_system import *
from clustering import *

# Print the keys (categories) and first few rows of one dataset
datasets = get_amazon_reviews_dataset

print("Categories:", list(datasets.keys()))
print(datasets["Automotive"].head())  # Change to any available category