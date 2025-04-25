from data_acquisition import *
from data_cleaning import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

def extract_brand(details):
    try:
        if isinstance(details, str):
            # Convert string representation of a dictionary into an actual dictionary
            details = ast.literal_eval(details)
        # If it's a dictionary, get the 'Brand' key, else return None
        return details.get('Brand') if isinstance(details, dict) else None
    except Exception as e:
        return None

def run_eda(df):
    # 1. Histogram of Star Ratings
    plt.figure(figsize=(10,5))
    sns.histplot(df["rating"], bins=5, kde=True)
    plt.title("Histogram of Star Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.show()

    # 2. Top 10 Categories by Review Count
    top_categories = df["main_category"].value_counts().head(10)
    top_categories.plot(kind="bar", title="Top 10 Categories by Review Count")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

    #3. Top 10 Brands by Total Review Count (Excluding "Unknown")
    # Apply the function to the 'details' column and store the result in a new column 'Brand'
    df['Extracted_Brand'] = df['details'].apply(extract_brand)

    top_10_brands = df['Extracted_Brand'].value_counts().head(10)
    top_10_brands.plot(kind="bar", color='lightcoral')
    plt.title("Top 10 Brands by Review Count")
    plt.xlabel("Brand")
    plt.ylabel("Review Count")
    plt.xticks(rotation=45)
    plt.show()

    # 4. Time-Based Trend: Line Chart of Average Star Rating Per Year
    if "timestamp" in df.columns:
        df["review_year"] = pd.to_datetime(df["timestamp"], unit='s').dt.year  # Convert Unix time to year
        avg_rating_by_year = df.groupby("review_year")["rating"].mean()
        
        plt.figure(figsize=(10,5))
        avg_rating_by_year.plot(kind="line", marker='o', color='b')
        plt.title("Average Star Rating Per Year")
        plt.xlabel("Year")
        plt.ylabel("Average Rating")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Column 'review_date' not found. Skipping time-based trend plot.")
    
    
    # 5. Correlation between Review Length and Rating
    print("Pearson Correlation between Review Length and Rating:", df[["review_length", "rating"]].corr().iloc[0,1])