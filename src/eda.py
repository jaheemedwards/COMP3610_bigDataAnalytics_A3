from data_cleaning import *
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns

def run_eda(merged_df):
    # 1. Histogram of Star Ratings
    plt.figure(figsize=(10,5))
    sns.histplot(merged_df["rating"], bins=5, kde=True)
    plt.title("Histogram of Star Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.show(block=True)

    # 2. Top 10 Categories by Review Count
    top_categories = merged_df["main_category"].value_counts().head(10)
    top_categories.plot(kind="bar", title="Top 10 Categories by Review Count")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show(block=True)
    
    #3. Top 10 Brands by Total Review Count (Excluding "Unknown")
    plt.figure(figsize=(12,6))
    brand_counts = merged_df[merged_df["details"] != "Unknown"]["details"].value_counts().head(10)
    brand_counts.plot(kind="bar", color='lightcoral')
    plt.title("Top 10 Brands by Review Count")
    plt.xlabel("Brand")
    plt.ylabel("Review Count")
    plt.xticks(rotation=45)
    plt.show(block=True)

    # 4. Time-Based Trend: Line Chart of Average Star Rating Per Year
    if "review_date" in merged_df.columns:
        merged_df["review_year"] = pd.to_datetime(merged_df["review_date"], unit='s').dt.year  # Convert Unix time to year
        avg_rating_by_year = merged_df.groupby("review_year")["rating"].mean()
        
        plt.figure(figsize=(10,5))
        avg_rating_by_year.plot(kind="line", marker='o', color='b')
        plt.title("Average Star Rating Per Year")
        plt.xlabel("Year")
        plt.ylabel("Average Rating")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show(block=True)
    else:
        print("Column 'review_date' not found. Skipping time-based trend plot.")
    
    
    # 5. Correlation between Review Length and Rating
    print("Pearson Correlation between Review Length and Rating:", merged_df[["review_length", "rating"]].corr().iloc[0,1])