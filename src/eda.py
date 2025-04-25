import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import ast
import numpy as np

def save_plot(plot, filename):
    print(f"💾 Saving plot: {filename}")
    save_path = r"D:\Jaheem\jaheemProjetToDelete\COMP3610_bigDataAnalytics_A3-master\src\data\processed"
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, filename)
    plot.savefig(file_path, bbox_inches='tight')
    plot.close()

def run_star_rating_histogram(base_cleaned_path):
    all_ratings = []
    for file in os.listdir(base_cleaned_path):
        if file.endswith(".parquet"):
            file_path = os.path.join(base_cleaned_path, file)
            print(f"✅ Loading file: {file_path}")
            try:
                df = pd.read_parquet(file_path, columns=["rating"])
                all_ratings.append(df["rating"])
            except:
                print(f"⚠️ Skipping {file}: 'rating' column not found.")

    if all_ratings:
        
        print("🔍 Calculating Star Rating Histogram...")
        ratings_series = pd.concat(all_ratings)
        # Calculate the total number of ratings
        total_ratings = len(ratings_series)
        bins = np.arange(0.5, 6, 1)  # Define bin edges for 1-5 stars

    # Create the figure first to avoid overwriting
        plt.figure(figsize=(12, 6))

    # Plot histogram with matplotlib and set alignment to 'center'
        counts, bins, patches = plt.hist(
        ratings_series, bins=bins, edgecolor='black', align='mid', color='skyblue'
        )

        plt.title("Histogram of Star Ratings")
        plt.xlabel("Rating")
        plt.ylabel("Count")
        plt.xticks([1, 2, 3, 4, 5])
        plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'))
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        # 📐 Scale y-axis so tallest bar takes ~75% of the graph height
        y_max = plt.gca().get_ylim()[1]
        plt.ylim(0, y_max * 1.33)

        # 📊 Add percentage labels above each bar
        for count, patch in zip(counts, patches):
            percentage = (count / total_ratings) * 100
            plt.text(patch.get_x() + patch.get_width() / 2, count + y_max * 0.01,
                 f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)

        # 🧮 Annotate the total number of ratings
        plt.text(0.95, 0.95, f'Total Ratings: {total_ratings:,}', ha='right', va='top',
            transform=plt.gca().transAxes, fontsize=12, color='black',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

        plt.tight_layout()
        save_plot(plt, "star_rating_histogram.png")



def run_top_categories_by_review_count(base_cleaned_path):
    all_categories = []
    for file in os.listdir(base_cleaned_path):
        if file.endswith(".parquet"):
            file_path = os.path.join(base_cleaned_path, file)
            print(f"✅ Loading file: {file_path}")
            try:
                df = pd.read_parquet(file_path, columns=["main_category"])
                all_categories.append(df["main_category"])
            except:
                print(f"⚠ Skipping {file}: 'main_category' column not found.")

    if all_categories:
        print("🔍 Calculating Top Categories by Review Count...")
        categories_series = pd.concat(all_categories)
        top_categories = categories_series.value_counts().head(10).reset_index()
        top_categories.columns = ['Category', 'Count']
        # Calculate the total number of categories
        total_categories = len(categories_series)
        plt.figure(figsize=(14, 7))
        sns.barplot(data=top_categories, x='Category', y='Count', palette='viridis')
        plt.title("Top 10 Categories by Review Count")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'))
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Grid lines on the y-axis
        # Annotate total number of categories in the top right corner
        plt.text(0.95, 0.95, f'Total Categories: {total_categories:,}', ha='right', va='top', 
                 transform=plt.gca().transAxes, fontsize=12, color='black', 
                 bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
        plt.tight_layout()
        save_plot(plt, "top_categories_by_review_count.png")

def run_top_brands_by_review_count(base_cleaned_path):
    all_brands = []

    for file in os.listdir(base_cleaned_path):
        if file.endswith(".parquet"):
            file_path = os.path.join(base_cleaned_path, file)
            print(f"✅ Loading file: {file_path}")
            try:
                # Read the parquet file with a focus on the 'brand' column
                df = pd.read_parquet(file_path, columns=["brand"])

                # Clean data
                df["brand"] = df["brand"].str.strip()  # Remove any leading/trailing spaces
                df = df[df["brand"].notna()]  # Remove rows with NaN values in the 'brand' column
                df = df[df["brand"].str.lower() != "unknown"]  # Remove 'unknown' brands

                all_brands.append(df["brand"])

            except Exception as e:
                print(f"⚠ Skipping {file}: 'brand' column not found or other error: {e}")

    if all_brands:
        print("🔍 Calculating Top Brands by Review Count...")

        # Concatenate all brands into one series
        brands_series = pd.concat(all_brands)

        # Get top 10 brands by review count
        top_brands = brands_series.value_counts().head(10)
        # Calculate the total number of brands
        total_brands = len(brands_series)

        # Plot the top brands
        plt.figure(figsize=(14, 6))
        # Specify colors using a seaborn color palette
        sns.barplot(x=top_brands.index, y=top_brands.values, palette='crest')
        plt.title("Top 10 Brands by Review Count (Excl. 'Unknown')")
        plt.xlabel("Brand")
        plt.ylabel("Review Count")
        plt.xticks(rotation=45)

        # Format y-axis for human-readable numbers
        plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'))
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Grid lines on the y-axis
        # Annotate total number of brands in the top right corner
        plt.text(0.95, 0.95, f'Total Brands: {total_brands:,}', ha='right', va='top', 
                 transform=plt.gca().transAxes, fontsize=12, color='black', 
                 bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
        plt.tight_layout()
        save_plot(plt, "top_brands_by_review_count.png")
    else:
        print("⚠ No valid brands found across files.")




def run_avg_star_rating_per_year(base_cleaned_path):
    all_years_and_ratings = []

    for file in os.listdir(base_cleaned_path):
        if file.endswith(".parquet"):
            file_path = os.path.join(base_cleaned_path, file)
            print(f"✅ Loading file: {file_path}")
            try:
                df = pd.read_parquet(file_path, columns=["timestamp", "rating"])

                # Convert and validate timestamps
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce") // 1000  # Convert to seconds
                    df = df[(df["timestamp"] > 0) & (df["timestamp"] < 2**31)]  # Filter valid timestamps
                    df["review_year"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce").dt.year
                    df = df.dropna(subset=["review_year", "rating"])
                    df["review_year"] = df["review_year"].astype(int)
                    all_years_and_ratings.append(df[["review_year", "rating"]])
            except:
                print(f"⚠ Skipping {file}: 'timestamp' or 'rating' column not found or unreadable.")

    if all_years_and_ratings:
        print("🔍 Calculating Average Star Rating Per Year...")
        combined_df = pd.concat(all_years_and_ratings)
        avg_rating_by_year = combined_df.groupby("review_year")["rating"].mean().reset_index()
        # Calculate the moving average (e.g., 3-year window)
        avg_rating_by_year['rolling_avg'] = avg_rating_by_year['rating'].rolling(window=3).mean()
        # Calculate the total number of ratings
        total_ratings = len(combined_df)
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=avg_rating_by_year, x='review_year', y='rating', marker='o', color='b')
        sns.lineplot(data=avg_rating_by_year, x='review_year', y='rolling_avg', marker='o', label='3-Year Moving Avg', color='g')
        plt.title("Average Star Rating Per Year with Moving Average")
        plt.xlabel("Year")
        plt.ylabel("Average Rating")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Grid lines on the y-axis
        plt.xticks(rotation=45)
        # Add a legend to distinguish between the lines
        plt.legend(loc='upper left')
        # Annotate the total number of ratings in the top right corner
        plt.text(0.95, 0.95, f'Total Ratings: {total_ratings:,}', ha='right', va='top', 
                 transform=plt.gca().transAxes, fontsize=12, color='black', 
                 bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
        plt.tight_layout()
        save_plot(plt, "avg_star_rating_per_year_with_rolling_avg.png")


def run_correlation_review_length_rating(base_cleaned_path):
    all_reviews = []
    for file in os.listdir(base_cleaned_path):
        if file.endswith(".parquet"):
            file_path = os.path.join(base_cleaned_path, file)
            print(f"✅ Loading file: {file_path}")
            try:
                df = pd.read_parquet(file_path, columns=["review_length", "rating"])
                all_reviews.append(df[["review_length", "rating"]])
            except:
                print(f"⚠ Skipping {file}: 'review_length' or 'rating' column not found.")

    if all_reviews:
        print("🔍 Calculating Correlation between Review Length and Rating...")
        combined_df = pd.concat(all_reviews)
        corr = combined_df["review_length"].corr(combined_df["rating"])
        print(f"📈 Pearson Correlation between Review Length and Rating: {corr:.3f}")
        if corr > 0:
            print("🔍 Interpretation: Longer reviews tend to have higher ratings.")
        elif corr < 0:
            print("🔍 Interpretation: Longer reviews tend to have lower ratings.")
        else:
            print("🔍 Interpretation: No correlation between review length and rating.")


def run_eda_on_all_cleaned(base_cleaned_path):
    print("📂 Starting to run EDA on all cleaned data...")
    run_star_rating_histogram(base_cleaned_path)
    run_top_categories_by_review_count(base_cleaned_path)
    run_top_brands_by_review_count(base_cleaned_path)
    run_avg_star_rating_per_year(base_cleaned_path)
    run_correlation_review_length_rating(base_cleaned_path)

# Set your base cleaned path
base_cleaned_path = r"D:\Jaheem\jaheemProjetToDelete\COMP3610_bigDataAnalytics_A3-master\src\data\cleaned"
run_eda_on_all_cleaned(base_cleaned_path)