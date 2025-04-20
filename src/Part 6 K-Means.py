import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load CSV in Pandas
df = pd.read_csv("cleaned_merged_data.csv")#update with file name

# Group by product
product_df = df.groupby('asin').agg({
    'rating': 'mean',
    'user_id': 'count',
    'brand': 'first',
    'main_category': 'first'
}).reset_index()

product_df.columns = ['asin', 'mean_rating', 'total_reviews', 'brand', 'category']

# Encode brand and category
le_brand = LabelEncoder()
le_cat = LabelEncoder()
product_df['brand_id'] = le_brand.fit_transform(product_df['brand'].fillna("Unknown"))
product_df['category_id'] = le_cat.fit_transform(product_df['category'].fillna("Unknown"))

# Features for clustering
features = product_df[['mean_rating', 'total_reviews', 'brand_id', 'category_id']]

# Apply k-means
kmeans = KMeans(n_clusters=5, random_state=42)
product_df['cluster'] = kmeans.fit_predict(features)

# Cluster summary
cluster_summary = product_df.groupby('cluster').agg({
    'asin': 'count',
    'mean_rating': 'mean',
    'total_reviews': 'mean',
    'brand_id': 'mean',
    'category_id': 'mean'
}).rename(columns={'asin': 'cluster_size'})

print(cluster_summary)

# Optional: Plot clusters by 2D projection
plt.figure(figsize=(8, 6))
plt.scatter(product_df['mean_rating'], product_df['total_reviews'], c=product_df['cluster'], cmap='viridis')
plt.xlabel('Mean Rating')
plt.ylabel('Total Reviews')
plt.title('K-means Clusters')
plt.colorbar(label='Cluster')
plt.show()