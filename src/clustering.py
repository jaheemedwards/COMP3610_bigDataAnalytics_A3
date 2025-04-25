from bigdata_a3_utils import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import joblib


print("🔹 Starting clustering process...")

# Load and sample data
print("🔹 Loading and sampling data...")
base_path = r"D:\Jaheem\jaheemProjetToDelete\COMP3610_bigDataAnalytics_A3-master\src\data\cleaned"
paths = [fr"{base_path}\cleaned_{category}.parquet" for category in VALID_CATEGORIES]

df_list = []
for path, category in zip(paths, VALID_CATEGORIES):
    print(f"   - Reading and sampling 5% of data for category: {category}")
    cat_df = pd.read_parquet(path, engine="pyarrow")
    sample = cat_df.sample(frac=0.05, random_state=42)
    df_list.append(sample)
df = pd.concat(df_list, ignore_index=True)

# Feature engineering
print("🔹 Performing feature engineering...")
product_df = df.groupby('asin').agg({
    'rating': 'mean',
    'user_id': 'count',
    'brand': 'first',
    'main_category': 'first'
}).reset_index()
product_df.columns = ['asin', 'mean_rating', 'total_reviews', 'brand', 'category']

# Encode brand and category
print("🔹 Encoding categorical features...")
le_brand = LabelEncoder()
le_cat = LabelEncoder()
product_df['brand_id'] = le_brand.fit_transform(product_df['brand'].fillna("Unknown"))
product_df['category_id'] = le_cat.fit_transform(product_df['category'].fillna("Unknown"))

# Features for clustering
print("🔹 Preparing data for clustering...")
features = product_df[['mean_rating', 'total_reviews', 'brand_id', 'category_id']]

# Apply K-means clustering
print("🔹 Running KMeans clustering (k=5)...")
kmeans = KMeans(n_clusters=5, random_state=42)
product_df['cluster'] = kmeans.fit_predict(features)

# Cluster summary
print("🔹 Clustering complete. Generating cluster summary...")
cluster_summary = product_df.groupby('cluster').agg({
    'asin': 'count',
    'mean_rating': 'mean',
    'total_reviews': 'mean',
    'brand_id': 'mean',
    'category_id': 'mean'
}).rename(columns={'asin': 'cluster_size'})

print("\n🔹 Cluster Summary:")
print(cluster_summary)

# Save clustered data
output_path = r"D:\Jaheem\jaheemProjetToDelete\COMP3610_bigDataAnalytics_A3-master\src\data\processed\clustered_data.parquet"
product_df.to_parquet(output_path, engine="pyarrow", index=False)
print(f"✅ Clustered data saved to {output_path}")

# Save the KMeans model
print("🔹 Saving KMeans model...")
model_dir = r"D:\Jaheem\jaheemProjetToDelete\COMP3610_bigDataAnalytics_A3-master\src\data\processed\models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "kmeans_model.pkl")
joblib.dump(kmeans, model_path)
print(f"✅ KMeans model saved to {model_path}")

# Plot clusters
print("🔹 Plotting clusters...")
plt.figure(figsize=(10, 7))
scatter = plt.scatter(product_df['mean_rating'], product_df['total_reviews'], 
                      c=product_df['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Mean Rating')
plt.ylabel('Total Reviews')
plt.title('K-means Clusters')
plt.colorbar(scatter, label='Cluster')

for i in range(5):
    cluster_data = product_df[product_df.cluster == i]
    plt.annotate(f"Cluster {i}",
                 (cluster_data['mean_rating'].mean(), cluster_data['total_reviews'].mean()),
                 fontsize=12, weight='bold', color='red')

plt.tight_layout()
plt.show()

print("✅ Clustering script finished.")