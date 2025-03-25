# Clustering (K-means)
# Implement k-means clustering here
from sklearn.cluster import KMeans

def apply_kmeans(df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    df['cluster'] = kmeans.fit_predict(df)
    return df