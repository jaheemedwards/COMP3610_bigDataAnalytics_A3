from data_acquisition import *
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import count
from pyspark.ml.feature import StringIndexer
import os

print("🚀 Starting ALS recommendation pipeline...")

# Spark session
print("🔧 Initializing Spark session...")
spark = SparkSession.builder \
    .appName("ALS Recommender") \
    .config("spark.local.dir", "D:/spark-temp") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer", "128m") \
    .config("spark.kryoserializer.buffer.max", "1024m") \
    .getOrCreate()

# Load parquet files
print("📂 Loading cleaned parquet files...")
base_path = r"D:\Jaheem\jaheemProjetToDelete\COMP3610_bigDataAnalytics_A3-master\src\data\cleaned"
paths = [fr"{base_path}\cleaned_{category}.parquet" for category in VALID_CATEGORIES]
df = spark.read.parquet(*paths)
print("✅ Parquet files loaded.")

# Stratified sample
print("🔍 Taking 5% stratified sample from each category...")
fractions = {category: 0.05 for category in VALID_CATEGORIES}
df = df.stat.sampleBy("main_category", fractions, seed=42)
print("✅ Sampling complete.")

# Filter users with >= 5 reviews
print("🧹 Filtering users with fewer than 5 reviews...")
ratings = df.select("user_id", "asin", "rating")
user_counts = ratings.groupBy("user_id").agg(count("*").alias("review_count"))
filtered_users = user_counts.filter("review_count >= 5").select("user_id")
ratings = ratings.join(filtered_users, on="user_id", how="inner")
print("✅ User filtering complete.")

# Index user and item
print("🔢 Indexing user_id and asin...")
user_indexer = StringIndexer(inputCol="user_id", outputCol="userIndex").fit(ratings)
item_indexer = StringIndexer(inputCol="asin", outputCol="itemIndex").fit(ratings)
ratings = user_indexer.transform(ratings)
ratings = item_indexer.transform(ratings)
print("✅ Indexing complete.")

# Train-test split
print("🔀 Splitting data into training and test sets...")
train, test = ratings.randomSplit([0.8, 0.2], seed=42)
print("✅ Split complete.")

# Train ALS
print("🧠 Training ALS model...")
als = ALS(userCol="userIndex", itemCol="itemIndex", ratingCol="rating",
          coldStartStrategy="drop", nonnegative=True)
recommendation_model = als.fit(train)
print("✅ ALS training complete.")

# Evaluate
print("📊 Evaluating ALS model...")
predictions = recommendation_model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"📈 ALS RMSE: {rmse:.4f}")

# Save model
print("💾 Saving model...")
model_path = r"D:\Jaheem\jaheemProjetToDelete\COMP3610_bigDataAnalytics_A3-master\src\data\processed\models\als_model"
if not os.path.exists(model_path):
    recommendation_model.save(model_path)
    print(f"📦 Model saved to: {model_path}")
else:
    print(f"⚠️ Model directory already exists: {model_path}. Skipping save.")

# Show top 5 recommendations for 3 users
print("👥 Showing top 5 recommendations for 3 sample users...")
user_subset = ratings.select("userIndex").distinct().limit(3)
user_recs = recommendation_model.recommendForUserSubset(user_subset, 5)
user_recs.show(truncate=False)

print("✅ ALS recommendation pipeline completed.")
