from bigdata_a3_utils import *
from data_acquisition import *
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, count

# Create Spark Session
spark = SparkSession.builder \
    .appName("ALS Recommender") \
    .config("spark.local.dir", "D:/spark-temp") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer", "128m") \
    .config("spark.kryoserializer.buffer.max", "1024m") \
    .getOrCreate()

# Load the cleaned CSV

base_path = r"D:\Jaheem\jaheemProjetToDelete\COMP3610_bigDataAnalytics_A3-master\src\data\cleaned"

# Construct the full paths
paths = [fr"{base_path}\cleaned_{category}.parquet" for category in VALID_CATEGORIES]
df = spark.read.parquet(*paths)

# Define a 5% sample fraction for each category
fractions = {category: 0.05 for category in VALID_CATEGORIES}

# Perform stratified sampling
df = df.stat.sampleBy("main_category", fractions, seed=42)

# Keep necessary columns
ratings = df.select("user_id", "asin", "rating")

# Drop users with < 5 reviews
user_counts = ratings.groupBy("user_id").agg(count("*").alias("review_count"))
filtered_users = user_counts.filter("review_count >= 5").select("user_id")
ratings = ratings.join(filtered_users, on="user_id", how="inner")

# Encode user_id and asin to numeric for ALS
from pyspark.ml.feature import StringIndexer

user_indexer = StringIndexer(inputCol="user_id", outputCol="userIndex").fit(ratings)
item_indexer = StringIndexer(inputCol="asin", outputCol="itemIndex").fit(ratings)

ratings = user_indexer.transform(ratings)
ratings = item_indexer.transform(ratings)

# Train-test split
train, test = ratings.randomSplit([0.8, 0.2], seed=42)

# ALS model
als = ALS(userCol="userIndex", itemCol="itemIndex", ratingCol="rating",
          coldStartStrategy="drop", nonnegative=True)
recommender_model = als.fit(train)

# Evaluate
predictions = recommender_model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"ALS RMSE: {rmse:.4f}")

# Show recommendations for 3 random users
user_subset = ratings.select("userIndex").distinct().limit(3)
user_recs = recommender_model.recommendForUserSubset(user_subset, 5)

user_recs.show(truncate=False)