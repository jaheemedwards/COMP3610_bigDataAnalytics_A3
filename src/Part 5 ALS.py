from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, count

# Create Spark Session
spark = SparkSession.builder \
    .appName("ALS Recommender") \
    .getOrCreate()

# Load the cleaned CSV
df = spark.read.csv("cleaned_merged_data.csv", header=True, inferSchema=True)#update with file name

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
model = als.fit(train)

# Evaluate
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"ALS RMSE: {rmse:.4f}")

# Show recommendations for 3 random users
user_subset = ratings.select("userIndex").distinct().limit(3)
user_recs = model.recommendForUserSubset(user_subset, 5)

user_recs.show(truncate=False)