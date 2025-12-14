from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, coalesce, to_json
from pyspark.sql.types import StructType, StructField, StringType

spark = (
    SparkSession.builder
    .appName("count-kafka-all-nulls")
    .master("spark://spark-master:7077")
    .config("spark.jars.packages","org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

kafka_bootstrap = "kafka:9092"
topic = "chicago-crime-stream"

# Read ALL current Kafka data (bounded batch read)
kdf = (
    spark.read
    .format("kafka")
    .option("kafka.bootstrap.servers", kafka_bootstrap)
    .option("subscribe", topic)
    .option("startingOffsets", "earliest")
    .option("endingOffsets", "latest")
    .load()
    .selectExpr("CAST(value AS STRING) AS json_str")
)

# Socrata/API JSON keys (snake_case) + nested location
schema = StructType([
    StructField("id", StringType(), True),
    StructField("case_number", StringType(), True),
    StructField("date", StringType(), True),
    StructField("block", StringType(), True),
    StructField("iucr", StringType(), True),
    StructField("primary_type", StringType(), True),
    StructField("description", StringType(), True),
    StructField("location_description", StringType(), True),
    StructField("arrest", StringType(), True),
    StructField("domestic", StringType(), True),
    StructField("beat", StringType(), True),
    StructField("district", StringType(), True),
    StructField("ward", StringType(), True),
    StructField("community_area", StringType(), True),
    StructField("fbi_code", StringType(), True),
    StructField("x_coordinate", StringType(), True),
    StructField("y_coordinate", StringType(), True),
    StructField("year", StringType(), True),
    StructField("updated_on", StringType(), True),
    StructField("latitude", StringType(), True),
    StructField("longitude", StringType(), True),
    StructField("location", StructType([
        StructField("latitude", StringType(), True),
        StructField("longitude", StringType(), True),
    ]), True),
])

parsed = kdf.select(
    col("json_str"),
    from_json(col("json_str"), schema, {"primitivesAsString": "true"}).alias("data")
)

# Build a "not-null-anywhere" check across all fields (including nested location)
check_cols = [col(f"data.{f.name}") for f in schema.fields if f.name != "location"]
check_cols.append(to_json(col("data.location")))  # treat nested as string for null-check

total = parsed.count()
all_null_count = parsed.filter(coalesce(*check_cols).isNull()).count()
parse_fail_count = parsed.filter(col("data").isNull()).count()

print("Total Kafka messages:", total)
print("All-null after parsing:", all_null_count)
print("Parse failed (data is null):", parse_fail_count)

# Optional: show a few bad messages
parsed.filter(coalesce(*check_cols).isNull()).select("json_str").show(20, truncate=False)

spark.stop()
