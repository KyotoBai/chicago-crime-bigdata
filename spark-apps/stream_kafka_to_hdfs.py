from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType

# 1. SparkSession talking to the Spark master container
spark = (
    SparkSession.builder
    .appName("ChicagoCrimeKafkaToHDFS")
    .master("spark://spark-master:7077") # Talk to Spark master
    # Add Kafka Structured Streaming connector
    .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0"
        )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# 2. Schema for the raw JSON (all strings in the raw zone)
fields = [
    "ID",
    "Case Number",
    "Date",
    "Block",
    "IUCR",
    "Primary Type",
    "Description",
    "Location Description",
    "Arrest",
    "Domestic",
    "Beat",
    "District",
    "Ward",
    "Community Area",
    "FBI Code",
    "X Coordinate",
    "Y Coordinate",
    "Year",
    "Updated On",
    "Latitude",
    "Longitude",
    "Location",
]

schema = StructType()
for f in fields:
    schema = schema.add(f, StringType(), nullable=True)

# 3. Read from Kafka as a stream
kafka_bootstrap = "kafka:9092"
topic = "chicago-crime-stream"

raw_kafka_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", kafka_bootstrap)
    .option("subscribe", topic)
    .option("startingOffsets", "earliest")
    .option("failOnDataLoss", "false")
    .load()
)

json_df = raw_kafka_df.selectExpr("CAST(value AS STRING) AS json_str")

parsed_df = (
    json_df
    .select(from_json(col("json_str"), schema).alias("data"))
    .select("data.*")
)

# 4. Write to HDFS as Parquet
output_path = "hdfs://namenode:9000/data/raw/chicago_crimes.parquet"
checkpoint_path = "hdfs://namenode:9000/user/jovyan/checkpoint/chicago_crime_stream"

query = (
    parsed_df.writeStream
    .format("parquet")
    .option("path", output_path)
    .option("checkpointLocation", checkpoint_path)
    .outputMode("append")
    .start()
)

query.awaitTermination()
