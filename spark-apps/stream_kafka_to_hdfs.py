from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, coalesce, to_json
from pyspark.sql.types import StructType, StringType, StructField

# SparkSession that talking to the Spark master container
spark = (
    SparkSession.builder
    .appName("ChicagoCrimeKafkaToHDFS")
    .master("spark://spark-master:7077") # Spark master
    # Kafka Structured Streaming connector
    .config("spark.jars.packages","org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.executor.memory", "3g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# Schema A: CSV-style keys
fields_csv = [
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
schema_csv = StructType([StructField(f, StringType(), True) for f in fields_csv])

fields_api = [
    "id",
    "case_number",
    "date",
    "block",
    "iucr",
    "primary_type",
    "description",
    "location_description",
    "arrest",
    "domestic",
    "beat",
    "district",
    "ward",
    "community_area",
    "fbi_code",
    "x_coordinate",
    "y_coordinate",
    "year",
    "updated_on",
    "latitude",
    "longitude"]
schema_api = StructType([StructField(f, StringType(), True) for f in fields_api] + [
    StructField("location", StructType([
        StructField("latitude", StringType(), True),
        StructField("longitude", StringType(), True),
    ]), True)
])

# Read from Kafka as a stream
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

p_csv = json_df.select(from_json(col("json_str"), schema_csv).alias("c"), col("json_str"))
p_api = p_csv.select(col("json_str"), col("c"),
                     from_json(col("json_str"), schema_api, {"primitivesAsString":"true"}).alias("a"))

# Build unified output
out = p_api.select(
    coalesce(col("c.ID"), col("a.id")).alias("ID"),
    coalesce(col("c.`Case Number`"), col("a.case_number")).alias("Case Number"),
    coalesce(col("c.Date"), col("a.date")).alias("Date"),
    coalesce(col("c.Block"), col("a.block")).alias("Block"),
    coalesce(col("c.IUCR"), col("a.iucr")).alias("IUCR"),
    coalesce(col("c.`Primary Type`"), col("a.primary_type")).alias("Primary Type"),
    coalesce(col("c.Description"), col("a.description")).alias("Description"),
    coalesce(col("c.`Location Description`"), col("a.location_description")).alias("Location Description"),
    coalesce(col("c.Arrest"), col("a.arrest")).alias("Arrest"),
    coalesce(col("c.Domestic"), col("a.domestic")).alias("Domestic"),
    coalesce(col("c.Beat"), col("a.beat")).alias("Beat"),
    coalesce(col("c.District"), col("a.district")).alias("District"),
    coalesce(col("c.Ward"), col("a.ward")).alias("Ward"),
    coalesce(col("c.`Community Area`"), col("a.community_area")).alias("Community Area"),
    coalesce(col("c.`FBI Code`"), col("a.fbi_code")).alias("FBI Code"),
    coalesce(col("c.`X Coordinate`"), col("a.x_coordinate")).alias("X Coordinate"),
    coalesce(col("c.`Y Coordinate`"), col("a.y_coordinate")).alias("Y Coordinate"),
    coalesce(col("c.Year"), col("a.year")).alias("Year"),
    coalesce(col("c.`Updated On`"), col("a.updated_on")).alias("Updated On"),
    coalesce(col("c.Latitude"), col("a.latitude")).alias("Latitude"),
    coalesce(col("c.Longitude"), col("a.longitude")).alias("Longitude"),
    coalesce(col("c.Location"), to_json(col("a.location"))).alias("Location"),
)

# 4. Write to HDFS as Parquet
output_path = "hdfs://namenode:9000/data/raw/chicago_crimes.parquet"
checkpoint_path = "hdfs://namenode:9000/user/spark-checkpoint/chicago_crime_stream"

query = (
    out.writeStream
    .format("parquet")
    .option("path", output_path)
    .option("checkpointLocation", checkpoint_path)
    .outputMode("append")
    .trigger(once=True)
    .start()
)

query.awaitTermination()
spark.stop()
print("[STREAM] Done (once=True).")
