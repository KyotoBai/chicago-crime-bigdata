"""
STEP 5: Analytics & Visualization
Generate insights and create visualizations
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import pymongo
import plotly.express as px
import plotly.graph_objects as go

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ========================================
# SPARK SESSION & DATA LOADING
# ========================================
spark = SparkSession.builder \
    .appName("ChicagoCrimeAnalytics") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

print("✅ Loading processed data from HDFS...")
df = spark.read.parquet("hdfs://namenode:9000/data/processed/chicago_crimes_clean.parquet")
print(f"📊 Total records: {df.count()}")

# ========================================
# ANALYTICS 1: CRIME TRENDS OVER TIME
# ========================================
print("\n📈 Analyzing crime trends over time...")

crime_by_year = df.groupBy("Year").agg(
    count("*").alias("crime_count")
).orderBy("Year").toPandas()

# Save to MongoDB
client = pymongo.MongoClient("mongodb://admin:admin123@mongodb:27017/")
db = client["crime_analysis"]
db["crime_by_year"].insert_many(crime_by_year.to_dict('records'))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(crime_by_year['Year'], crime_by_year['crime_count'], marker='o', linewidth=2)
plt.title('Chicago Crime Incidents Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Crimes', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/jovyan/work/crime_trends.png', dpi=300, bbox_inches='tight')
print("✅ Saved: crime_trends.png")

# ========================================
# ANALYTICS 2: CRIME TYPES DISTRIBUTION
# ========================================
print("\n📊 Analyzing crime types...")

crime_types = df.groupBy("Primary Type").agg(
    count("*").alias("count")
).orderBy(desc("count")).limit(15).toPandas()

# Save to MongoDB
db["crime_types"].insert_many(crime_types.to_dict('records'))

# Plot
plt.figure(figsize=(14, 8))
plt.barh(crime_types['Primary Type'], crime_types['count'], color='steelblue')
plt.xlabel('Number of Incidents', fontsize=12)
plt.title('Top 15 Crime Types in Chicago', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('/home/jovyan/work/crime_types.png', dpi=300, bbox_inches='tight')
print("✅ Saved: crime_types.png")

# ========================================
# ANALYTICS 3: HOURLY PATTERNS
# ========================================
print("\n⏰ Analyzing hourly patterns...")

hourly_crimes = df.groupBy("hour").agg(
    count("*").alias("crime_count")
).orderBy("hour").toPandas()

# Save to MongoDB
db["hourly_patterns"].insert_many(hourly_crimes.to_dict('records'))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(hourly_crimes['hour'], hourly_crimes['crime_count'], marker='o', linewidth=2, color='crimson')
plt.title('Crime Distribution by Hour of Day', fontsize=16, fontweight='bold')
plt.xlabel('Hour (24-hour format)', fontsize=12)
plt.ylabel('Number of Crimes', fontsize=12)
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/jovyan/work/hourly_patterns.png', dpi=300, bbox_inches='tight')
print("✅ Saved: hourly_patterns.png")

# ========================================
# ANALYTICS 4: ARREST RATES BY CRIME TYPE
# ========================================
print("\n🚔 Analyzing arrest rates...")

arrest_analysis = df.groupBy("Primary Type").agg(
    count("*").alias("total_crimes"),
    sum("arrest_made").alias("arrests")
).withColumn(
    "arrest_rate", 
    (col("arrests") / col("total_crimes") * 100)
).orderBy(desc("arrest_rate")).limit(15).toPandas()

# Save to MongoDB
db["arrest_rates"].insert_many(arrest_analysis.to_dict('records'))

# Plot
plt.figure(figsize=(14, 8))
plt.barh(arrest_analysis['Primary Type'], arrest_analysis['arrest_rate'], color='forestgreen')
plt.xlabel('Arrest Rate (%)', fontsize=12)
plt.title('Arrest Rates by Crime Type (Top 15)', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('/home/jovyan/work/arrest_rates.png', dpi=300, bbox_inches='tight')
print("✅ Saved: arrest_rates.png")

# ========================================
# ANALYTICS 5: DISTRICT COMPARISON
# ========================================
print("\n🗺️ Analyzing by district...")

district_analysis = df.groupBy("District").agg(
    count("*").alias("crime_count"),
    avg("arrest_made").alias("arrest_rate")
).orderBy(desc("crime_count")).limit(20).toPandas()

# Save to MongoDB
db["district_analysis"].insert_many(district_analysis.to_dict('records'))

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Crime counts
ax1.barh(district_analysis['District'].astype(str), district_analysis['crime_count'], color='coral')
ax1.set_xlabel('Number of Crimes', fontsize=12)
ax1.set_title('Crime Count by District', fontsize=14, fontweight='bold')
ax1.invert_yaxis()

# Arrest rates
ax2.barh(district_analysis['District'].astype(str), district_analysis['arrest_rate'] * 100, color='teal')
ax2.set_xlabel('Arrest Rate (%)', fontsize=12)
ax2.set_title('Arrest Rate by District', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('/home/jovyan/work/district_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: district_analysis.png")

# ========================================
# ANALYTICS 6: CRIME HEATMAP
# ========================================
print("\n🗺️ Creating crime heatmap...")

# Sample data for heatmap (using a subset for performance)
sample_df = df.filter(
    (col("Latitude").isNotNull()) & 
    (col("Longitude").isNotNull())
).sample(fraction=0.01, seed=42).select("Latitude", "Longitude").toPandas()

# Create folium map centered on Chicago
chicago_map = folium.Map(
    location=[41.8781, -87.6298],
    zoom_start=11,
    tiles='OpenStreetMap'
)

# Prepare data for heatmap
heat_data = [[row['Latitude'], row['Longitude']] for _, row in sample_df.iterrows()]

# Add heatmap layer
HeatMap(heat_data, radius=10, blur=15, max_zoom=13).add_to(chicago_map)

# Save map
chicago_map.save('/home/jovyan/work/crime_heatmap.html')
print("✅ Saved: crime_heatmap.html")

# ========================================
# SUMMARY STATISTICS
# ========================================
print("\n📊 Generating summary statistics...")

total_crimes = df.count()
total_arrests = df.filter(col("arrest_made") == 1).count()
arrest_rate = (total_arrests / total_crimes) * 100

most_common_crime = df.groupBy("Primary Type").count().orderBy(desc("count")).first()
most_dangerous_district = df.groupBy("District").count().orderBy(desc("count")).first()

summary = {
    "total_crimes": int(total_crimes),
    "total_arrests": int(total_arrests),
    "overall_arrest_rate": float(arrest_rate),
    "most_common_crime": most_common_crime["Primary Type"],
    "most_common_crime_count": int(most_common_crime["count"]),
    "highest_crime_district": str(most_dangerous_district["District"]),
    "highest_crime_district_count": int(most_dangerous_district["count"])
}

# Save to MongoDB
db["summary_stats"].insert_one(summary)

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total Crimes: {summary['total_crimes']:,}")
print(f"Total Arrests: {summary['total_arrests']:,}")
print(f"Overall Arrest Rate: {summary['overall_arrest_rate']:.2f}%")
print(f"Most Common Crime: {summary['most_common_crime']} ({summary['most_common_crime_count']:,} incidents)")
print(f"Highest Crime District: {summary['highest_crime_district']} ({summary['highest_crime_district_count']:,} incidents)")
print("="*60)

print("\n✅ All analytics complete!")
print("📁 Results saved to: /home/jovyan/work/")
print("📁 MongoDB collections populated: crime_analysis database")

spark.stop()
