NOTE!!!!: the folder or file marked using [NONE] are not in current project files, need to create

# Proeject Ideally Structure

```
chicago-crime-bigdata/
│
├── docker-compose.yml
├── hadoop.env
├── README.md
│
├── data/                           # Raw input data (mounted to containers)
│   └── chicago_crimes.csv          # Downloaded from Kaggle
│
├── dockerfiles/jupyter 
│   └── chicago_crimes.csv 
│
├── notebooks/                      # Jupyter notebooks (your code)
│   ├── .ipynb [NONE]
│   ├── .ipynb [NONE]
│   ├── .ipynb [NONE]
│   ├── ...
│   ├── ...
│   ├── .ipynb [NONE]
│   │
│   └── output/ [NONE]                    # Generated visualizations
│       ├── .png
│       ├── .png
│       ├── .png
│       ├── .png
│       ├── .png
│       └── crime_heatmap.html
│
└── spark-apps/                     # Spark application scripts
    └── batch_jobs/ [NONE]                # For production Spark jobs
```

# HDFS Ideally Structure

```
HDFS (hdfs://namenode:9000/)
│
├── /data/                         # Main data directory
│   │
│   ├── raw/                       # Raw ingested data (immutable)
│   │   ├── ... [NONE]
│   │   │   
│   │   │
│   │   └── _SUCCESS [NONE]              # Marker file (write completed)
│   │
│   ├── processed/                 # Cleaned & transformed data
│   │   ├──...[NONE]
│   │   │
│   │   └── _SUCCESS[NONE]
│   │
│   ├── results/                   # ML predictions & analytics
│   │   ├──...[NONE]
│   │   │
│   │   └── aggregated/   [NONE]         # Pre-computed aggregations
│   │       ...
│   │
│   └── staging/      [NONE]             # Temporary processing area
│       └── temp_transformations/[NONE]
│
├── /user/                         # User directories
│   ├── spark/                     # Spark working directory
│   │   ├── .sparkStaging/[NONE]
│   │   └── warehouse/  [NONE]           # Spark SQL warehouse
│   │
│   └── jovyan/          [NONE]          # Jupyter user directory
│       └── checkpoint/   [NONE]         # Streaming checkpoints
│
└── /tmp/         [NONE]                 # Temporary files
    └── spark-temp/[NONE]
```

# MongoDB Ideally Structure
NOTE: only init database crime_analysis is there, no other new table in side
```
MongoDB (mongodb:27017)
│
├── admin                          # System database
│   └── system.users
│
└── crime_analysis                 # Your project database
    │
    ├── ml_results                 # ML model metrics
    │   └── documents:
    │       ├── {
    │       │     "_id": ObjectId("..."),
    │       │     "model_type": "RandomForestClassifier",
    │       │     "timestamp": "2024-12-06 10:30:00",
    │       │     "metrics": {
    │       │       "auc": 0.85,
    │       │       "accuracy": 0.78,
    │       │       "f1_score": 0.80
    │       │     },
    │       │     "feature_importance": [...]
    │       │   }
    │       └── ...
    │
    ├── crime_by_year              # Analytics: yearly trends
    │   └── documents:
    │       ├── {"Year": 2001, "crime_count": 485716}
    │       ├── {"Year": 2002, "crime_count": 488209}
    │       └── ...
    │
    ├── crime_types                # Analytics: crime type distribution
    │   └── documents:
    │       ├── {"Primary Type": "THEFT", "count": 1523047}
    │       ├── {"Primary Type": "BATTERY", "count": 1238567}
    │       └── ...
    │
    ├── hourly_patterns            # Analytics: crimes by hour
    │   └── documents:
    │       ├── {"hour": 0, "crime_count": 125432}
    │       ├── {"hour": 1, "crime_count": 98234}
    │       └── ...
    │
    ├── arrest_rates               # Analytics: arrest effectiveness
    │   └── documents:
    │       ├── {
    │       │     "Primary Type": "NARCOTICS",
    │       │     "total_crimes": 489528,
    │       │     "arrests": 441287,
    │       │     "arrest_rate": 90.14
    │       │   }
    │       └── ...
    │
    ├── district_analysis          # Analytics: by district
    │   └── documents:
    │       ├── {
    │       │     "District": "11",
    │       │     "crime_count": 125678,
    │       │     "arrest_rate": 0.23
    │       │   }
    │       └── ...
    │
    └── summary_stats              # Overall statistics
        └── documents:
            └── {
                  "_id": ObjectId("..."),
                  "total_crimes": 7500000,
                  "total_arrests": 1875000,
                  "overall_arrest_rate": 25.0,
                  "most_common_crime": "THEFT",
                  "highest_crime_district": "11"
                }
```

# Data Ingestion

```
┌────────────────┐
│  Kaggle CSV    │
│  (8M records)  │
│  1.5 GB        │
└────────┬───────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  STEP 1: Kafka Producer (Python)           │
│  - Reads CSV line by line                  │
│  - Converts to JSON                        │
│  - Publishes to Kafka topic                │
└────────┬───────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  KAFKA BROKER                              │
│  Topic: chicago-crime-stream               │
│  - Stores in partitions                    │
│  - Retains for 7 days (default)            │
│  - Allows replay                           │
└────────┬───────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  STEP 2: Spark Streaming Consumer          │
│  - Reads from Kafka                        │
│  - Processes in micro-batches              │
│  - Parallelized across 2 workers           │
└────────┬───────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  HDFS: /data/raw/                          │
│  chicago_crimes.parquet                    │
│  - Distributed across 3 datanodes          │
│  - Replicated 2x for fault tolerance       │
│  - Compressed (Parquet format)             │
│                                            │
└────────────────────────────────────────────┘
```
---

# Data Transformation

```
┌────────────────────────────────────────────┐
│  HDFS: /data/raw/                          │
│  chicago_crimes.parquet                    │
└────────┬───────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────┐
│  STEP 3: Spark ETL Job                        │
│                                               │
│  TRANSFORMATION PIPELINE:                     │
│  1. Data Cleaning                             │
│     - Remove nulls                            │
│     - Drop duplicates                         │
│                                               │
│  2. Feature Engineering                       │
|     - Timestamp parsed                        |
|       (can sort, filter by time)              |
│     - Extract time features                   │
|       (hour, day, month)                      |
|       (Create Time Period Categories)         |
|       (change the time period based on sesson)|
│     - Create binary flags                     │
│     - Categorize crimes                       │
|       (severity, time period)                 |
│     - Calculate risk scores                   │
│                                               │
│  PARALLEL EXECUTION:                          │
│  ┌──────────────┐  ┌──────────────┐           │
│  │ Worker 1     │  │ Worker 2     │           │
│  │ Processes:   │  │ Processes:   │           │
│  │ - Part 0-3   │  │ - Part 4-7   │           │
│  └──────────────┘  └──────────────┘           │
└────────┬──────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  HDFS: /data/processed/                    │
│  chicago_crimes_clean.parquet              │
│                                            │
│  PARTITIONED BY YEAR:                      │
│  ├── Year=2001/ (15 MB, 485K records)     │
│  ├── Year=2002/ (16 MB, 488K records)     │
│  ├── Year=2003/ (17 MB, 475K records)     │
│  ├── ...                                   │
│  └── Year=2024/ (12 MB, 312K records)     │
│                                            │
│                                            │
└────────────────────────────────────────────┘
```

# Machine Learning
```
┌────────────────────────────────────────────┐
│  HDFS: /data/processed/                    │
│  chicago_crimes_clean.parquet              │
└────────┬───────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  STEP 4: Spark ML Pipeline                │
│                                            │
│  1. Feature Preparation                    │
│     - String indexing (categorical → int)  │
│     - One-hot encoding                     │
│     - Vector assembly                      │
│                                            │
│  2. Train-Test Split                       │
│     - 80% train (6M records)               │
│     - 20% test (1.5M records)              │
│                                            │
│  3. Distributed Training                   │
│     Random Forest (100 trees)              │
│     ┌──────────────┐  ┌──────────────┐    │
│     │ Worker 1     │  │ Worker 2     │    │
│     │ Trains:      │  │ Trains:      │    │
│     │ - Trees 0-49 │  │ - Trees 50-99│    │
│     └──────────────┘  └──────────────┘    │
│                                            │
│  4. Model Evaluation                       │
│     - AUC-ROC: 0.85                        │
│     - Accuracy: 78%                        │
│     - F1 Score: 0.80                       │
└────────┬───────────────────────────────────┘
         │
         ├──────────────────────┐
         ▼                      ▼
┌─────────────────┐   ┌──────────────────────┐
│  HDFS           │   │  MONGODB             │
│  /data/results/ │   │  crime_analysis      │
│  predictions    │   │  - ml_results        │
│  .parquet       │   │  - feature_importance│
│  (1.5M records) │   │  - model_metrics     │
|                 |   |  - ...other data     |
└─────────────────┘   └──────────────────────┘
```

# Analytics & Visualization
```
┌────────────────────────────────────────────┐
│  HDFS: /data/processed/                    │
│  chicago_crimes_clean.parquet              │
└────────┬───────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  STEP 5: Spark Analytics Jobs             │
│                                            │
│  PARALLEL AGGREGATIONS:                    │
│  ┌──────────────┐  ┌──────────────┐       │
│  │ Job 1        │  │ Job 2        │       │
│  │ Group by:    │  │ Group by:    │       │
│  │ - Year       │  │ - Crime Type │       │
│  └──────────────┘  └──────────────┘       │
│  ┌──────────────┐  ┌──────────────┐       │
│  │ Job 3        │  │ Job 4        │       │
│  │ Group by:    │  │ Group by:    │       │
│  │ - Hour       │  │ - District   │       │
│  └──────────────┘  └──────────────┘       │
└────────┬───────────────────────────────────┘
         │
         ├──────────────────────┐
         ▼                      ▼
┌─────────────────┐   ┌──────────────────────┐
│  MONGODB        │   │  LOCAL FILESYSTEM    │
│  Collections:   │   │  ./notebooks/output/ │
│  - crime_by_year│   │                      │
│  - crime_types  │   │  Visualizations:     │
│  - hourly       │   │  - crime_trends.png  │
│  - arrests      │   │  - crime_types.png   │
│  - districts    │   │  - hourly_patterns   │
│  - summary      │   │  - arrest_rates.png  │
└─────────┬───────┘   │  - district_analysis │
          │           │  - crime_heatmap.html│
          │           └──────────────────────┘
          ▼
┌──────────────────────┐
│  GRAFANA DASHBOARD   │
│  :3000               │
│                      │
│  Real-time charts:   │
│  - Time series       │
│  - Bar charts        │
│  - Gauges            │
│  - Tables            │
└──────────────────────┘
```