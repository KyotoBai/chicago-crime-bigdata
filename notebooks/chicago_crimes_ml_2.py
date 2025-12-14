from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml import Pipeline
import pymongo
from datetime import datetime

# 和你的 ETL 一致的输入
PROCESSED_PATH = "hdfs://namenode:9000/data/processed/chicago_crimes_clean.parquet"
# hotspot 预测结果输出
RESULTS_PATH = "hdfs://namenode:9000/data/results/hotspot_predictions_rf.parquet"


def build_spark():
    spark = (
        SparkSession.builder.appName("ChicagoCrimeHotspotML")
        .master("spark://spark-master:7077")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_processed(spark):
    df = spark.read.parquet(PROCESSED_PATH)
    return df


def build_monthly_aggregates(df):
    """
    从事件级别聚合到 (crime_year, crime_month, District) 级别
    """
    df = df.dropna(subset=["District", "crime_year", "crime_month"])

    monthly = (
        df.groupBy("crime_year", "crime_month", "District")
        .agg(
            F.count("*").alias("crime_count"),
            F.sum(F.col("is_violent").cast("int")).alias("violent_count"),
            F.avg("risk_score").alias("avg_risk_score"),
            F.avg("location_risk_weight").alias("avg_location_risk"),
        )
    )

    # 每个 District 的“历史平均犯罪数”（到当前月份之前的平均）
    w_dist_time_full = (
        Window.partitionBy("District")
        .orderBy("crime_year", "crime_month")
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    monthly = monthly.withColumn(
        "long_term_avg_crime",
        F.avg("crime_count").over(w_dist_time_full),
    )

    return monthly


def add_temporal_features(monthly):
    """
    为每个 (year, month, District) 增加：
      - prev_1m_crime_count / prev_3m_crime_avg
      - prev_1m_violent_count / prev_3m_violent_avg
      - long_term_avg_crime（已在上一步计算）
    并在每个月内部按 crime_count 排名，定义 is_hotspot（二分类标签）
    """
    # 按 District + 时间排序，构造历史特征
    w_dist_time = Window.partitionBy("District").orderBy("crime_year", "crime_month")

    monthly = monthly.withColumn(
        "prev_1m_crime_count",
        F.lag("crime_count", 1).over(w_dist_time),
    ).withColumn(
        "prev_1m_violent_count",
        F.lag("violent_count", 1).over(w_dist_time),
    )

    # 前 3 个月平均（不含当前月）
    w_prev3 = w_dist_time.rowsBetween(-3, -1)
    monthly = monthly.withColumn(
        "prev_3m_crime_avg",
        F.avg("crime_count").over(w_prev3),
    ).withColumn(
        "prev_3m_violent_avg",
        F.avg("violent_count").over(w_prev3),
    )

    # 在每个 (year, month) 内，根据 crime_count 排序，取 Top 20% 为 hotspot
    w_month = Window.partitionBy("crime_year", "crime_month").orderBy(
        F.desc("crime_count")
    )
    monthly = monthly.withColumn(
        "crime_rank_pct",
        F.percent_rank().over(w_month),
    )

    HOT_PCT = 0.2
    monthly = monthly.withColumn(
        "is_hotspot",
        (F.col("crime_rank_pct") <= HOT_PCT).cast("double"),
    )

    # 需要的特征列不能为 null —— 去掉最开始几个月没有历史数据的行
    feature_cols = [
        "prev_1m_crime_count",
        "prev_3m_crime_avg",
        "prev_1m_violent_count",
        "prev_3m_violent_avg",
        "long_term_avg_crime",
        "avg_risk_score",
        "avg_location_risk",
    ]
    monthly = monthly.dropna(subset=feature_cols)

    return monthly


def time_based_train_test_split(monthly_feat):
    """
    按时间做 train/test 划分：
      - 如果年份跨度 >= 4 年： 用最早到 (max_year - 2) 作为 train，最后两年作为 test
      - 否则 fallback 到 randomSplit 80/20
    """
    stats = monthly_feat.select(
        F.min("crime_year").alias("min_year"),
        F.max("crime_year").alias("max_year"),
    ).first()

    min_year = stats["min_year"]
    max_year = stats["max_year"]
    print(f"Year range in monthly data: {min_year} ~ {max_year}")

    if max_year - min_year >= 4:
        cutoff_year = max_year - 2
        print(f"Using temporal split: train <= {cutoff_year}, test > {cutoff_year}")
        train_df = monthly_feat.filter(F.col("crime_year") <= cutoff_year)
        test_df = monthly_feat.filter(F.col("crime_year") > cutoff_year)
    else:
        print("Year span too small, using random split 80/20 instead.")
        train_df, test_df = monthly_feat.randomSplit([0.8, 0.2], seed=42)

    print(f"Training rows: {train_df.count()}")
    print(f"Test rows: {test_df.count()}")

    return train_df, test_df


def train_hotspot_model(train_df, test_df):
    """
    使用 RandomForestClassifier 预测 is_hotspot (0/1)
    特征：历史统计 + 月份 + District 编码
    """
    label_col = "is_hotspot"

    numeric_cols = [
        "prev_1m_crime_count",
        "prev_3m_crime_avg",
        "prev_1m_violent_count",
        "prev_3m_violent_avg",
        "long_term_avg_crime",
        "avg_risk_score",
        "avg_location_risk",
        "crime_month",
    ]

    # District 作为类别特征
    dist_indexer = StringIndexer(
        inputCol="District",
        outputCol="district_idx",
        handleInvalid="keep",
    )

    assembler = VectorAssembler(
        inputCols=numeric_cols + ["district_idx"],
        outputCol="features",
    )

    rf = RandomForestClassifier(
        labelCol=label_col,
        featuresCol="features",
        numTrees=50,
        maxDepth=8,
        maxBins=64,
        featureSubsetStrategy="sqrt",
        subsamplingRate=0.7,
        seed=42,
    )

    pipeline = Pipeline(stages=[dist_indexer, assembler, rf])

    print("Training RandomForest hotspot model...")
    model = pipeline.fit(train_df)
    print("Model training complete.")

    print("Making predictions on test set...")
    predictions = model.transform(test_df)

    # 评估指标
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="accuracy",
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="f1",
    )

    auc = evaluator_auc.evaluate(predictions)
    accuracy = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)

    print("\n" + "=" * 50)
    print("HOTSPOT MODEL PERFORMANCE")
    print("=" * 50)
    print(f"AUC-ROC : {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("=" * 50 + "\n")

    print("Hotspot confusion (prediction vs label):")
    predictions.crosstab("prediction", label_col).show()

    return model, predictions, {"auc": auc, "accuracy": accuracy, "f1": f1}


def save_predictions(predictions):
    """
    将 test 集上的 hotspot 预测结果写回 HDFS
    """
    cols = [
        "crime_year",
        "crime_month",
        "District",
        "crime_count",
        "violent_count",
        "is_hotspot",
        "prediction",
        "probability",
    ]
    existing = [c for c in cols if c in predictions.columns]

    (
        predictions.select(*existing)
        .write.mode("overwrite")
        .parquet(RESULTS_PATH)
    )

    print(f"Hotspot predictions saved to {RESULTS_PATH}")


def save_metrics_to_mongo(model, metrics, train_df, test_df, predictions):
    """
    和你原来的逮捕模型一样，把指标 + 特征重要性写入 MongoDB
    """
    rf_model = model.stages[-1]
    importances = rf_model.featureImportances

    # 取出 features 中每一维的名字
    attrs = predictions.schema["features"].metadata["ml_attr"]["attrs"]
    idx2name = {}
    for attr_group in attrs.values():
        for attr in attr_group:
            idx2name[attr["idx"]] = attr["name"]

    feature_importance = [
        {"feature": idx2name[i], "importance": float(importances[i])}
        for i in range(len(importances))
        if float(importances[i]) > 0
    ]

    results_doc = {
        "model_type": "RandomForestClassifier",
        "task_type": "monthly_hotspot_prediction",
        "created_at": datetime.utcnow().isoformat(),
        "metrics": {
            "auc": float(metrics["auc"]),
            "accuracy": float(metrics["accuracy"]),
            "f1": float(metrics["f1"]),
        },
        "training_records": train_df.count(),
        "test_records": test_df.count(),
        "feature_importance": feature_importance,
    }

    mongo_uri = "mongodb://admin:admin123@mongodb:27017/?authSource=admin"
    client = pymongo.MongoClient(mongo_uri)
    db = client["crime_analysis"]
    collection = db["ml_results"]
    collection.insert_one(results_doc)
    client.close()
    print("Hotspot model results saved to MongoDB (crime_analysis.ml_results)")


def main():
    spark = build_spark()

    print("=== Loading processed incident-level data ===")
    df = load_processed(spark)
    print(f"Incident rows: {df.count()}")

    print("=== Building monthly district-level aggregates ===")
    monthly = build_monthly_aggregates(df)
    print(f"Monthly rows (year, month, District): {monthly.count()}")

    print("=== Adding temporal features & hotspot labels ===")
    monthly_feat = add_temporal_features(monthly)

    print("=== Train / Test split (time-based) ===")
    train_df, test_df = time_based_train_test_split(monthly_feat)

    print("=== Train hotspot prediction model ===")
    model, predictions, metrics = train_hotspot_model(train_df, test_df)

    print("=== Save predictions to HDFS ===")
    save_predictions(predictions)

    print("=== Save metrics & feature importance to MongoDB ===")
    save_metrics_to_mongo(model, metrics, train_df, test_df, predictions)

    spark.stop()
    print("Hotspot ML pipeline complete.")


if __name__ == "__main__":
    main()
