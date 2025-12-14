from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, ArrayType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array

import pymongo
from datetime import datetime

PROCESSED_PATH = "hdfs://namenode:9000/data/processed/chicago_crimes_clean.parquet"
RESULTS_PATH = "hdfs://namenode:9000/data/results/crime_type_predictions_rf.parquet"


def build_spark():
    spark = (
        SparkSession.builder
        .appName("ChicagoCrimeTypeML")
        .master("spark://spark-master:7077")
        # 减少 shuffle task 的内存压力
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_processed(spark):
    df = spark.read.parquet(PROCESSED_PATH)
    return df


def main():
    spark = build_spark()

    # -------------------------------------------------
    # 1. 读取处理好的数据
    # -------------------------------------------------
    df = load_processed(spark)
    total_count = df.count()
    print(f"Loaded {total_count} records from {PROCESSED_PATH}")

    print("Columns in processed df:")
    print(df.columns)

    # -------------------------------------------------
    # 2. 选择本任务需要的列
    #    Label: crime_category（多分类）
    #    特征：时间 + 地点 + 条件（简化后的版本）
    # -------------------------------------------------
    ml_df = df.select(
        "ID",
        "crime_datetime",
        # 把 District / Community Area 直接 cast 成数值列，避免作为大类别特征
        F.col("District").cast(DoubleType()).alias("District"),
        F.col("Community Area").cast(DoubleType()).alias("Community_Area"),
        "Location Description",
        "location_category",
        "crime_category",          # label
        "crime_hour",
        "crime_day_of_week",
        "crime_month",
        "time_of_day",
        "season",
        "location_risk_weight",
        "Latitude",
        "Longitude",
        "is_night",
        "is_domestic_incident",
        "is_public_space",
    )

    # 基础必需列不能是 null
    ml_df = ml_df.na.drop(
        subset=[
            "crime_category",
            "crime_hour",
            "crime_day_of_week",
            "crime_month",
            "District",
            "location_category",
        ]
    )

    # 布尔列转 double
    bool_cols = [
        "is_night",
        "is_domestic_incident",
        "is_public_space",
    ]
    for c in bool_cols:
        if c in ml_df.columns:
            ml_df = ml_df.withColumn(c, F.col(c).cast(DoubleType()))

    # 数值特征列（包含 District / Community_Area）
    numeric_cols = [
        "crime_hour",
        "crime_day_of_week",
        "crime_month",
        "location_risk_weight",
        "Latitude",
        "Longitude",
        "District",
        "Community_Area",
        "is_night",
        "is_domestic_incident",
        "is_public_space",
    ]
    # 数值列中的 null 全部填成 0，避免 VectorAssembler 报错
    ml_df = ml_df.na.fill(0.0, subset=numeric_cols)

    # -------------------------------------------------
    # ⭐ 关键降内存手段 1：对整体数据做采样
    # -------------------------------------------------
    SAMPLE_FRACTION = 1.0  # 可根据内存再调小一点，比如 0.3
    sampled_total = int(total_count * SAMPLE_FRACTION)

    ml_df = ml_df.sample(withReplacement=False, fraction=SAMPLE_FRACTION, seed=42)
    print(f"Using a {SAMPLE_FRACTION*100:.1f}% sample for crime-type model:")
    print(f"Sampled records: {ml_df.count()} (out of {total_count})")

    # -------------------------------------------------
    # 3. 类别特征编码 + Label 编码
    #    只保留少量类别特征，避免高基数 categorical split
    # -------------------------------------------------
    MAX_BINS = 64

    # Label：crime_category -> label（多分类）
    label_indexer = StringIndexer(
        inputCol="crime_category",
        outputCol="label",
        handleInvalid="skip",
    )

    # 仅保留少量低基数类别特征
    all_cat_cols_info = [
        ("time_of_day", "time_of_day_idx"),
        ("season", "season_idx"),
        ("location_category", "location_category_idx"),
        # 不再对 District / Community_Area 做 StringIndexer
    ]

    cat_cols_info = []
    print("\n[Check categorical cardinality]")
    for input_col, output_col in all_cat_cols_info:
        if input_col not in ml_df.columns:
            continue
        distinct_cnt = ml_df.select(input_col).distinct().count()
        print(f"  {input_col}: distinct = {distinct_cnt}")
        if distinct_cnt <= MAX_BINS:
            cat_cols_info.append((input_col, output_col))
        else:
            print(
                f"  -> Skip {input_col} because distinct {distinct_cnt} > MAX_BINS ({MAX_BINS})"
            )

    print("\n[Will use categorical columns]:", [c for c, _ in cat_cols_info])

    indexers = []
    for input_col, output_col in cat_cols_info:
        indexer = StringIndexer(
            inputCol=input_col,
            outputCol=output_col,
            handleInvalid="keep",
        )
        indexers.append(indexer)

    idx_cols = [out for (_, out) in cat_cols_info]

    assembler = VectorAssembler(
        inputCols=numeric_cols + idx_cols,
        outputCol="features",
        handleInvalid="skip",
    )

    # -------------------------------------------------
    # 4. Train/Test 划分（在采样后数据上 80/20）
    # -------------------------------------------------
    train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
    print(f"Training set: {train_df.count()} records")
    print(f"Test set: {test_df.count()} records")

    # -------------------------------------------------
    # 5. 定义 RF 多分类模型（降内存版本）
    # -------------------------------------------------
    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=30,          # 从 60 降到 30
        maxDepth=8,           # 从 10 降到 8
        maxBins=MAX_BINS,     # 64
        featureSubsetStrategy="sqrt",
        subsamplingRate=0.5,  # 每棵树只用 50% 样本
        seed=42,
    )

    pipeline = Pipeline(stages=[label_indexer] + indexers + [assembler, rf])

    print("Training Random Forest (crime type prediction, light version)...")
    model = pipeline.fit(train_df)
    print("Model training complete.")

    # 在测试集上预测
    print("Making predictions on test set...")
    predictions = model.transform(test_df)

    # -------------------------------------------------
    # 6. 评估指标：Accuracy / F1（多分类）
    # -------------------------------------------------
    print("Evaluating model performance...")

    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy",
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1",
    )

    accuracy = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)

    print("\n" + "=" * 50)
    print("CRIME TYPE MODEL PERFORMANCE")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("=" * 50 + "\n")

    # -------------------------------------------------
    # 7. 特征重要性
    # -------------------------------------------------
    print("Computing feature importance...")

    rf_model = model.stages[-1]
    importances = rf_model.featureImportances

    attrs = predictions.schema["features"].metadata["ml_attr"]["attrs"]
    idx2name = {}
    for attr_group in attrs.values():
        for attr in attr_group:
            idx2name[attr["idx"]] = attr["name"]

    feature_importance = [
        (idx2name[i], float(importances[i]))
        for i in range(len(importances))
        if float(importances[i]) > 0
    ]

    importance_df = (
        spark.createDataFrame(feature_importance, ["feature", "importance"])
        .orderBy(F.desc("importance"))
    )

    importance_df.show(15, truncate=False)

    # -------------------------------------------------
    # 8. 把指标 & 特征重要性写入 MongoDB
    # -------------------------------------------------
    print("Saving metrics to MongoDB...")

    importance_list = [
        {"feature": row["feature"], "importance": float(row["importance"])}
        for row in importance_df.collect()
    ]

    # label 映射（index -> crime_category 名字）
    label_indexer_model = model.stages[0]
    label_classes = list(label_indexer_model.labels)

    results_doc = {
        "model_type": "RandomForestClassifier",
        "task_type": "crime_type_prediction",
        "label_col": "crime_category",
        "label_classes": label_classes,
        "created_at": datetime.utcnow().isoformat(),
        "metrics": {
            "accuracy": float(accuracy),
            "f1": float(f1),
        },
        "training_records": train_df.count(),
        "test_records": test_df.count(),
        "feature_importance": importance_list,
        "sample_fraction": SAMPLE_FRACTION,
    }

    mongo_uri = "mongodb://admin:admin123@mongodb:27017/?authSource=admin"
    client = pymongo.MongoClient(mongo_uri)
    db = client["crime_analysis"]
    collection = db["ml_results"]
    collection.insert_one(results_doc)
    client.close()
    print("Crime-type model results saved to MongoDB (crime_analysis.ml_results)")

    # -------------------------------------------------
    # 9. 预测结果：加上可读的预测类别 + Top-3 候选
    # -------------------------------------------------
    mapping_dict = {i: label for i, label in enumerate(label_classes)}

    def idx_to_label(idx):
        if idx is None:
            return None
        return mapping_dict.get(int(idx), None)

    idx_to_label_udf = F.udf(idx_to_label, StringType())

    predictions = predictions.withColumn(
        "predicted_crime_category",
        idx_to_label_udf(F.col("prediction")),
    )

    print("\nBuilding Top-3 candidate predictions for sample rows...")

    preds_with_array = predictions.withColumn(
        "prob_array", vector_to_array("probability")
    )

    def top3_labels(prob_list):
        if prob_list is None:
            return None
        pairs = list(enumerate(prob_list))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top3 = pairs[:3]
        return [f"{mapping_dict[i]}:{p:.3f}" for i, p in top3]

    top3_udf = F.udf(top3_labels, ArrayType(StringType()))

    sample_top3 = (
        preds_with_array
        .withColumn("top3_candidates", top3_udf("prob_array"))
        .select(
            "crime_datetime",
            "District",
            "location_category",
            "crime_hour",
            "crime_day_of_week",
            "crime_month",
            "crime_category",             # 真实
            "predicted_crime_category",   # 预测 Top-1
            "top3_candidates",            # Top-3 + 概率
        )
        .limit(20)
    )

    sample_top3.show(truncate=False)

    # -------------------------------------------------
    # 10. 把完整预测结果写回 HDFS
    # -------------------------------------------------
    print("\nSaving full predictions to HDFS...")

    output_cols = [
        "ID",
        "crime_datetime",
        "District",
        "Community_Area",
        "Location Description",
        "location_category",
        "crime_hour",
        "crime_day_of_week",
        "crime_month",
        "time_of_day",
        "season",
        "crime_category",            # 真实 label
        "predicted_crime_category",  # 预测 label
        "probability",               # 所有类别的概率分布
    ]
    existing_cols = [c for c in output_cols if c in predictions.columns]

    (
        predictions.select(*existing_cols)
        .write.mode("overwrite")
        .parquet(RESULTS_PATH)
    )

    print(f"Crime-type predictions saved to {RESULTS_PATH}")

    spark.stop()
    print("\nCrime type prediction pipeline complete!")


if __name__ == "__main__":
    main()
