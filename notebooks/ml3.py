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
    #    Features: 时间 + 地点 + 条件（不使用由 Primary Type 推出来的列）
    # -------------------------------------------------
    ml_df = df.select(
        "ID",
        "crime_datetime",
        "District",
        "Community Area",
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

    # 丢掉缺失值
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

    # -------------------------------------------------
    # 3. 类别特征编码 + Label 编码
    # -------------------------------------------------
    MAX_BINS = 128

    # Label：crime_category -> label
    label_indexer = StringIndexer(
        inputCol="crime_category",
        outputCol="label",
        handleInvalid="skip",  # 跳过没有 label 的
    )

    # 候选类别特征
    all_cat_cols_info = [
        ("time_of_day", "time_of_day_idx"),
        ("season", "season_idx"),
        ("District", "district_idx"),
        ("Community Area", "community_area_idx"),
        ("location_category", "location_category_idx"),
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

    # 数值特征
    numeric_cols = [
        "crime_hour",
        "crime_day_of_week",
        "crime_month",
        "location_risk_weight",
        "Latitude",
        "Longitude",
        "is_night",
        "is_domestic_incident",
        "is_public_space",
    ]
    
    ml_df = ml_df.na.fill(0.0, subset=numeric_cols)

    idx_cols = [out for (_, out) in cat_cols_info]

    assembler = VectorAssembler(
        inputCols=numeric_cols + idx_cols,
        outputCol="features",
        handleInvalid="skip",  # 如果仍有 NaN/null，直接丢掉那一行
    )

    # -------------------------------------------------
    # 4. Train/Test 划分（随机 80/20）
    #    如果你想做按年份的时间划分，可以改成 filter Year <= cutoff
    # -------------------------------------------------
    train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
    print(f"Training set: {train_df.count()} records")
    print(f"Test set: {test_df.count()} records")

    # -------------------------------------------------
    # 5. 定义 RF 多分类模型 + Pipeline
    # -------------------------------------------------
    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=60,
        maxDepth=10,
        maxBins=MAX_BINS,
        featureSubsetStrategy="sqrt",
        subsamplingRate=0.7,
        seed=42,
    )

    pipeline = Pipeline(stages=[label_indexer] + indexers + [assembler, rf])

    print("Training Random Forest (crime type prediction)...")
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

    # label 映射（index -> crime_category 名字），方便以后解码 probability
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
    }

    mongo_uri = "mongodb://admin:admin123@mongodb:27017/?authSource=admin"
    client = pymongo.MongoClient(mongo_uri)
    db = client["crime_analysis"]
    collection = db["ml_results"]
    collection.insert_one(results_doc)
    client.close()
    print("Crime-type model results saved to MongoDB (crime_analysis.ml_results)")

    # -------------------------------------------------
    # 9. 为预测结果加上可读的预测类别（predicted_crime_category）
    # -------------------------------------------------
    # prediction 列是 label index，需要映射回字符串
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

    # -------------------------------------------------
    # 10. 示例：给出 Top-3 候选类别及其概率（仅打印示例用）
    # -------------------------------------------------
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
    # 11. 把完整预测结果写回 HDFS（后面可以做可视化）
    # -------------------------------------------------
    print("\nSaving full predictions to HDFS...")

    output_cols = [
        "ID",
        "crime_datetime",
        "District",
        "Community Area",
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
