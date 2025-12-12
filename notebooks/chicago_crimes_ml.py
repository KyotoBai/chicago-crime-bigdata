"""
STEP 4: Machine Learning - Predict Crime Arrest Likelihood
Using Spark MLlib for distributed ML

输入：
  - HDFS: hdfs://namenode:9000/data/processed/chicago_crimes_clean.parquet
    列包括：
    ['ID', 'Case Number', 'Date', 'Primary Type', 'Description',
     'Location Description', 'Arrest', 'Domestic', 'Beat', 'District',
     'Ward', 'Community Area', 'Latitude', 'Longitude', 'crime_datetime',
     'crime_year', 'crime_month', 'crime_day', 'crime_hour',
     'crime_day_of_week', 'time_of_day', 'season', 'location_category',
     'location_risk_weight', 'crime_category', 'crime_severity_level',
     'crime_severity_weight', 'is_violent', 'is_night',
     'was_arrest_made', 'is_domestic_incident', 'is_public_space',
     'risk_score', 'Year']

训练目标（label）：
  - was_arrest_made（是否发生逮捕，布尔 → 0/1）

特征（features）：
  - 数值：crime_hour, crime_day_of_week, crime_month,
          location_risk_weight, crime_severity_weight, risk_score,
          Latitude, Longitude,
          is_violent, is_night, is_domestic_incident, is_public_space
  - 类别（用 index 编码）：Primary Type, Location Description,
          crime_severity_level, location_category, crime_category

输出：
  - HDFS: hdfs://namenode:9000/data/results/arrest_predictions_rf.parquet
    包含：ID, crime_datetime, Primary Type, Location Description, District,
          crime_hour, was_arrest_made(真实), prediction(预测), probability(概率)
  - MongoDB: crime_analysis.ml_results
    存模型指标（AUC, Accuracy, F1）和特征重要性
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml import Pipeline
import pymongo
from datetime import datetime


def main():
    # -------------------------------------------------
    # 1. 创建 SparkSession（连到 spark-master）
    # -------------------------------------------------
    spark = (
        SparkSession.builder.appName("ChicagoCrimeML")
        .master("spark://spark-master:7077")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print("Spark Session Created - Starting ML Pipeline")

    # -------------------------------------------------
    # 2. 从 HDFS 读取处理好的 parquet
    # -------------------------------------------------
    processed_path = "hdfs://namenode:9000/data/processed/chicago_crimes_clean.parquet"
    df = spark.read.parquet(processed_path)
    total_count = df.count()
    print(f"Loaded {total_count} records from {processed_path}")

    print("Columns in raw df:")
    print(df.columns)

    # -------------------------------------------------
    # 3. 选择需要的列（特征 + label）
    # -------------------------------------------------
    ml_df = df.select(
        "was_arrest_made",         # Label（是否逮捕）
        "crime_hour",
        "crime_day_of_week",
        "crime_month",
        "location_risk_weight",
        "crime_severity_weight",
        "risk_score",
        "Latitude",
        "Longitude",
        "is_violent",
        "is_night",
        "is_domestic_incident",
        "is_public_space",
        "Primary Type",
        "Location Description",
        "crime_severity_level",
        "location_category",
        "crime_category",
    )

    # 丢掉缺失值，避免后续出错
    ml_df = ml_df.na.drop()

    # 布尔列转 double (0/1)，方便作为数值特征
    bool_cols = [
        "is_violent",
        "is_night",
        "is_domestic_incident",
        "is_public_space",
        "was_arrest_made",
    ]
    for c in bool_cols:
        if c in ml_df.columns:
            ml_df = ml_df.withColumn(c, F.col(c).cast("double"))

    # -------------------------------------------------
    # 4. 类别特征编码：只用 StringIndexer（不做 OneHot）
    #    ⭐ 自动跳过 distinct 个数 > maxBins 的高基数类别列
    # -------------------------------------------------
    MAX_BINS = 128

    # 所有候选的类别列
    all_cat_cols_info = [
        ("Primary Type", "primary_type_idx"),
        ("Location Description", "location_description_idx"),
        ("crime_severity_level", "crime_severity_level_idx"),
        ("location_category", "location_category_idx"),
        ("crime_category", "crime_category_idx"),
    ]

    cat_cols_info = []
    print("\n[Check categorical cardinality]")
    for input_col, output_col in all_cat_cols_info:
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
            handleInvalid="keep",  # 测试集中若出现新类别会被归为一个特殊类
        )
        indexers.append(indexer)

    # 数值特征列
    numeric_cols = [
        "crime_hour",
        "crime_day_of_week",
        "crime_month",
        "location_risk_weight",
        "crime_severity_weight",
        "risk_score",
        "Latitude",
        "Longitude",
        "is_violent",
        "is_night",
        "is_domestic_incident",
        "is_public_space",
    ]

    # 类别 index 列
    idx_cols = [out for (_, out) in cat_cols_info]

    # 把所有特征拼成一个 features 向量
    assembler = VectorAssembler(
        inputCols=numeric_cols + idx_cols,
        outputCol="features",
    )

    label_col = "was_arrest_made"

    # -------------------------------------------------
    # 5. Train/Test 划分
    # -------------------------------------------------
    train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
    print(f"Training set: {train_df.count()} records")
    print(f"Test set: {test_df.count()} records")
    
    # train_df = train_df.sample(withReplacement=False, fraction=0.3, seed=42)
    # print(f"Downsampled training set: {train_df.count()} records")

    # -------------------------------------------------
    # 6. 定义 RandomForest 模型 + Pipeline
    #    ⭐ 这里加上 maxBins=256，解决 maxBins 太小的问题
    # -------------------------------------------------
    rf = RandomForestClassifier(
        labelCol=label_col,
        featuresCol="features",
        numTrees=30,              # 可以先用 50 棵树，内存压力小一点
        maxDepth=8,               # 深度也稍微收一收
        maxBins=64,         # 和上面检查保持一致
        featureSubsetStrategy="sqrt",
        subsamplingRate=0.7,
        seed=42,
    )
    MAX_BINS = 64
    
    pipeline = Pipeline(stages=indexers + [assembler, rf])

    print("Training Random Forest...")
    model = pipeline.fit(train_df)
    print("Model training complete.")

    # 在测试集上做预测
    print("Making predictions on test set...")
    predictions = model.transform(test_df)

    # -------------------------------------------------
    # 7. 评估指标：AUC / Accuracy / F1
    # -------------------------------------------------
    print("Evaluating model performance...")

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
    print("MODEL PERFORMANCE")
    print("=" * 50)
    print(f"AUC-ROC : {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("=" * 50 + "\n")

    # -------------------------------------------------
    # 8. 特征重要性
    # -------------------------------------------------
    print("Computing feature importance...")

    rf_model = model.stages[-1]  # Pipeline 最后一个 stage 是 RF
    importances = rf_model.featureImportances

    # 从 metadata 里拿到 features 每一维的名字
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

    importance_df.show(10)

    # -------------------------------------------------
    # 9. 把指标和特征重要性写入 MongoDB
    # -------------------------------------------------
    print("Saving metrics to MongoDB...")

    importance_list = [
        {"feature": row["feature"], "importance": float(row["importance"])}
        for row in importance_df.collect()
    ]

    results_doc = {
        "model_type": "RandomForestClassifier",
        "created_at": datetime.utcnow().isoformat(),
        "metrics": {
            "auc": float(auc),
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
    print("Results saved to MongoDB (crime_analysis.ml_results)")

    # -------------------------------------------------
    # 10. 把预测结果写回 HDFS
    # -------------------------------------------------
    print("Saving predictions to HDFS...")

    RESULTS_PATH = "hdfs://namenode:9000/data/results/arrest_predictions_rf.parquet"

    output_cols = [
        "ID",
        "crime_datetime",
        "Primary Type",
        "Location Description",
        "District",
        "crime_hour",
        "was_arrest_made",
        "prediction",
        "probability",
    ]
    existing_cols = [c for c in output_cols if c in predictions.columns]

    (
        predictions.select(*existing_cols)
        .write.mode("overwrite")
        .parquet(RESULTS_PATH)
    )

    print(f"Predictions saved to {RESULTS_PATH}")

    # -------------------------------------------------
    # 11. 打印几条样例预测
    # -------------------------------------------------
    print("\nSample predictions:")
    predictions.select(
        "was_arrest_made",
        "prediction",
        "probability",
        "Primary Type",
        "Location Description",
        "crime_hour",
    ).show(20, truncate=False)

    spark.stop()
    print("\nML Pipeline complete!")


if __name__ == "__main__":
    main()
