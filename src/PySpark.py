
# //////////////////////////////////////////////////////////////////////////
# ***** Regression and Classification using PySpark MLlib pipelines *****
# src/PySpark.py
# PySpark regression & classification with CLI args, numeric-only features, JSON metrics out

import argparse, json, time, os
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression as SparkLinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator

NUMERIC_TYPES = {"double", "float", "int", "bigint", "smallint", "tinyint"}

def numeric_columns(sdf):
    return [name for name, dtype in sdf.dtypes if dtype in NUMERIC_TYPES]

def ensure_rating_binary(sdf):
    # Prefer rating_text_ord (>=3 -> 1), else median split on rating_number
    if "rating_binary" in sdf.columns:
        return sdf
    if "rating_text_ord" in sdf.columns:
        return sdf.withColumn("rating_binary", (F.col("rating_text_ord") >= F.lit(3)).cast("int"))
    if "rating_number" in sdf.columns:
        # median via approxQuantile
        med = sdf.approxQuantile("rating_number", [0.5], 0.01)[0]
        return sdf.withColumn("rating_binary", (F.col("rating_number") >= F.lit(med)).cast("int"))
    # If neither exists, create nulls (stage will no-op for classification)
    return sdf.withColumn("rating_binary", F.lit(None).cast("int"))

def run_regression(sdf):
    out = {"mse": None, "r2": None, "train_time_s": None, "infer_time_s": None}
    if "rating_number" not in sdf.columns:
        return out

    num_cols = numeric_columns(sdf)
    feature_cols = [c for c in num_cols if c != "rating_number"]
    if not feature_cols:
        return out

    sdf_reg = sdf.select(*feature_cols, "rating_number").dropna(subset=["rating_number"])
    if sdf_reg.count() < 5:
        return out

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler    = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    lr        = SparkLinearRegression(featuresCol="features", labelCol="rating_number", predictionCol="prediction")
    pipe      = Pipeline(stages=[assembler, scaler, lr])

    train_reg, test_reg = sdf_reg.randomSplit([0.8, 0.2], seed=42)

    t0 = time.perf_counter()
    model = pipe.fit(train_reg)
    out["train_time_s"] = round(time.perf_counter() - t0, 4)

    t1 = time.perf_counter()
    pred = model.transform(test_reg).cache()
    _ = pred.count()  # force materialization to time inference
    out["infer_time_s"] = round(time.perf_counter() - t1, 4)

    eval_mse = RegressionEvaluator(labelCol="rating_number", predictionCol="prediction", metricName="mse")
    eval_r2  = RegressionEvaluator(labelCol="rating_number", predictionCol="prediction", metricName="r2")
    out["mse"] = float(eval_mse.evaluate(pred))
    out["r2"]  = float(eval_r2.evaluate(pred))
    return out

def run_classification(sdf):
    out = {"accuracy": None, "train_time_s": None, "infer_time_s": None}
    if "rating_binary" not in sdf.columns:
        return out

    num_cols = numeric_columns(sdf)
    feature_cols = [c for c in num_cols if c not in {"rating_binary", "rating_number"}]
    if not feature_cols:
        return out

    sdf_clf = sdf.select(*feature_cols, "rating_binary").dropna(subset=["rating_binary"])
    if sdf_clf.count() < 5:
        return out

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler    = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    logreg    = LogisticRegression(featuresCol="features", labelCol="rating_binary", predictionCol="prediction",
                                   maxIter=100, regParam=0.1, elasticNetParam=0.0)
    pipe      = Pipeline(stages=[assembler, scaler, logreg])

    train_clf, test_clf = sdf_clf.randomSplit([0.8, 0.2], seed=42)

    t0 = time.perf_counter()
    model = pipe.fit(train_clf)
    out["train_time_s"] = round(time.perf_counter() - t0, 4)

    t1 = time.perf_counter()
    pred = model.transform(test_clf).cache()
    _ = pred.count()
    out["infer_time_s"] = round(time.perf_counter() - t1, 4)

    eval_acc = MulticlassClassificationEvaluator(labelCol="rating_binary", predictionCol="prediction", metricName="accuracy")
    out["accuracy"] = float(eval_acc.evaluate(pred))
    return out

def main(p_in, p_out):
    spark = SparkSession.builder.appName("uc-assignment-pyspark").getOrCreate()
    try:
        sdf = spark.read.parquet(p_in)

        # keep numeric columns only (VectorAssembler requirement)
        sdf = ensure_rating_binary(sdf)

        reg_metrics = run_regression(sdf)
        clf_metrics = run_classification(sdf)

        metrics = {"pyspark": {"regression": reg_metrics, "classification": clf_metrics}}
        os.makedirs(os.path.dirname(p_out), exist_ok=True)
        with open(p_out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[PySpark] metrics written to {p_out}")
    finally:
        spark.stop()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  required=True)
    ap.add_argument("--out", dest="outp", required=True)
    args = ap.parse_args()
    main(args.inp, args.outp)
