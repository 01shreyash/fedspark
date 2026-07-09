import json

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, from_json, schema_of_json, udf
from pyspark.sql.types import StringType, StructType


def create_kafka_stream(
    spark: SparkSession,
    topic: str,
    bootstrap_servers: str = "redpanda:9092",
    starting_offset: str = "earliest",
    fail_on_data_loss: bool = False,
) -> DataFrame:
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", starting_offset)
        .option("failOnDataLoss", fail_on_data_loss)
        .load()
    )


def parse_kafka_json(df: DataFrame, schema: StructType) -> DataFrame:
    return df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")


def write_lakehouse(
    df: DataFrame,
    output_path: str,
    checkpoint_path: str,
    format: str = "parquet",
    trigger_interval: str = "10 seconds",
    foreach_batch_fn=None,
):
    writer = (
        df.writeStream.format(format)
        .option("path", output_path)
        .option("checkpointLocation", checkpoint_path)
        .trigger(processingTime=trigger_interval)
    )
    if foreach_batch_fn:
        writer = writer.foreachBatch(foreach_batch_fn)
    return writer.start()
