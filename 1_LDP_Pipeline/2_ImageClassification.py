# Databricks notebook source
import dlt
from pyspark.sql.functions import expr, concat, lit, col, get_json_object

# COMMAND ----------

@dlt.table(
    name="pedroz_genai_catalog.default.gold_images_classified",
    comment="Gold table with the images classified"
)
def silver_images_classified():
    df = dlt.read_stream("pedroz_genai_catalog.default.silver_images_preprocessed")

    df = (
        df.withColumnRenamed("class", "actual_class")
    )

    # Zero-shot multi-class classification
    df = df.withColumn(
        "predicted_class",
        expr(
            f"""
            AI_QUERY(
                "databricks-llama-4-maverick",
                "What animal do you see in the following image: an elephant, a horse, or a cat?",
                files => img,
                responseFormat => '{{
                "type": "json_schema",
                "json_schema": {{
                    "name": "ImageClassification",
                    "schema": {{
                        "type": "object",
                        "properties": {{
                            "animal": {{
                                "type": "string",
                                "enum": ["elephant", "horse", "cat"]
                            }}
                        }},
                        "required": ["animal"],
                        "additionalProperties": false
                    }},
                    "strict": true
                }}
            }}'
            )
            """
        )
    )

    return df.select(
        "path",
        "actual_class",
        "predicted_class"
    )