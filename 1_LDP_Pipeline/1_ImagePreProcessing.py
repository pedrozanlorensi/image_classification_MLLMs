# Databricks notebook source
from PIL import Image
import io
import dlt
import base64
import pyspark.sql.functions as F
from pyspark.sql.types import BinaryType
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import udf

# COMMAND ----------

resize_dimension = (256, 256)

# COMMAND ----------

# Function to resize the image
def resize_image(image_bytes, resize_dimension):
    # Open the image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    # Resize the image
    resized_image = image.resize(resize_dimension)

    # Save resized image into a bytes buffer
    img_byte_arr = io.BytesIO()
    resized_image.save(img_byte_arr, format='JPEG') # format it as jpg to match original type
    img_byte_arr.seek(0)  # Reset pointer to the start of the buffer
    
    return img_byte_arr.read()  # Return as bytes

# Register the function as a UDF and parameterize it to resize images
@udf(BinaryType())  # The return type is Binary (bytes)
def resize_image_udf(image_bytes):
    return resize_image(image_bytes, resize_dimension)  # Resize to resize_dimension

# COMMAND ----------

# Create the silver dataset
@dlt.table(
    name="pedroz_genai_catalog.default.silver_images_preprocessed",
    comment="Streaming table of images with metadata and base64 encoding"
)
def silver_images_preprocessed():
    # Stream images in binary format
    df = spark.readStream.format("cloudFiles")   \
        .option("cloudFiles.format", "binaryFile") \
        .option("pathGlobFilter", "*.jp*g") \
        .load("/Volumes/pedroz_genai_catalog/default/bronze_images/*")

    # Extract class (elephant/horse/cat)
    df = df.withColumn(
        "class",
        F.regexp_extract("path", "./(elephant|horse|cat)/.*", 1)
    )

    # Save image binary content into column img
    df = df.withColumn("img", F.col("content"))

    # Convert it to base64 in column img_b64
    df = df.withColumn("img_b64", F.base64(F.col("content")))

    # Save resized image (256x256) in column img_resized
    df = df.withColumn("img_resized", resize_image_udf(F.col("img")))

    # Convert the resized image to base64 in column img_resized_b64
    df = df.withColumn("img_resized_b64", F.base64(F.col("img_resized")))

    return df.select(
        "path",
        "class",
        "img",
        "img_b64",
        "img_resized",
        "img_resized_b64"
    )