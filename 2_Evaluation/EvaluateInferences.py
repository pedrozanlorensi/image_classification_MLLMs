# Databricks notebook source
import pyspark.pandas as ps
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
)
import json

# COMMAND ----------

# Load table as Spark DataFrame
pdf_predictions = spark.read.table("pedroz_genai_catalog.default.gold_images_classified").toPandas()

# COMMAND ----------

pdf_predictions.head()

# COMMAND ----------

# Transform prediction columns to get only the predicted label

# List of columns to transform
cols = [
    "predicted_class"
]

# Apply JSON parsing to each column
for c in cols:
    pdf_predictions[c] = pdf_predictions[c].apply(
        lambda x: json.loads(x).get("animal") if pd.notnull(x) else None
    )

# COMMAND ----------

pdf_predictions.head()

# COMMAND ----------

# Multi-class classification
multi_counts = pdf_predictions['actual_class'].value_counts()
print("\nMulti-class counts:\n", multi_counts)
fig, axes = plt.subplots(1, 1, figsize=(12, 6))
# Plot class distributions
sns.barplot(x=multi_counts.index, y=multi_counts.values, ax=axes)
axes.set_title("Multi-Class Distribution")
axes.set_ylabel("# of images")
plt.show()


# COMMAND ----------

def evaluate_classification(actual, predicted, labels):
    metrics = {}
    metrics['accuracy'] = accuracy_score(actual, predicted)
    metrics['precision'] = precision_score(actual, predicted, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(actual, predicted, average='weighted', zero_division=0)
    metrics['f1_score'] = f1_score(actual, predicted, average='weighted', zero_division=0)
    
    cm = confusion_matrix(actual, predicted, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels) if labels else pd.DataFrame(cm)
    return metrics, cm_df


# COMMAND ----------

scenarios = [
    {
        "name": "Classification",
        "actual": "actual_class",
        "predicted": "predicted_class",
        "labels": ["elephant", "horse", "cat"]
    }
]


# COMMAND ----------

results = {}

for s in scenarios:
    name = s['name']
    metrics, cm_df = evaluate_classification(pdf_predictions[s['actual']], pdf_predictions[s['predicted']], labels=s['labels'])
    results[name] = {
        "metrics": metrics,
        "confusion_matrix": cm_df
    }
    
    print(f"\n===== {name} =====")
    print("Metrics:")
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm_df)
    
    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()


# COMMAND ----------

for s in scenarios:
    print(f"\n===== {s['name']} - Per-Class Metrics =====")
    print(classification_report(
        pdf_predictions[s['actual']], 
        pdf_predictions[s['predicted']], 
        labels=s['labels'],
        zero_division=0
    ))