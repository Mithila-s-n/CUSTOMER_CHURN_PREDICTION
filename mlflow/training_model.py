# ==============================
# Customer Churn Prediction
# XGBoost + MLflow
# ==============================


import pandas as pd
import numpy as np

import mlflow
# pyrefly: ignore [missing-import]
import mlflow.xgboost


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import joblib
import os


# ==============================
# Load Dataset
# ==============================

data = pd.read_csv(
    "../dataset.csv",
    sep="\t"
)


# Remove spaces in column names

data.columns = data.columns.str.strip()


print(data.columns)

print(data.head())


# ==============================
# Data Preprocessing
# ==============================


# Drop Customer ID

data.drop(
    "CustomerID",
    axis=1,
    inplace=True
)


# Encode categorical columns

encoder = LabelEncoder()


categorical_columns = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]


for col in categorical_columns:

    data[col] = encoder.fit_transform(
        data[col]
    )


# Split features and target

X = data.drop(
    "Churn",
    axis=1
)


y = data["Churn"]



# ==============================
# Train Test Split
# ==============================


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y

)



# ==============================
# MLflow Setup
# ==============================

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "Customer Churn Prediction Final"
)

# ==============================
# Model Training
# ==============================


with mlflow.start_run():


    model = XGBClassifier(

        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42

    )


    model.fit(

        X_train,
        y_train

    )


    # Prediction

    y_pred = model.predict(
        X_test
    )



    # ==============================
    # Evaluation
    # ==============================


    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    print(
        "Accuracy:",
        accuracy
    )


    print(
        "\nClassification Report:"
    )


    print(
        classification_report(
            y_test,
            y_pred
        )
    )



    # Confusion Matrix

    cm = confusion_matrix(

        y_test,
        y_pred

    )


    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )


    disp.plot()


    plt.title(
        "Customer Churn Confusion Matrix"
    )


    plt.savefig(
        "confusion_matrix.png"
    )


    plt.close()



    # ==============================
    # MLflow Logging
    # ==============================


    mlflow.log_param(
        "model",
        "XGBClassifier"
    )


    mlflow.log_param(
        "n_estimators",
        200
    )


    mlflow.log_param(
        "max_depth",
        6
    )


    mlflow.log_metric(
        "accuracy",
        accuracy
    )


    mlflow.log_artifact(
        "confusion_matrix.png"
    )



    # Log model

    mlflow.xgboost.log_model(
    model,
    name="model"
)



    # ==============================
    # Save Model
    # ==============================


    joblib.dump(

        model,

        "churn_model.joblib"

    )



    print(
        "\nModel Logged to MLflow Successfully!"
    )


    print(
        "Model Saved Successfully!"
    )