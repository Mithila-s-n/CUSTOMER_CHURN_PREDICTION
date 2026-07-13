import pandas as pd
import joblib
# pyrefly: ignore [missing-import]
import mlflow
# pyrefly: ignore [missing-import]
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("dataset.csv", sep="\t")

print("Dataset Loaded Successfully!")

# ==========================
# Data Preprocessing
# ==========================

# Drop CustomerID
df.drop("CustomerID", axis=1, inplace=True)

# Encode Categorical Columns
encoder = LabelEncoder()

categorical_columns = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Started...")

# ==========================
# MLflow Experiment
# ==========================

mlflow.set_experiment("Customer Churn Prediction")

with mlflow.start_run():

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    # Train Model
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy :", accuracy)

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    # Log Parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 5)

    # Log Metric
    mlflow.log_metric("accuracy", accuracy)

    # Save Model
    joblib.dump(model, "churn_model.joblib")
    # Log XGBoost Model to MLflow
    try:
        mlflow.xgboost.log_model(model, name="model")
        print("Model Logged to MLflow Successfully!")
    except Exception as e:
        print("MLflow Model Logging Skipped!")
        print(e)

print("\nModel Saved Successfully!")
print("Project Completed Successfully!")