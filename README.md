# 🚀 Customer Churn Prediction

A Machine Learning project that predicts whether a customer is likely to churn using the **XGBoost** algorithm. The project includes model training, experiment tracking with **MLflow**, a **FastAPI** REST API for predictions, and **Docker** support for deployment.

---

## 📌 Features

* Customer churn prediction using XGBoost
* Data preprocessing and categorical encoding
* Model evaluation with accuracy and confusion matrix
* MLflow experiment tracking
* FastAPI REST API for real-time predictions
* Docker support for containerized deployment
* Trained model saved using Joblib

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* FastAPI
* MLflow
* Joblib
* Docker
* Git & GitHub

---

## 📂 Project Structure

```text
CUSTOMER_CHURN_PREDICTION/
│── app.py                 # FastAPI application
│── train_model.py         # Model training script
│── requirements.txt       # Project dependencies
│── Dockerfile             # Docker configuration
│── dataset.csv            # Customer churn dataset
│── churn_model.joblib     # Trained model
│── mlflow/                # MLflow experiments and artifacts
│── mlflow.db              # MLflow database
│── .gitignore
│── README.md
```

---

## 📊 Dataset

The dataset contains customer-related information such as:

* Customer ID
* Age
* Gender
* Tenure
* Usage Frequency
* Support Calls
* Payment Delay
* Subscription Type
* Contract Length
* Total Spend
* Last Interaction
* Churn (Target)

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Nidhi6620/CUSTOMER_CHURN_PREDICTION.git
cd CUSTOMER_CHURN_PREDICTION
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it (Windows):

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Model

```bash
python train_model.py
```

---

## 🌐 Run the API

```bash
uvicorn app:app --reload
```

After the server starts, open:

* Swagger UI: `http://127.0.0.1:8000/docs`
* API Root: `http://127.0.0.1:8000`

---

## 📈 MLflow

Start the MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

MLflow tracks:

* Model parameters
* Metrics
* Accuracy
* Confusion Matrix
* Saved model artifacts

---

## 🐳 Docker

Build the Docker image:

```bash
docker build -t customer-churn .
```

Run the container:

```bash
docker run -p 8000:8000 customer-churn
```

---

## 📈 Model Workflow

1. Load the dataset
2. Clean and preprocess data
3. Encode categorical features
4. Split into training and testing sets
5. Train an XGBoost classifier
6. Evaluate performance
7. Log experiments with MLflow
8. Save the trained model
9. Serve predictions using FastAPI

---

## 🔮 Future Improvements

* Hyperparameter tuning
* Cloud deployment (AWS/Azure)
* CI/CD pipeline
* Streamlit dashboard
* Model monitoring and retraining

---

## 👨‍💻 Author
**Mithila S**

**Computer Science Engineering (Data Science)**

This project was developed as part of a machine learning learning and deployment workflow using FastAPI, MLflow, Docker, and XGBoost.
