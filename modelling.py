import pandas as pd
import mlflow
import mlflow.sklearn
import argparse

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Parse argument
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="namadataset_preprocessing/heart_clean.csv")
args = parser.parse_args()

# MLflow config
mlflow.set_experiment("CI-Heart-Disease")

# Load data
df = pd.read_csv(args.data_path)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

print("Training via CI selesai")
