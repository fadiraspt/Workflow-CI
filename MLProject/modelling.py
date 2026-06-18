import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Loan_Prediction_Basic")


mlflow.sklearn.autolog()

df = pd.read_csv("loan_preprocessing.csv")

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = RandomForestClassifier(
        random_state=42
    )

model.fit(
        X_train,
        y_train
    )

y_pred = model.predict(
        X_test
    )

accuracy = accuracy_score(
        y_test,
        y_pred
    )

print(
        f"Accuracy: {accuracy:.4f}"
    )

joblib.dump(
        model,
        "best_model.pkl"
    )

print("Training selesai")