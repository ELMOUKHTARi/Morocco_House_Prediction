import pandas as pd
import pickle
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from preprocess import load_data, preprocess_data, split_data

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:./mlruns")   # stocke les runs dans ./mlruns
mlflow.set_experiment("house-price-prediction")


def train_model():
    print("🔹 Chargement des données...")
    
    df = load_data("model/DataPOO.csv")

    print("🔹 Préprocessing...")
    df = preprocess_data(df)

    print("🔹 Split des données...")
    X, y = split_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10
    )

    print("🔹 Entraînement du modèle...")
    model = LinearRegression()

    # Démarrer un run MLflow
    with mlflow.start_run():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        # Log des paramètres et métriques
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 10)
        mlflow.log_metric("r2_score", score)

        # Sauvegarde du modèle dans MLflow
        mlflow.sklearn.log_model(model, "model")

    print(f"✅ Accuracy (R² score) : {score}")

    # Créer dossier artifacts si n'existe pas
    os.makedirs("artifacts", exist_ok=True)

    print("🔹 Sauvegarde du modèle...")
    with open("artifacts/5abd_sana_model.pickle", "wb") as f:
        pickle.dump(model, f)

    print("🔹 Sauvegarde des colonnes...")
    columns = {
        'data_columns': [col.lower() for col in X.columns]
    }
    with open("artifacts/5columns_ma.json", "w") as f:
        json.dump(columns, f)

    print("🎉 Entraînement terminé avec succès !")


if __name__ == "__main__":
    train_model()
