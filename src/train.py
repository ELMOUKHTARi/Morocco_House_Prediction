import mlflow
import mlflow.sklearn

def train_model():
    print("🔹 Chargement des données...")
    df = load_data("../model/DataPOO.csv")

    print("🔹 Préprocessing...")
    df = preprocess_data(df)

    print("🔹 Split des données...")
    X, y = split_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10
    )

    print("🔹 Entraînement du modèle...")
    model = LinearRegression()

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
