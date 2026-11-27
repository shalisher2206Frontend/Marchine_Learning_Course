import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Где MLflow хранит данные + название эксперимента
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-experiment")

# Данные
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Гиперпараметры
params = {"n_estimators": 100, "max_depth": 5}

with mlflow.start_run():
    # Обучение
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    # Оценка
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    # 1. Логируем параметры
    mlflow.log_params(params)

    # 2. Логируем метрики
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # 3. Сохраняем модель для FastAPI и логируем её как артефакт
    joblib.dump(model, "model.joblib")
    mlflow.log_artifact("model.joblib")

    # 4. Регистрируем модель в Model Registry (имя + версия)
    mlflow.sklearn.log_model(
        model, "model", registered_model_name="iris-classifier"
    )

    print(f"accuracy={accuracy:.3f}, f1={f1:.3f}")