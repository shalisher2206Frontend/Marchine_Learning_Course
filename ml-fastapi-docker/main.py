from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Iris ML API")

# Load the trained model 
model = joblib.load("model.joblib")
class_names = ["setosa", "versicolor", "virginica"]


# Input features expected from the user
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# 1) GET / : confirms the API is running
@app.get("/")
def root():
    return {"message": "ML API is running"}


# 2) POST /predict : accepts features, generates a prediction, returns the result
@app.post("/predict")
def predict(features: IrisFeatures):
    x = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]
    prediction = int(model.predict(x)[0])
    return {
        "prediction": prediction,
        "species": class_names[prediction],
    }