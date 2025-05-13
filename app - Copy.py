from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.datasets import load_iris
import uvicorn
import os

# Load model and label names
model = joblib.load("iris_model.pkl")
iris = load_iris()

# Setup FastAPI and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Input schema for API
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Predict API
@app.post("/predict")
def predict_species(data: IrisInput):
    df = pd.DataFrame([{
        "sepal length (cm)": data.sepal_length,
        "sepal width (cm)": data.sepal_width,
        "petal length (cm)": data.petal_length,
        "petal width (cm)": data.petal_width
    }])
    
    # Prediction and probability
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    confidence = round(max(probabilities) * 100, 2)  # Confidence in percentage
    species = iris.target_names[prediction]

    # Log
    df["prediction"] = species
    df["confidence (%)"] = confidence
    df.to_csv("prediction_log.csv", mode="a", header=not os.path.exists("prediction_log.csv"), index=False)

    return {
        "species": species,
        "confidence (%)": confidence
    }


# HTML Form Endpoint
@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# Handle form submission
@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request,
                    sepal_length: float = Form(...),
                    sepal_width: float = Form(...),
                    petal_length: float = Form(...),
                    petal_width: float = Form(...)):

    input_df = pd.DataFrame([{
        "sepal length (cm)": float(sepal_length),
        "sepal width (cm)": float(sepal_width),
        "petal length (cm)": float(petal_length),
        "petal width (cm)": float(petal_width)
    }])

    prediction = model.predict(input_df)[0]
    species = iris.target_names[prediction]

    # Log to CSV
    input_df["prediction"] = species
    input_df.to_csv("prediction_log.csv", mode="a", header=not os.path.exists("prediction_log.csv"), index=False)

    return templates.TemplateResponse("form.html", {
        "request": request,
        "result": species
    })

# Run locally
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
