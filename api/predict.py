from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# Train lightweight model once at startup
iris = load_iris()
X, y = iris.data, iris.target
target_names = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/", response_class=HTMLResponse)
def serve_home():
    html_path = Path(__file__).resolve().parent.parent / "index.html"
    return html_path.read_text(encoding="utf-8")

@app.post("/api/predict")
def predict_species(data: IrisFeatures):
    try:
        features = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]])

        pred_idx = int(clf.predict(features)[0])
        probs = clf.predict_proba(features)[0]

        return {
            "species": target_names[pred_idx],
            "confidence": round(float(probs[pred_idx]) * 100, 2),
            "probabilities": {
                target_names[i]: round(float(probs[i]) * 100, 2)
                for i in range(len(target_names))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))