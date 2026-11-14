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
        raise HTTPException(status_code=400, detail=str(e))from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# Train lightweight Random Forest classifier once on startup
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

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Iris Botanical Species Classifier | Patience Korbla</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --text: #f8fafc;
      --muted: #94a3b8;
      --border: #334155;
      --success: #10b981;
    }
    * { box-sizing: border-box; font-family: 'Segoe UI', system-ui, sans-serif; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      padding: 2rem 1rem;
      display: flex;
      justify-content: center;
    }
    .container { max-width: 900px; width: 100%; }
    header { text-align: center; margin-bottom: 2rem; }
    header h1 { margin: 0 0 0.5rem; font-size: 1.9rem; color: var(--accent); }
    header p { margin: 0; color: var(--muted); }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
    }
    @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
    }
    .card h2 { margin-top: 0; font-size: 1.2rem; border-bottom: 1px solid var(--border); padding-bottom: 0.75rem; }
    .presets { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1.25rem; }
    .preset-btn {
      background: #0f172a;
      color: var(--accent);
      border: 1px solid var(--accent);
      padding: 0.4rem 0.75rem;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.85rem;
    }
    .preset-btn:hover { background: var(--accent); color: #0f172a; }
    .control { margin-bottom: 1.1rem; }
    .control label { display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 0.35rem; }
    .control span { color: var(--accent); font-weight: 600; }
    input[type="range"] { width: 100%; accent-color: var(--accent); cursor: pointer; }
    .submit-btn {
      width: 100%;
      padding: 0.85rem;
      background: var(--accent);
      color: #0f172a;
      border: none;
      border-radius: 8px;
      font-weight: 700;
      font-size: 1rem;
      cursor: pointer;
      margin-top: 0.5rem;
    }
    .submit-btn:hover { background: var(--accent-hover); color: #fff; }
    .result-box {
      background: #0f172a;
      border-left: 4px solid var(--success);
      padding: 1rem;
      border-radius: 6px;
      margin-bottom: 1.25rem;
    }
    .result-species { font-size: 1.5rem; font-weight: 700; color: var(--success); }
    .bar-group { margin-bottom: 0.85rem; }
    .bar-label { display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.25rem; }
    .bar-bg { background: #0f172a; height: 10px; border-radius: 99px; overflow: hidden; }
    .bar-fill { background: var(--accent); height: 100%; transition: width 0.35s ease; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>🌺 Iris Botanical Species Classifier</h1>
      <p>Multi-Class Random Forest Inference Portal | Deployed on Vercel</p>
    </header>

    <div class="grid">
      <div class="card">
        <h2>Morphological Measurements (cm)</h2>
        <div class="presets">
          <button class="preset-btn" onclick="setPreset(5.1, 3.5, 1.4, 0.2)">Preset: Setosa</button>
          <button class="preset-btn" onclick="setPreset(6.0, 2.7, 4.5, 1.5)">Preset: Versicolor</button>
          <button class="preset-btn" onclick="setPreset(6.7, 3.1, 5.6, 2.4)">Preset: Virginica</button>
        </div>

        <div class="control">
          <label>Sepal Length <span id="sl_val">5.8 cm</span></label>
          <input type="range" id="sepal_length" min="4.0" max="8.0" step="0.1" value="5.8" oninput="updateVal('sepal_length', 'sl_val')" />
        </div>
        <div class="control">
          <label>Sepal Width <span id="sw_val">3.0 cm</span></label>
          <input type="range" id="sepal_width" min="2.0" max="4.5" step="0.1" value="3.0" oninput="updateVal('sepal_width', 'sw_val')" />
        </div>
        <div class="control">
          <label>Petal Length <span id="pl_val">4.3 cm</span></label>
          <input type="range" id="petal_length" min="1.0" max="7.0" step="0.1" value="4.3" oninput="updateVal('petal_length', 'pl_val')" />
        </div>
        <div class="control">
          <label>Petal Width <span id="pw_val">1.3 cm</span></label>
          <input type="range" id="petal_width" min="0.1" max="2.5" step="0.1" value="1.3" oninput="updateVal('petal_width', 'pw_val')" />
        </div>

        <button class="submit-btn" onclick="predictSpecies()">Classify Botanical Species</button>
      </div>

      <div class="card">
        <h2>Classification Output</h2>
        <div class="result-box">
          <div style="font-size:0.85rem; color:var(--muted);">Predicted Taxonomic Class</div>
          <div class="result-species" id="pred_species">Iris Versicolor</div>
          <div style="font-size:0.9rem; margin-top:0.25rem;" id="pred_conf">Confidence: 98.00%</div>
        </div>

        <h3 style="font-size:1rem; color:var(--muted); margin-bottom:0.75rem;">Class Probability Distribution</h3>
        <div id="prob_bars"></div>
      </div>
    </div>
  </div>

  <script>
    function updateVal(inputId, spanId) {
      document.getElementById(spanId).innerText = parseFloat(document.getElementById(inputId).value).toFixed(1) + " cm";
    }

    function setPreset(sl, sw, pl, pw) {
      document.getElementById("sepal_length").value = sl;
      document.getElementById("sepal_width").value = sw;
      document.getElementById("petal_length").value = pl;
      document.getElementById("petal_width").value = pw;
      updateVal("sepal_length", "sl_val");
      updateVal("sepal_width", "sw_val");
      updateVal("petal_length", "pl_val");
      updateVal("petal_width", "pw_val");
      predictSpecies();
    }

    async function predictSpecies() {
      const payload = {
        sepal_length: parseFloat(document.getElementById("sepal_length").value),
        sepal_width: parseFloat(document.getElementById("sepal_width").value),
        petal_length: parseFloat(document.getElementById("petal_length").value),
        petal_width: parseFloat(document.getElementById("petal_width").value)
      };

      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (data.species) {
        document.getElementById("pred_species").innerText = data.species;
        document.getElementById("pred_conf").innerText = `Confidence: ${data.confidence.toFixed(2)}%`;

        const container = document.getElementById("prob_bars");
        container.innerHTML = "";
        for (const [label, pct] of Object.entries(data.probabilities)) {
          container.innerHTML += `
            <div class="bar-group">
              <div class="bar-label"><span>${label}</span><span>${pct.toFixed(2)}%</span></div>
              <div class="bar-bg"><div class="bar-fill" style="width:${pct}%"></div></div>
            </div>
          `;
        }
      }
    }

    window.onload = predictSpecies;
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_home():
    return HTML_PAGE

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