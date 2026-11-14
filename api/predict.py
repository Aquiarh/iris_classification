from http.server import BaseHTTPRequestHandler
import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Train lightweight model in memory once per cold start (< 20ms)
iris = load_iris()
X, y = iris.data, iris.target
target_names = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))

            features = np.array([[
                float(data["sepal_length"]),
                float(data["sepal_width"]),
                float(data["petal_length"]),
                float(data["petal_width"])
            ]])

            pred_idx = int(clf.predict(features)[0])
            probs = clf.predict_proba(features)[0]

            response_data = {
                "species": target_names[pred_idx],
                "confidence": round(float(probs[pred_idx]) * 100, 2),
                "probabilities": {
                    target_names[i]: round(float(probs[i]) * 100, 2)
                    for i in range(len(target_names))
                }
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))

        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))