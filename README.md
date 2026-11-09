# Iris Flower Classification System

An end-to-end machine learning project classifying Iris flower species (*Setosa*, *Versicolor*, *Virginica*) based on sepal and petal dimensions.

## Features
- **Exploratory Data Analysis (EDA):** Morphometric distribution and feature correlation.
- **Model Training:** Multi-class classification using Random Forest (`scikit-learn`).
- **Interactive UI:** Web-based inference app built with Streamlit.
- **Evaluation:** Evaluated across Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

## Installation & Local Run
```bash
git clone [https://github.com/Aquiarh/iris_classification.git](https://github.com/Aquiarh/iris_classification.git)
cd iris_classification
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py