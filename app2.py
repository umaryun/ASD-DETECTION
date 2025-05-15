# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sqlite3

app2 = Flask(__name__)
CORS(app2)  # Enable CORS to allow requests from the frontend

# Load the saved model, scaler, and encoders
with open("asd_detection2.pkl", "rb") as file:
    model_data = pickle.load(file)
    model = model_data["model"]
    scaler = model_data["scaler"]
    label_encoders = model_data["encoders"]
    svm_model = model_data["svm_model"]
    knn_model = model_data["knn_model"]
    dt_model = model_data["dt_model"]

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('asd_predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                A1 INTEGER, A2 INTEGER, A3 INTEGER, A4 INTEGER, A5 INTEGER,
                A6 INTEGER, A7 INTEGER, A8 INTEGER, A9 INTEGER, A10 INTEGER,
                Age INTEGER, Sex TEXT, Jaundice TEXT, Family_ASD TEXT,
                Prediction TEXT, Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

@app2.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    new_data = {
        "A1": data["A1"], "A2": data["A2"], "A3": data["A3"], "A4": data["A4"], "A5": data["A5"],
        "A6": data["A6"], "A7": data["A7"], "A8": data["A8"], "A9": data["A9"], "A10": data["A10"],
        "Age": data["Age"], "Sex": data["Sex"], "Jauundice": data["Jaundice"], "Family_ASD": data["Family_ASD"]
    }
    new_df = pd.DataFrame([new_data])
    for col in ["Sex", "Jauundice", "Family_ASD"]:
        new_df[col] = label_encoders[col].transform(new_df[col].str.lower())
    new_df["Age"] = scaler.transform(new_df[["Age"]])
    svm_proba = svm_model.predict_proba(new_df)[:, 1]
    knn_proba = knn_model.predict_proba(new_df)[:, 1]
    dt_proba = dt_model.predict_proba(new_df)[:, 1]
    stacked_input = pd.DataFrame({'SVM': svm_proba, 'KNN': knn_proba, 'DecisionTree': dt_proba})
    prediction = model.predict(stacked_input)[0]
    result = "ASD Positive" if prediction == 1 else "ASD Negative"

    # Save to database
    conn = sqlite3.connect('asd_predictions.db')
    c = conn.cursor()
    c.execute('''INSERT INTO predictions (A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, Age, Sex, Jaundice, Family_ASD, Prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (new_data["A1"], new_data["A2"], new_data["A3"], new_data["A4"], new_data["A5"],
                new_data["A6"], new_data["A7"], new_data["A8"], new_data["A9"], new_data["A10"],
                new_data["Age"], new_data["Sex"], new_data["Jauundice"], new_data["Family_ASD"], result))
    conn.commit()
    conn.close()

    return jsonify({"result": result})

if __name__ == '__main__':
    app2.run(host='0.0.0.0', port=5500)