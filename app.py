# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load model, scaler, and top features
# model = joblib.load('ids_model.pkl')
# scaler = joblib.load('scaler.pkl')
# top_features = joblib.load('top_features.pkl')

# st.title("Intrusion Detection System")

# # Manual input
# # st.header("Manual Input")
# # input_data = {}
# # for feature in top_features:
# #     input_data[feature] = st.number_input(feature, value=0.0)

# # if st.button("Predict"):
# #     input_array = np.array([[input_data[f] for f in top_features]])
# #     st.write("Input Values:", dict(zip(top_features, input_array[0])))
# #     input_scaled = scaler.transform(input_array)
# #     st.write("Scaled Input:", input_scaled[0])
# #     pred = model.predict(input_scaled)[0]
# #     probs = model.predict_proba(input_scaled)[0]
# #     st.write(f"Probabilities: Benign={probs[0]:.4f}, FTP-BruteForce={probs[1]:.4f}, SSH-BruteForce={probs[2]:.4f}")
# #     prob = probs[[1, 2]].max()
# #     result = "Benign" if pred == 0 else "Anomaly/Attack"
# #     st.write(f"Result: {result} (Confidence: {prob:.4f})")

# # CSV upload
# st.header("Upload CSV")
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     missing_cols = [f for f in top_features if f not in df.columns]
#     if missing_cols:
#         st.error(f"Missing columns in CSV: {missing_cols}")
#     else:
#         X = df[top_features].values
#         X_scaled = scaler.transform(X)
#         predictions = model.predict(X_scaled)
#         probabilities = model.predict_proba(X_scaled)
#         df['Prediction'] = ['Benign' if p == 0 else 'Anomaly/Attack' for p in predictions]
#         df['Confidence'] = probabilities[:, [1, 2]].max(axis=1)
#         st.write("Predictions:")
#         st.write(df)
#         for i, row in enumerate(probabilities):
#             st.write(f"Row {i+1} Probabilities: Benign={row[0]:.4f}, FTP-BruteForce={row[1]:.4f}, SSH-BruteForce={row[2]:.4f}")


# # Manual input
# st.header("Manual Input")
# input_data = {}
# for feature in top_features:
#     input_data[feature] = st.number_input(feature, value=0.0)

# if st.button("Predict"):
#     input_array = np.array([[input_data[f] for f in top_features]])
#     st.write("Input Values:", dict(zip(top_features, input_array[0])))
#     input_scaled = scaler.transform(input_array)
#     st.write("Scaled Input:", input_scaled[0])
#     pred = model.predict(input_scaled)[0]
#     probs = model.predict_proba(input_scaled)[0]
#     st.write(f"Probabilities: Benign={probs[0]:.4f}, FTP-BruteForce={probs[1]:.4f}, SSH-BruteForce={probs[2]:.4f}")
#     prob = probs[[1, 2]].max()
#     result = "Benign" if pred == 0 else "Anomaly/Attack"
#     st.write(f"Result: {result} (Confidence: {prob:.4f})")

from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
import joblib
import io

# Load model, scaler, and top features
model = joblib.load('ids_model.pkl')
scaler = joblib.load('scaler.pkl')
top_features = joblib.load('top_features.pkl')

app = FastAPI(title="Intrusion Detection System API")

@app.get("/")
def home():
    return {"message": "Welcome to the IDS API"}

# Predict for manual input
@app.post("/predict_manual/")
def predict_manual(features: dict):
    try:
        input_array = np.array([[features[f] for f in top_features]])
        input_scaled = scaler.transform(input_array)
        pred = model.predict(input_scaled)[0]
        probs = model.predict_proba(input_scaled)[0]
        result = "Benign" if pred == 0 else "Anomaly/Attack"
        return {
            "Result": result,
            "Probabilities": {
                "Benign": float(probs[0]),
                "FTP-BruteForce": float(probs[1]),
                "SSH-BruteForce": float(probs[2])
            },
            "Confidence": float(probs[[1, 2]].max())
        }
    except Exception as e:
        return {"error": str(e)}

# Predict for CSV upload
@app.post("/predict_csv/")
def predict_csv(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        missing_cols = [f for f in top_features if f not in df.columns]
        if missing_cols:
            return {"error": f"Missing columns in CSV: {missing_cols}"}
        X = df[top_features].values
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        df['Prediction'] = ['Benign' if p == 0 else 'Anomaly/Attack' for p in predictions]
        df['Confidence'] = probabilities[:, [1, 2]].max(axis=1)
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}
