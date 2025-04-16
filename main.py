import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction details below to check if it's **Fraudulent** or **Legitimate**.")

st.subheader("Transaction Features")

# Navigation: Toggle between Single and Batch
option = st.radio("Choose prediction mode:", ("Single Prediction", "Batch Prediction (CSV Upload)"))

# --- Single Prediction Mode ---
if option == "Single Prediction":
    st.subheader("Enter Transaction Details")

    time = st.number_input("Time", min_value=0.0)
    amount = st.number_input("Amount", min_value=0.0, step=0.01)
    v_features = {}
    for i in range(0, 29):
        v_features[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

    if st.button("Check for Fraud"):
        input_data = pd.DataFrame([[
            time,
            *[v_features[f"V{i}"] for i in range(0, 29)],
            amount
        ]], columns=["Time"] + [f"V{i}" for i in range(0, 29)] + ["Amount"])

        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][prediction]

        if prediction == 1:
            st.error(f"⚠️ Fraudulent transaction with {confidence*100:.2f}% confidence.")
        else:
            st.success(f"✅ Legitimate transaction with {confidence*100:.2f}% confidence.")

# --- Batch Prediction ---
else:
    st.subheader("Upload a CSV File for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            required_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

            if all(col in data.columns for col in required_cols):
                st.info(f"Original rows: {len(data)}")

                # Limit for performance
                data = data.head(5000)
                st.success(f"Processing first 5000 rows for better speed.")

                batch_input = data[required_cols]

                predictions = model.predict(batch_input)
                probs = model.predict_proba(batch_input)

                data["Prediction"] = predictions
                data["Confidence"] = probs.max(axis=1)
                data["Result"] = data["Prediction"].map({0: "Legitimate", 1: "Fraudulent"})

                st.dataframe(data[["Time", "Amount", "Result", "Confidence"]].head(20))

                csv = data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

                # Show charts only if toggled
                if st.checkbox("Show Visualizations (on sample of 1000 rows)", value=False):
                    viz_data = data.sample(n=min(1000, len(data)), random_state=42)

                    count_df = viz_data["Result"].value_counts().reset_index()
                    count_df.columns = ["Transaction Type", "Count"]
                    
                    pie_chart = px.pie(
                        count_df,
                        values="Count",
                        names="Transaction Type",
                        title="Fraud vs Legitimate Transactions"
                    )
                    st.plotly_chart(pie_chart)

                    box_plot = px.box(viz_data, x="Result", y="Amount", title="Amount by Transaction Type")
                    st.plotly_chart(box_plot)

                    hist = px.histogram(viz_data, x="Confidence", color="Result", title="Prediction Confidence")
                    st.plotly_chart(hist)

            else:
                st.warning("Uploaded file does not contain the required columns.")

        except Exception as e:
            st.error(f"Error processing file: {e}")
