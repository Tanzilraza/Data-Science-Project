import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ----- CONFIG -----
st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide", initial_sidebar_state="expanded")

# CSS for improved UI
st.markdown("""3
<style>
    .main {
        background: linear-gradient(135deg, #e0f7fa, #ffffff);
        padding: 2rem 3rem;
        border-radius: 15px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #00695c !important;
        color: white !important;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6em 1.4em;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #004d40 !important;
        color: #a7ffeb !important;
    }
    .stDownloadButton>button {
        background-color: #0288d1 !important;
        color: white !important;
        border-radius: 10px;
        font-weight: 600;
    }
    .stDownloadButton>button:hover {
        background-color: #01579b !important;
    }
    .css-1lcbmhc {
        padding-top: 0rem;
    }
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
DATA_FILE = "creditcard.csv"

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_FILE)
    return df

def train_model(df):
    st.info("Training model, please wait...")
    X = df[FEATURE_COLUMNS]
    y = df["Class"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    st.success("✅ Model trained and saved successfully!")
    return model, scaler, X_test, y_test

if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
    @st.cache_resource(show_spinner=False)
    def load_model_and_scaler():
        m = joblib.load(MODEL_FILE)
        s = joblib.load(SCALER_FILE)
        return m, s
    model, scaler = load_model_and_scaler()
    test_data = None
else:
    df = load_data()
    model, scaler, X_test, y_test = train_model(df)
    test_data = (X_test, y_test)

st.title("💳 Credit Card Fraud Detection System")
st.markdown("### Built using Streamlit + Random Forest Machine Learning")
st.markdown("---")

option = st.sidebar.selectbox("Select Option", [
    "🔍 Predict Transaction",
    "📊 Visualize Results",
    "🧪 Evaluate Model",
    "⚙️ Live Simulation"
])

if option == "🔍 Predict Transaction":
    mode = st.radio("Prediction Mode:", ["Single Transaction", "Batch Transactions (CSV)"])

    if mode == "Single Transaction":
        st.subheader("Enter Transaction Details")

        uploaded_csv = st.file_uploader("Optional: Upload CSV to autofill transaction", type=["csv"])
        df_csv = None
        selected_index = None

        if uploaded_csv:
            try:
                df_csv = pd.read_csv(uploaded_csv)
                if not all(col in df_csv.columns for col in FEATURE_COLUMNS):
                    st.warning(f"CSV missing columns: {set(FEATURE_COLUMNS) - set(df_csv.columns)}")
                    df_csv = None
                else:
                    st.success(f"CSV loaded with {len(df_csv)} rows.")
                    selected_index = st.selectbox("Select transaction row index to autofill inputs", df_csv.index)
            except Exception as e:
                st.error(f"CSV loading error: {e}")

        def get_val(col):
            if df_csv is not None and selected_index is not None:
                try:
                    return float(df_csv.loc[selected_index, col])
                except:
                    return 0.0
            return 0.0

        time_input = st.number_input("Time", min_value=0.0, format="%.2f", value=get_val("Time"))
        amount_input = st.number_input("Amount", min_value=0.0, step=0.01, format="%.2f", value=get_val("Amount"))

        cols = st.columns(4)
        v_values = []
        for i in range(1, 29):
            default_val = get_val(f"V{i}")
            with cols[(i - 1) % 4]:
                val = st.number_input(f"V{i}", value=default_val, format="%.5f")
                v_values.append(val)

        if st.button("🔎 Check for Fraud"):
            input_row = [time_input] + v_values + [amount_input]
            input_df = pd.DataFrame([input_row], columns=FEATURE_COLUMNS)

            if df_csv is not None:
                df_rounded = df_csv[FEATURE_COLUMNS].round(5)
                input_rounded = input_df.round(5)
                match = ((df_rounded == input_rounded.iloc[0]).all(axis=1)).any()
                if not match:
                    st.error("🚫 Input values do not match in uploaded CSV -- Fraud Transaction")
                else:
                    st.success("✅ Transaction FOUND in CSV → Not Fraud Transaction")
            else:
                try:
                    input_scaled = scaler.transform(input_df)
                    prediction = model.predict(input_scaled)[0]
                    confidence = model.predict_proba(input_scaled)[0][prediction]
                    bar_color = "#e74c3c" if prediction == 1 else "#27ae60"

                    if prediction == 1:
                        st.error(f"⚠️ Fraud Detected! Confidence: {confidence*100:.2f}%")
                    else:
                        st.success(f"✅ Legitimate Transaction. Confidence: {confidence*100:.2f}%")

                    st.progress(confidence)
                    st.metric("Risk Score", f"{confidence*100:.2f}%", delta="High" if confidence > 0.8 else "Low")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

    else:
        st.subheader("Batch Prediction (Upload CSV)")

        file = st.file_uploader("Upload CSV with transactions", type=["csv"])

        if file:
            try:
                df = pd.read_csv(file)
                if all(col in df.columns for col in FEATURE_COLUMNS):
                    df = df.head(5000)
                    input_scaled = scaler.transform(df[FEATURE_COLUMNS])
                    df["Prediction"] = model.predict(input_scaled)
                    df["Confidence"] = model.predict_proba(input_scaled).max(axis=1)
                    df["Result"] = df["Prediction"].map({0: "Legitimate", 1: "Fraudulent"})

                    st.success(f"Batch Prediction Done for {len(df)} transactions")

                    st.dataframe(df[["Time", "Amount", "Result", "Confidence"]].head(20))

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇ Download Results CSV", csv, "batch_predictions.csv", "text/csv")
                else:
                    st.warning(f"CSV missing required columns: {set(FEATURE_COLUMNS) - set(df.columns)}")
            except Exception as e:
                st.error(f"CSV read error: {e}")

elif option == "📊 Visualize Results":
    st.subheader("Visualize Prediction Results")

    vis_file = st.file_uploader("Upload Prediction CSV", type="csv")
    if vis_file:
        try:
            vis_df = pd.read_csv(vis_file).head(1000)

            if "Result" not in vis_df.columns and "Prediction" in vis_df.columns:
                vis_df["Result"] = vis_df["Prediction"].map({0: "Legitimate", 1: "Fraudulent"})

            if "Result" in vis_df.columns:
                pie_data = vis_df["Result"].value_counts().reset_index()
                pie_data.columns = ["Transaction Type", "Count"]

                st.plotly_chart(px.pie(pie_data, names="Transaction Type", values="Count",
                                      title="Transaction Type Distribution", color_discrete_sequence=px.colors.qualitative.Set2),
                                use_container_width=True)

                st.plotly_chart(px.histogram(vis_df, x="Confidence", color="Result",
                                             title="Prediction Confidence Distribution",
                                             labels={"Confidence": "Confidence Score"},
                                             nbins=40, barmode="overlay",
                                             color_discrete_map={"Legitimate": "green", "Fraudulent": "red"}),
                                use_container_width=True)

                st.plotly_chart(px.box(vis_df, x="Result", y="Amount",
                                       title="Transaction Amount by Type",
                                       labels={"Amount": "Transaction Amount"},
                                       color="Result",
                                       color_discrete_map={"Legitimate": "green", "Fraudulent": "red"}),
                                use_container_width=True)
            else:
                st.warning("CSV must contain a 'Result' or 'Prediction' column to visualize")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

elif option == "🧪 Evaluate Model":
    st.subheader("Evaluate Model on Test Data")

    if test_data:
        X_test, y_test = test_data
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
                    xticklabels=["Legitimate", "Fraudulent"],
                    yticklabels=["Legitimate", "Fraudulent"], ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)

        report = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraudulent"])
        st.text_area("Classification Report", report, height=220)
    else:
        st.info("Upload test CSV file with features + 'Class' column for evaluation")
        test_file = st.file_uploader("Upload Test Dataset", type="csv")
        if test_file:
            try:
                test_df = pd.read_csv(test_file)
                required_cols = FEATURE_COLUMNS + ["Class"]
                if all(col in test_df.columns for col in required_cols):
                    X_test = scaler.transform(test_df[FEATURE_COLUMNS])
                    y_test = test_df["Class"]
                    y_pred = model.predict(X_test)

                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
                                xticklabels=["Legitimate", "Fraudulent"],
                                yticklabels=["Legitimate", "Fraudulent"], ax=ax)
                    ax.set_xlabel("Predicted Label")
                    ax.set_ylabel("True Label")
                    st.pyplot(fig)

                    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraudulent"])
                    st.text_area("Classification Report", report, height=220)
                else:
                    st.warning(f"Dataset missing required columns: {set(required_cols) - set(test_df.columns)}")
            except Exception as e:
                st.error(f"Dataset load error: {e}")

elif option == "⚙️ Live Simulation":
    st.subheader("Live Transaction Simulation")

    if st.button("Start Simulation"):
        st.info("Simulating 10 random transactions...")
        progress_bar = st.progress(0)
        placeholder = st.empty()

        for i in range(10):
            # Generate fake transaction with 30 features
            fake_data = np.random.randn(1, 30)
            fake_df = pd.DataFrame(fake_data, columns=FEATURE_COLUMNS)

            # Scale features
            scaled = scaler.transform(fake_df)

            # Predict fraud or not
            pred = model.predict(scaled)[0]
            prob = model.predict_proba(scaled)[0][pred]

            msg = "⚠️ Fraud Detected!" if pred == 1 else "✅ Legitimate"
            color = "red" if pred == 1 else "green"

            # Display result in same placeholder to simulate live update
            placeholder.markdown(
                f"**Transaction {i+1}:** "
                f"<span style='color:{color}; font-weight:bold'>{msg}</span> — Confidence: {prob*100:.2f}%", 
                unsafe_allow_html=True
            )

            # Update progress bar
            progress_bar.progress((i + 1) * 10)

            # Wait 1 second before next
            time.sleep(1)

        st.success("Simulation Complete!")

