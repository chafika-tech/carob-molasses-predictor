import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVR
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Carob Extract Predictor", page_icon="🌿", layout="wide")

# --- LOAD MODEL BUNDLE ---
@st.cache_resource
def load_bundle():
    file_path = 'svr_model_bundle.pkl'
    if os.path.exists(file_path):
        return joblib.load(file_path)
    return None

bundle = load_bundle()

# --- MAIN INTERFACE ---
st.title("🌿 Carob Molasses Predictive Tool")

# --- CENTERED BLUE BOX ---
st.markdown(
    """
    <div style="background-color: #e8f4f9; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff; text-align: center; margin-bottom: 20px;">
        <span style="color: #0056b3; font-size: 18px; font-weight: bold;">
            Enter experimental parameters to predict Yield, TPC, and 5-HMF
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

if bundle is None:
    st.error("Error: Could not find 'svr_model_bundle.pkl'. Please ensure it's in the same folder!")
else:
    # Sidebar for Inputs
    with st.sidebar:
        st.header("Extraction Parameters")
        in_time = st.slider("Time (min)", 10.0, 60.0, 35.0)
        in_power = st.slider("Power (W)", 200, 800, 500)
        in_ratio = st.number_input("Ratio (g/mL)", 0.0160, 0.0500, 0.0250, format="%.4f")
        
        st.markdown("---")
        predict_btn = st.button("Predict Results")

    # --- PREDICTION LOGIC ---
    if predict_btn:
        # 1. Create Interaction Features
        X_int = np.array([[
            in_time, in_power, in_ratio,
            in_time * in_power,
            in_time * in_ratio,
            in_power * in_ratio
        ]])

        # 2. Scale
        X_scaled = bundle['scaler_int'].transform(X_int)
        
       # 3. Layout for Results
        st.subheader("Predicted Outcomes")
        c1, c2, c3 = st.columns(3)
        cols_list = [c1, c2, c3]
        
        # Match these keys EXACTLY to your bundle['y_cols']
        units = {'yield': "(%)", 'TPC': "(mg/g)", 'HMF': "(mg/100g)"}
        # Create a display mapping for the UI labels
        display_names = {'yield': "Yield", 'TPC': "TPC", 'HMF': "5-HMF"}

        # 4. Predict Loop
        for i, col_name in enumerate(bundle['y_cols']):
            res_data = bundle['results'][col_name]
            model = SVR(kernel='rbf', **res_data['params'])
            model.fit(bundle['X_train_final'], bundle['Y_train_final'][:, i])
            
            pred_s = model.predict(X_scaled)
            final_val = bundle['scalers_y'][col_name].inverse_transform(pred_s.reshape(-1, 1))[0, 0]
            
            # Use the display_names to force "5-HMF" on the screen
            label = display_names.get(col_name, col_name)
            unit_label = units.get(col_name, "")
            
            cols_list[i].metric(f"{label} {unit_label}", f"{final_val:.2f}")

        # --- CENTERED SUCCESS MESSAGE ---
        st.markdown(
            """
            <div style="text-align: center; padding: 10px; color: #155724; background-color: #d4edda; border-radius: 5px; margin-top: 20px;">
                Calculations based on optimized SVR models.
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("Model validated using Leave-One-Out Cross-Validation (LOOCV).")