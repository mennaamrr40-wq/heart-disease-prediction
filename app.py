import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# --- Configuration ---
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model & Assets ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('final_model.pkl')
        with open('model_columns.json', 'r') as f:
            model_columns = json.load(f)
        return model, model_columns
    except FileNotFoundError:
        st.error("Model files not found. Please run 'python setup_model.py' first.")
        return None, None

model, model_columns = load_assets()

# --- Custom Dark Medical Theme CSS ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1e26;
        border-right: 1px solid #2d333b;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #e6e9ef !important;
        font-family: 'Segoe UI', sans-serif;
    }
    p, label {
        color: #c9d1d9 !important;
    }
    
    /* Inputs */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #e6e9ef !important;
        font-weight: 500;
        font-size: 1rem;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        box-shadow: 0 0 10px rgba(46, 160, 67, 0.4);
    }
    
    /* Custom Info Box */
    .info-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Result Cards */
    .result-card-high {
        background-color: #3d1416;
        border: 1px solid #f85149;
        color: #ffa198;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 4px 20px rgba(248, 81, 73, 0.2);
    }
    .result-card-low {
        background-color: #11291b;
        border: 1px solid #238636;
        color: #7ee787;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 4px 20px rgba(35, 134, 54, 0.2);
    }
    
    /* Tooltip adjustments */
    .stTooltipIcon {
        color: #8b949e !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Input (Patient Centric) ---
st.sidebar.title("Patient Details")
st.sidebar.write("Please fill in the medical information below.")
st.sidebar.markdown("---")

def user_input_features():
    # --- Demographics ---
    st.sidebar.subheader("Demographics")
    
    # Sex
    sex = st.sidebar.selectbox(
        "Sex",
        ["Male", "Female"],
        help="Biological sex affects heart disease risk. Men generally have a higher risk at younger ages, while women's risk increases after menopause."
    )
    
    # Age
    age = st.sidebar.slider(
        "Age (years)",
        20, 90, 50,
        help="Age is a significant risk factor. The risk of heart disease increases as you get older."
    )

    # --- Vital Signs ---
    st.sidebar.subheader("Vital Signs")
    
    # Blood Pressure
    trestbps = st.sidebar.number_input(
        "Resting Blood Pressure (mm Hg)",
        80, 200, 120,
        help="High blood pressure (hypertension) strain the heart and arteries. Normal is typically around 120/80 mmHg."
    )
    
    # Cholesterol
    chol = st.sidebar.number_input(
        "Cholesterol (mg/dl)",
        100, 600, 200,
        help="High levels of cholesterol can clog instability in arteries. >240 mg/dl is considered high."
    )
    
    # Heart Rate
    thalach = st.sidebar.slider(
        "Max Heart Rate Achieved",
        60, 220, 150,
        help="The maximum heart rate reached during exercise test. Lower maximum rates can indicate heart issues."
    )

    # --- Clinical Symptoms & Test Results ---
    st.sidebar.subheader("Clinical Factors")
    
    # Chest Pain
    cp = st.sidebar.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
        help="""
        • Typical Angina: Chest pain caused by physical exertion or stress.
        • Atypical Angina: Chest pain with some but not all symptoms of angina.
        • Non-anginal Pain: Chest pain not related to the heart.
        • Asymptomatic: No chest pain symptoms, but may have silent ischemia.
        """
    )
    
    # Fasting Blood Sugar
    fbs = st.sidebar.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["False", "True"],
        format_func=lambda x: "Yes (High)" if x == "True" else "No (Normal)",
        help="Indicates if fasting blood sugar is elevated (possible diabetes indicator). Diabetes increases heart disease risk."
    )
    
    # Exercise Induced Angina
    exang = st.sidebar.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"],
        help="Chest pain that occurs specifically during exercise. This is a tell-tale sign of reduced blood flow to the heart."
    )
    
    # Resting ECG
    restecg = st.sidebar.selectbox(
        "Resting ECG Results",
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
        help="""
        • Normal: No abnormalities.
        • ST-T Abnormality: Unusual heartbeat patterns usually requiring observation.
        • LV Hypertrophy: Thickening of the heart's main pumping chamber.
        """
    )
    
    # Slope
    slope = st.sidebar.selectbox(
        "Slope of Peak Exercise ST Segment",
        ["Upsloping", "Flat", "Downsloping"],
        help="Indicates how the heart rate recovers after exercise. 'Downsloping' and 'Flat' can be concerning signs."
    )
    
    # Thalassemia
    thal = st.sidebar.selectbox(
        "Thalassemia (Blood Disorder Status)",
        ["Normal", "Fixed Defect", "Reversable Defect"],
        help="Thalassemia is a blood disorder. 'Fixed Defect' and 'Reversable Defect' in this context refer to blood flow defects seen in nuclear stress tests."
    )
    
    # Major Vessels
    ca = st.sidebar.slider(
        "Number of Major Vessels Colored by Fluoroscopy (0-3)",
        0, 3, 0,
        help="The number of major blood vessels (0-3) that light up on a fluoroscopy scan. Fewer vessels showing up indicates blockages."
    )
    
    # Mappings to match Model Training
    sex_map = {"Male": "Male", "Female": "Female"}
    cp_map = {
        "Typical Angina": "typical angina", 
        "Atypical Angina": "atypical angina", 
        "Non-anginal Pain": "non-anginal", 
        "Asymptomatic": "asymptomatic"
    }
    fbs_map = {"False": False, "True": True}
    
    data = {
        'age': age,
        'sex': sex_map[sex],
        # Dataset removed from UI, default to "Cleveland" or baseline (missing logic handles this by default as 0s if we skip it)
        # Using a fixed placeholder if needed for logic, but simply omitting it relies on alignment later.
        'cp': cp_map[cp],
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs_map[fbs],
        'restecg': restecg.lower() if restecg != "Left Ventricular Hypertrophy" else "lv hypertrophy",
        'thalach': thalach,
        'exang': True if exang == "Yes" else False,
        'oldpeak': 0.0, # Not in recent list, user didn't ask to remove but focused on simplify. Wait, I should keep 'oldpeak' but explain it properly.
        # User said "Keep ALL patient input widgets". I missed 'oldpeak' in simple copy-paste above? Let me check.
        # Ah, I should look at my previous code. Oldpeak was there. "ST Depression Induced by Exercise".
        'slope': slope.lower(),
        'ca': ca,
        'thal': thal.lower()
    }
    
    # Handling specific map tweaks
    if restecg == "Normal": data['restecg'] = "normal"
    elif restecg == "ST-T Wave Abnormality": data['restecg'] = "st-t abnormality" 
    elif restecg == "Left Ventricular Hypertrophy": data['restecg'] = "lv hypertrophy"
    
    # Fix 'thal'
    if thal == "Normal": data['thal'] = "normal"
    elif thal == "Fixed Defect": data['thal'] = "fixed defect"
    elif thal == "Reversable Defect": data['thal'] = "reversable defect"

    return data, age # Return age separately if needed for logic, but data has it.

# I missed 'oldpeak' in the UI section above, let's add it back before returning.
# Re-defining function content to include oldpeak properly.

def get_sidebar_inputs():
    # --- Demographics ---
    st.sidebar.subheader("Demographics")
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"], help="Biological sex affects heart disease risk.")
    age = st.sidebar.slider("Age (years)", 20, 90, 50, help="Risk increases with age.")

    # --- Vital Signs ---
    st.sidebar.subheader("Vital Signs")
    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Normal is ~120/80.")
    chol = st.sidebar.number_input("Cholesterol (mg/dl)", 100, 600, 200, help=">240 mg/dl is high.")
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150, help="Max heart rate during stress test.")

    # --- Clinical ---
    st.sidebar.subheader("Clinical Factors")
    cp = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], 
                              help="Typical: Exertion pain. Atypical: Non-classic symptoms. Non-anginal: Unrelated to heart. Asymptomatic: Silent.")
    
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"], 
                               format_func=lambda x: "Yes (High)" if x == "True" else "No (Normal)",
                               help="Indicator for diabetes.")
    
    restecg = st.sidebar.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
                                   help="ECG scan results while at rest.")
                                   
    exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"], help="Pain during exercise?")
    
    # Re-adding Oldpeak
    oldpeak = st.sidebar.number_input("ST Depression (Oldpeak)", 0.0, 6.2, 0.0, step=0.1, 
                                      help="A finding on the ECG trace during exercise. Higher values indicate more stress on the heart.")
    
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"],
                                 help="Pattern of the heart rate recovery.")
    
    ca = st.sidebar.slider("Major Vessels (0-3)", 0, 3, 0, help="Number of healthy vessels visible on scan. Lower is worse.")
    
    thal = st.sidebar.selectbox("Thalassemia Status", ["Normal", "Fixed Defect", "Reversable Defect"],
                                help="Blood flow status from nuclear test.")

    # Mapping
    data = {
        'age': age,
        'sex': sex, # Map later or here
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return data

# --- Main App ---

# Main Title & Overview
st.title("💓 Heart Disease Risk Assessment")
st.markdown("""
<div class="info-box">
    <h4>Medical Decision Support System</h4>
    <p>This professional tool utilizes advanced machine learning (Random Forest) to assess the probability of heart disease based on clinical parameters.</p>
    <p><b>Instructions:</b> Use the sidebar to enter the patient's demographics, vital signs, and clinical test results.</p>
</div>
""", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    # Sidebar inputs
    inputs = get_sidebar_inputs()
    
    # Display Summary of Inputs (Optional, cleanly formatted)
    st.subheader("Patient Summary")
    
    # Create a nice layout for summary
    summary_df = pd.DataFrame([inputs])
    # Tweak display for summary
    display_inputs = inputs.copy()
    display_inputs['fbs'] = "High" if inputs['fbs'] == "True" else "Normal"
    st.dataframe(pd.DataFrame([display_inputs]), hide_index=True)
    
    st.markdown("---")
    
    if st.button("RUN ASSESSMENT", type="primary"):
        if model is not None:
            # --- Preprocessing Logic ---
            
            # Map values back to model expected format
            sex_map = {"Male": "Male", "Female": "Female"}
            cp_map = {
                "Typical Angina": "typical angina", 
                "Atypical Angina": "atypical angina", 
                "Non-anginal Pain": "non-anginal", 
                "Asymptomatic": "asymptomatic"
            }
            fbs_map = {"False": False, "True": True}
            
            raw_data = {
                'age': inputs['age'],
                'sex': sex_map[inputs['sex']],
                'dataset': 'Cleveland', # Default placeholder to avoid errors, alignment handles it anyway
                'cp': cp_map[inputs['cp']],
                'trestbps': inputs['trestbps'],
                'chol': inputs['chol'],
                'fbs': fbs_map[inputs['fbs']],
                'restecg': inputs['restecg'].lower(),
                'thalach': inputs['thalach'],
                'exang': True if inputs['exang'] == "Yes" else False,
                'oldpeak': inputs['oldpeak'],
                'slope': inputs['slope'].lower(),
                'ca': inputs['ca'],
                'thal': inputs['thal'].lower()
            }
            
            # Fix specific strings
            if inputs['restecg'] == "Left Ventricular Hypertrophy": raw_data['restecg'] = "lv hypertrophy"
            elif inputs['restecg'] == "ST-T Wave Abnormality": raw_data['restecg'] = "st-t abnormality"
            
            if inputs['thal'] == "Fixed Defect": raw_data['thal'] = "fixed defect"
            elif inputs['thal'] == "Reversable Defect": raw_data['thal'] = "reversable defect"
            
            input_df = pd.DataFrame([raw_data])
            
            # Feature Engineering
            input_df['high_chol'] = (input_df['chol'] > 240).astype(int)
            input_df['exang_int'] = input_df['exang'].replace({True: 1, False: 0})
            input_df['exercise_risk'] = ((input_df['exang_int'] == 1) & (input_df['oldpeak'] > 1)).astype(int)
            input_df.drop('exang_int', axis=1, inplace=True)
            input_df['age_oldpeak'] = input_df['age'] * input_df['oldpeak']
            
            # Encoding & Alignment
            input_encoded = pd.get_dummies(input_df, drop_first=True)
            
            for col in model_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            input_encoded = input_encoded[model_columns]
            
            # Predict
            prediction = model.predict(input_encoded)
            prediction_proba = model.predict_proba(input_encoded)
            risk_score = prediction_proba[0][1]
            
            # Results
            st.markdown("### Assessment Result")
            
            if risk_score > 0.5:
                st.markdown(f"""
                <div class="result-card-high">
                    <h2>⚠️ HIGH RISK DETECTED</h2>
                    <h1 style="color: #ff8b8b !important;">{risk_score:.1%}</h1>
                    <p>The model indicates a high probability of heart disease presence.</p>
                    <p><b>Recommendation:</b> Immediate clinical evaluation suggested.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-low">
                    <h2>✅ LOW RISK</h2>
                    <h1 style="color: #7ee787 !important;">{risk_score:.1%}</h1>
                    <p>The model indicates a low probability of heart disease presence.</p>
                    <p><b>Recommendation:</b> Maintain healthy lifestyle and routine checkups.</p>
                </div>
                """, unsafe_allow_html=True)

with col2:
    # Right column can be used for additional context or just spacing
    # Adding a medical disclaimer in the sidebar or here
    st.markdown(" ")
    
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ **Disclaimer:** This tool is for educational purposes only. Always consult a healthcare professional for diagnosis.")
