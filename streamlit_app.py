

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# -----------------------
# Custom CSS
# -----------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Background Image */
    .stApp {
        background: url("https://media.istockphoto.com/id/1091385068/photo/farmer-at-work-under-storm.jpg?s=612x612&w=0&k=20&c=dE8yaxZHn2mgUwZ36fnswTbl7nk7fEr2Ykv54iqGcPc=") no-repeat center center fixed;
        background-size: cover;
    }

    /* Main Content Card */
    .main {
        background: rgba(223, 208, 184, 0.95);
        padding: 30px;
        border-radius: 18px;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.25);
    }

    /* Headings */
    h1, h2, h3 {
        color: #DFD0B8 !important;
        background: #222831;
        padding: 10px 15px;
        border-radius: 12px;
        font-weight: 700;
        
        margin: 25px 0 20px 0 !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #393E46;
        color: #DFD0B8;
        border-radius: 12px;
        font-size: 18px;
        padding: 0.6em 1.2em;
        transition: 0.3s;
        border: none;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background-color: #948979;
        color: #222831;
        transform: scale(1.05);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #222831;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #DFD0B8 !important;
    }

    /* Improved Sticky Notes */
    .note {
        background: #FDF6E3; /* light background */
        padding: 20px;
        border-radius: 14px;
        box-shadow: 4px 6px 12px rgba(0,0,0,0.2);
        border-left: 8px solid #393E46;
        min-height: 150px;
        margin-bottom: 20px;
    }
    .note h3 {
        background: #222831; /* dark header */
        color: #DFD0B8 !important; /* light text */
        padding: 10px 15px;
        border-radius: 10px;
        margin-top: 0;
    }
    .note p {
        color: #393E46;
        font-size: 15px;
        margin-top: 10px;
    }

    /* Custom Recommendation Box */
    .recommend-box {
        background: #393E46;
        color: #DFD0B8;
        padding: 20px;
        border-radius: 14px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
        margin: 20px 0;
    }

    /* Progress Bar */
    .stProgress .st-bo {
        background-color: #948979;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Load Model
# -----------------------
model = joblib.load("crop_recommendation_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# -----------------------
# Crop Info
# -----------------------
crop_info = {
    "rice": "Rice is a staple food crop grown in waterlogged fields. Requires high rainfall and humidity.",
    "wheat": "Wheat is a temperate climate crop requiring moderate rainfall and well-drained soil.",
    "maize": "Maize (corn) is versatile, grown in warm climate with moderate rainfall.",
    "chickpea": "Chickpeas need dry and cool climate with minimal rainfall.",
    "mungbean": "Mung beans grow well in warm climate with moderate rainfall.",
    "lentil": "Lentils prefer cooler climate and well-drained loamy soil.",
}

# -----------------------
# Title
# -----------------------
st.title("🌾 AI-Powered Crop Recommendation System")

# -----------------------
# Sidebar
# -----------------------
st.sidebar.title("📌 Navigation")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=120)
st.sidebar.image(
    "https://vodenglish.news/wp-content/uploads/2022/07/signal-2022-07-05-045019_002.jpeg",
    caption="Farmer",
    use_container_width=True
)

section = st.sidebar.radio("Go to", ["Dataset Explorer", "Model Information", "Crop Information", "Crop Recommendation"])

# -----------------------
# Dataset Explorer
# -----------------------
if section == "Dataset Explorer":
    st.header("📊 Dataset Explorer")
    try:
        df = pd.read_csv("crop_recommendation.csv")
        st.subheader("🔍 Preview of Dataset")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("📈 Feature Distribution")
        feature = st.selectbox("Choose a feature", df.columns[:-1])
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, ax=ax)
        st.pyplot(fig)
    except FileNotFoundError:
        st.warning("Dataset not found. Please upload crop_recommendation.csv")

# -----------------------
# Model Info
# -----------------------
elif section == "Model Information":
    st.header("🤖 Model Information")
    st.markdown("""
    <div class='note'>
        <h3>🤖 Model Details</h3>
        <p>
        - <b>Algorithm</b>: Random Forest Classifier <br>
        - <b>Features</b>: N, P, K, Temperature, Humidity, pH, Rainfall <br>
        - <b>Output</b>: Recommended Crop
        </p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------
# Crop Information
# -----------------------
elif section == "Crop Information":
    st.header("🌱 Crop Information")

    # Display in 2-column grid
    cols = st.columns(2)
    i = 0
    for crop, desc in crop_info.items():
        with cols[i % 2]:
            st.markdown(f"""
            <div class='note'>
                <h3>🌱 {crop.capitalize()}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
        i += 1

# -----------------------
# Crop Recommendation
# -----------------------
elif section == "Crop Recommendation":
    st.header("🌱 Crop Recommendation")

    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0)
        ph = st.number_input("pH Value", 0.0, 14.0, 6.5)
    with col2:
        P = st.number_input("Phosphorus (P)", 0.0, 200.0, 50.0)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    with col3:
        K = st.number_input("Potassium (K)", 0.0, 200.0, 50.0)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

    temperature = st.slider("🌡️ Temperature (°C)", 0.0, 60.0, 25.0)

    if st.button("🔍 Recommend Crop"):
        feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        X = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_names)

        pred_id = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        classes = label_encoder.classes_.tolist()
        pred_label = classes[pred_id]

        # Custom styled recommendation box
        st.markdown(f"""
        <div class="recommend-box">
            🌟 Recommended Crop: {pred_label.capitalize()}
        </div>
        """, unsafe_allow_html=True)

        st.write("### 📊 Prediction Confidence")
        top3_idx = np.argsort(proba)[::-1][:3]
        for i in top3_idx:
            st.progress(proba[i])
            st.write(f"**{classes[i]}** → {proba[i]:.2%}")

        if pred_label in crop_info:
            st.markdown(f"""
            <div class='note'>
                <h3>ℹ️ About {pred_label.capitalize()}</h3>
                <p>{crop_info[pred_label]}</p>
            </div>
            """, unsafe_allow_html=True)

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.caption("🌱 AI for Agriculture ")
