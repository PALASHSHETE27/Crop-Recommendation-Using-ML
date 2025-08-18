# import streamlit as st
# import joblib
# import numpy as np

# st.set_page_config(page_title="Crop Recommendation", page_icon="🌾")
# st.title("🌾 Crop Recommendation (AI)")
# st.write("Enter soil and climate parameters to get the best crop suggestion.")

# model = joblib.load("crop_recommendation_model.joblib")
# label_encoder = joblib.load("label_encoder.joblib")

# col1, col2, col3 = st.columns(3)
# with col1:
#     N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
#     ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
# with col2:
#     P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
#     humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
# with col3:
#     K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
#     rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)

# temperature = st.slider("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0, step=0.5)

# if st.button("Recommend Crop"):
#     X = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=float)
#     pred_id = int(model.predict(X)[0])
#     proba = model.predict_proba(X)[0]
#     classes = label_encoder.classes_.tolist()
#     pred_label = classes[pred_id]

#     st.success(f"**Recommended Crop:** {pred_label}")
#     top3_idx = np.argsort(proba)[::-1][:3]
#     st.write("Top 3 recommendations:")
#     for i in top3_idx:
#         st.write(f"- {classes[i]}: {proba[i]:.2%}")

# st.caption("Model: RandomForestClassifier | Features: N, P, K, temperature, humidity, pH, rainfall")








# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # -----------------------
# # Page Config
# # -----------------------
# st.set_page_config(
#     page_title="🌾 Crop Recommendation System",
#     page_icon="🌱",
#     layout="wide"
# )

# # -----------------------
# # Custom CSS for Styling
# # -----------------------
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

#     html, body, [class*="css"] {
#         font-family: 'Poppins', sans-serif;
#     }
#     .main {
#         background-color: #f4fff4;
#         background-image: linear-gradient(to right, #f7fff7, #e6ffe6);
#         padding: 20px;
#         border-radius: 15px;
#     }
#     h1 {
#         color: #2e7d32 !important;
#         text-align: center;
#         font-weight: 700;
#     }
#     h2, h3 {
#         color: #388e3c !important;
#     }
#     .stButton>button {
#         background-color: #2e7d32;
#         color: white;
#         border-radius: 10px;
#         font-size: 18px;
#         padding: 0.6em 1.2em;
#         transition: 0.3s;
#     }
#     .stButton>button:hover {
#         background-color: #1b5e20;
#         transform: scale(1.05);
#     }
#     .stProgress .st-bo {
#         background-color: #81c784;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # -----------------------
# # Load ML Artifacts
# # -----------------------
# model = joblib.load("crop_recommendation_model.joblib")
# label_encoder = joblib.load("label_encoder.joblib")

# # -----------------------
# # Crop Information with Direct Image URLs
# # -----------------------
# crop_info = {
#     "rice": {
#         "desc": "Rice is a staple food crop grown in waterlogged fields. Requires high rainfall and humidity.",
#         "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ3TKcWv-VxNg4owOnP6OfwuyhWAr2BaRyIBQ&s"
#     },
#     "wheat": {
#         "desc": "Wheat is a temperate climate crop requiring moderate rainfall and well-drained soil.",
#         "image": "https://www.google.com/imgres?q=wheat&imgurl=https%3A%2F%2Fengrain.us%2Fwp-content%2Fuploads%2F2023%2F05%2Fshutterstock_488899324-1080x675.jpg&imgrefurl=https%3A%2F%2Fengrain.us%2Funderstanding-wheat-protein-content%2F&docid=xP6W-ZFAyUILBM&tbnid=AB6Yn-ZKdU4KhM&vet=12ahUKEwjkyuzJlI-PAxW9n2MGHYPJJOEQM3oECCAQAA..i&w=1080&h=675&hcb=2&ved=2ahUKEwjkyuzJlI-PAxW9n2MGHYPJJOEQM3oECCAQAA"
#     },
#     "maize": {
#         "desc": "Maize (corn) is versatile, grown in warm climate with moderate rainfall.",
#         "image": "https://www.google.com/imgres?q=maize&imgurl=https%3A%2F%2Fwww.keshrinandan.com%2Fwp-content%2Fuploads%2F2015%2F08%2Fke_maize_nutrition.jpg&imgrefurl=https%3A%2F%2Fwww.keshrinandan.com%2Fmaize%2Fmaize-nutrition-facts%2F&docid=FGnUor4Lzyuw6M&tbnid=s_Q0TOnu2lu1DM&vet=12ahUKEwj66vXwlI-PAxWI9zgGHTaeAxIQM3oECBwQAA..i&w=3000&h=1993&hcb=2&ved=2ahUKEwj66vXwlI-PAxWI9zgGHTaeAxIQM3oECBwQA"
#     },
#     "chickpea": {
#         "desc": "Chickpeas need dry and cool climate with minimal rainfall.",
#         "image": "https://www.google.com/imgres?q=chickpea&imgurl=https%3A%2F%2Fi0.wp.com%2Fpost.medicalnewstoday.com%2Fwp-content%2Fuploads%2Fsites%2F3%2F2022%2F04%2Fchickpeas_closeup_1296x728_header-1024x575.jpg%3Fw%3D1155%26h%3D1528&imgrefurl=https%3A%2F%2Fwww.medicalnewstoday.com%2Farticles%2F280244&docid=qwhITZYpJZ9R8M&tbnid=x7J3flpRUsq5VM&vet=12ahUKEwjK7LaMlY-PAxXt76ACHeo4LQ4QM3oECBIQAA..i&w=1024&h=575&hcb=2&ved=2ahUKEwjK7LaMlY-PAxXt76ACHeo4LQ4QM3oECBIQAA"
#     },
#     "mungbean": {
#         "desc": "Mung beans grow well in warm climate with moderate rainfall.",
#         "image": "https://www.google.com/imgres?q=mungbean&imgurl=https%3A%2F%2Fcdn-prod.medicalnewstoday.com%2Fcontent%2Fimages%2Farticles%2F324%2F324156%2Fmung-beans.jpg&imgrefurl=https%3A%2F%2Fwww.medicalnewstoday.com%2Farticles%2F324156&docid=hIPgqEwpPP8V_M&tbnid=WR7_Lbk5Zp54lM&vet=12ahUKEwi3w72jlY-PAxWJn2MGHTtsJAYQM3oECBcQAA..i&w=1100&h=734&hcb=2&ved=2ahUKEwi3w72jlY-PAxWJn2MGHTtsJAYQM3oECBcQAA"
#     },
#     "lentil": {
#         "desc": "Lentils prefer cooler climate and well-drained loamy soil.",
#         "image": "https://www.google.com/imgres?q=lentil&imgurl=https%3A%2F%2Fwww.foodandwine.com%2Fthmb%2FgNVN2qyjJjDWIjb0Ihk0YTZIQHY%3D%2F1500x0%2Ffilters%3Ano_upscale()%3Amax_bytes(150000)%3Astrip_icc()%2FLentils-Are-Your-Ticket-to-Quick-and-Hearty-Meals-FT-BLOG0125-83a85aa4ddcf47f4ac0a5a71870a379d.jpg&imgrefurl=https%3A%2F%2Fwww.foodandwine.com%2Fhow-to-cook-lentils-8781157&docid=xJJz9OdRvZzsqM&tbnid=mP-rgxHV8fLSWM&vet=12ahUKEwjfueGylY-PAxUmwzgGHdlMFPoQM3oECCIQAA..i&w=1500&h=1000&hcb=2&ved=2ahUKEwjfueGylY-PAxUmwzgGHdlMFPoQM3oECCIQAA"
#     },
# }

# # -----------------------
# # Title + Intro
# # -----------------------
# st.title("🌾 AI-Powered Crop Recommendation System")
# st.markdown(
#     """
#     <div style="text-align:center; font-size:18px; color:#2e7d32;">
#     Welcome to the <b>Crop Recommendation App</b>.  
#     This tool uses <b>Machine Learning</b> to recommend the best crop to cultivate  
#     based on soil and climatic conditions. 🌱
#     </div>
#     <hr>
#     """,
#     unsafe_allow_html=True
# )

# # -----------------------
# # Sidebar Navigation
# # -----------------------
# st.sidebar.title("📌 Navigation")
# section = st.sidebar.radio("Go to", ["Dataset Explorer", "Model Information", "Crop Recommendation"])

# # -----------------------
# # 1. Dataset Explorer
# # -----------------------
# if section == "Dataset Explorer":
#     st.header("📊 Dataset Explorer")

#     try:
#         df = pd.read_csv("crop_recommendation.csv")
#         st.subheader("🔍 Preview of Dataset")
#         st.dataframe(df.head())

#         st.subheader("📈 Feature Distribution")
#         feature = st.selectbox("Choose a feature", df.columns[:-1])  # exclude label column
#         fig, ax = plt.subplots()
#         sns.histplot(df[feature], kde=True, ax=ax)
#         st.pyplot(fig)

#     except FileNotFoundError:
#         st.warning("Dataset file not found. Please ensure 'crop_recommendation.csv' exists.")

# # -----------------------
# # 2. Model Information
# # -----------------------
# elif section == "Model Information":
#     st.header("🤖 Model Information")

#     st.write("""
#     - **Algorithm**: Random Forest Classifier  
#     - **Features Used**:  
#         - Nitrogen (N)  
#         - Phosphorus (P)  
#         - Potassium (K)  
#         - Temperature (°C)  
#         - Humidity (%)  
#         - pH Value  
#         - Rainfall (mm)  
#     - **Output**: Recommended Crop  
#     """)

#     st.success("✅ Model and Label Encoder Loaded Successfully!")

# # -----------------------
# # 3. Crop Recommendation
# # -----------------------
# elif section == "Crop Recommendation":
#     st.header("🌱 Crop Recommendation")

#     st.write("Enter soil and climate parameters below:")

#     col1, col2, col3 = st.columns(3)
#     with col1:
#         N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
#         ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
#     with col2:
#         P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
#         humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
#     with col3:
#         K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
#         rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)

#     temperature = st.slider("🌡️ Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0, step=0.5)

#     if st.button("🔍 Recommend Crop"):
#         X = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=float)
#         pred_id = int(model.predict(X)[0])
#         proba = model.predict_proba(X)[0]
#         classes = label_encoder.classes_.tolist()
#         pred_label = classes[pred_id]

#         # ✅ Show Result
#         st.success(f"🌟 **Recommended Crop:** {pred_label.capitalize()}")

#         # Confidence
#         st.write("### 📊 Prediction Confidence")
#         top3_idx = np.argsort(proba)[::-1][:3]
#         for i in top3_idx:
#             st.progress(proba[i])
#             st.write(f"**{classes[i]}** → {proba[i]:.2%}")

#         # ✅ Show Crop Info + Image
#         if pred_label in crop_info:
#             st.subheader(f"ℹ️ About {pred_label.capitalize()}")
#             st.write(crop_info[pred_label]["desc"])
#             st.image(crop_info[pred_label]["image"], caption=pred_label.capitalize(), use_column_width=True)

# # -----------------------
# # Footer
# # -----------------------
# st.markdown("---")
# st.caption(" AI for Agriculture 🌱")












# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # -----------------------
# # Page Config
# # -----------------------
# st.set_page_config(
#     page_title="🌾 Crop Recommendation System",
#     page_icon="🌱",
#     layout="wide"
# )

# # -----------------------
# # Custom CSS
# # -----------------------
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

#     html, body, [class*="css"] {
#         font-family: 'Poppins', sans-serif;
#     }

#     /* ✅ Background Image for Main App */
#     .stApp {
#         background: url("https://media.istockphoto.com/id/1091385068/photo/farmer-at-work-under-storm.jpg?s=612x612&w=0&k=20&c=dE8yaxZHn2mgUwZ36fnswTbl7nk7fEr2Ykv54iqGcPc=") no-repeat center center fixed;
#         background-size: cover;
#     }

#     /* ✅ Transparent white card for content */
#     .main {
#         background: rgba(255, 255, 255, 0.9);
#         padding: 25px;
#         border-radius: 15px;
#         box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
#     }

#     h1 {
#         color: #2e7d32 !important;
#         text-align: center;
#         font-weight: 700;
#         padding: 10px;
#         background: rgba(255, 255, 255, 0.7);
#         border-radius: 12px;
#     }

#     h2, h3 {
#         color: #1b5e20 !important;
#         font-weight: 600;
#     }

#     /* ✅ Buttons */
#     .stButton>button {
#         background-color: #2e7d32;
#         color: white;
#         border-radius: 10px;
#         font-size: 18px;
#         padding: 0.6em 1.2em;
#         transition: 0.3s;
#         border: none;
#         box-shadow: 0px 3px 10px rgba(0,0,0,0.2);
#     }
#     .stButton>button:hover {
#         background-color: #1b5e20;
#         transform: scale(1.05);
#     }

#     /* ✅ Sidebar Styling - Black Theme */
#     [data-testid="stSidebar"] {
#         background: #000000;
#         color: #ffffff;
#     }
#     [data-testid="stSidebar"] h1, 
#     [data-testid="stSidebar"] h2, 
#     [data-testid="stSidebar"] h3, 
#     [data-testid="stSidebar"] p, 
#     [data-testid="stSidebar"] label {
#         color: #00e676 !important; /* Green text */
#     }
#     [data-testid="stSidebar"] .stRadio label {
#         color: #ffffff !important;
#     }

#     /* ✅ Progress bar color */
#     .stProgress .st-bo {
#         background-color: #81c784;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # -----------------------
# # Load ML Model
# # -----------------------
# model = joblib.load("crop_recommendation_model.joblib")
# label_encoder = joblib.load("label_encoder.joblib")

# # -----------------------
# # Crop Info
# # -----------------------
# crop_info = {
#     "rice": {
#         "desc": "Rice is a staple food crop grown in waterlogged fields. Requires high rainfall and humidity.",
#         "image": "https://unsplash.com/photos/closeup-of-mixed-rice-SWbjWvvubZM"
#     },
#     "wheat": {
#         "desc": "Wheat is a temperate climate crop requiring moderate rainfall and well-drained soil.",
#         "image": "https://unsplash.com/photos/wheat-field-y4xZxzN754M"
#     },
#     "maize": {
#         "desc": "Maize (corn) is versatile, grown in warm climate with moderate rainfall.",
#         "image": "https://unsplash.com/photos/a-close-up-of-a-corn-on-the-cob-gLDElXUHwNg"
#     },
#     "chickpea": {
#         "desc": "Chickpeas need dry and cool climate with minimal rainfall.",
#         "image": "https://unsplash.com/photos/a-bowl-of-popcorn-on-a-blue-cloth-SAOO2A95Vpc"
#     },
#     "mungbean": {
#         "desc": "Mung beans grow well in warm climate with moderate rainfall.",
#         "image": "https://unsplash.com/photos/plants-grow-on-a-sandy-beach-at-night-0P7-VRPKq4U"
#     },
#     "lentil": {
#         "desc": "Lentils prefer cooler climate and well-drained loamy soil.",
#         "image": "https://unsplash.com/photos/a-wooden-table-topped-with-bowls-of-food-MLp_rPW9G00"
#     },
# }

# # -----------------------
# # Title
# # -----------------------
# st.title("🌾 AI-Powered Crop Recommendation System")

# # -----------------------
# # Sidebar Navigation
# # -----------------------
# st.sidebar.title("📌 Navigation")
# st.sidebar.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=120)  # 🌱 Crop icon under nav
# section = st.sidebar.radio("Go to", ["Dataset Explorer", "Model Information", "Crop Recommendation"])

# # -----------------------
# # Dataset Explorer
# # -----------------------
# if section == "Dataset Explorer":
#     st.header("📊 Dataset Explorer")
#     try:
#         df = pd.read_csv("crop_recommendation.csv")
#         st.subheader("🔍 Preview of Dataset")
#         st.dataframe(df.head())

#         st.subheader("📈 Feature Distribution")
#         feature = st.selectbox("Choose a feature", df.columns[:-1])
#         fig, ax = plt.subplots()
#         sns.histplot(df[feature], kde=True, ax=ax)
#         st.pyplot(fig)
#     except FileNotFoundError:
#         st.warning("Dataset not found. Please upload crop_recommendation.csv")

# # -----------------------
# # Model Info
# # -----------------------
# elif section == "Model Information":
#     st.header("🤖 Model Information")

#     st.write("""
#     - **Algorithm**: Random Forest Classifier  
#     - **Features Used**:  
#         - Nitrogen (N)  
#         - Phosphorus (P)  
#         - Potassium (K)  
#         - Temperature (°C)  
#         - Humidity (%)  
#         - pH Value  
#         - Rainfall (mm)  
#     - **Output**: Recommended Crop  
#     """)

#     st.success("✅ Model and Label Encoder Loaded Successfully!")

# # -----------------------
# # Crop Recommendation
# # -----------------------
# elif section == "Crop Recommendation":
#     st.header("🌱 Crop Recommendation")

#     col1, col2, col3 = st.columns(3)
#     with col1:
#         N = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0)
#         ph = st.number_input("pH Value", 0.0, 14.0, 6.5)
#     with col2:
#         P = st.number_input("Phosphorus (P)", 0.0, 200.0, 50.0)
#         humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
#     with col3:
#         K = st.number_input("Potassium (K)", 0.0, 200.0, 50.0)
#         rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

#     temperature = st.slider("🌡️ Temperature (°C)", 0.0, 60.0, 25.0)

#     if st.button("🔍 Recommend Crop"):
#         X = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=float)
#         pred_id = int(model.predict(X)[0])
#         proba = model.predict_proba(X)[0]
#         classes = label_encoder.classes_.tolist()
#         pred_label = classes[pred_id]

#         st.success(f"🌟 Recommended Crop: {pred_label.capitalize()}")

#         st.write("### 📊 Prediction Confidence")
#         top3_idx = np.argsort(proba)[::-1][:3]
#         for i in top3_idx:
#             st.progress(proba[i])
#             st.write(f"**{classes[i]}** → {proba[i]:.2%}")

#         if pred_label in crop_info:
#             st.subheader(f"ℹ️ About {pred_label.capitalize()}")
#             st.write(crop_info[pred_label]["desc"])
#             st.image(crop_info[pred_label]["image"], caption=pred_label.capitalize(), use_column_width=True)

# # -----------------------
# # Footer
# # -----------------------
# st.markdown("---")
# st.caption("🌱 AI for Agriculture ")







# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # -----------------------
# # Page Config
# # -----------------------
# st.set_page_config(
#     page_title="🌾 Crop Recommendation System",
#     page_icon="🌱",
#     layout="wide"
# )

# # -----------------------
# # Custom CSS
# # -----------------------
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

#     html, body, [class*="css"] {
#         font-family: 'Poppins', sans-serif;
#     }

#     /* ✅ Background Image for Main App */
#     .stApp {
#         background: url("https://media.istockphoto.com/id/1091385068/photo/farmer-at-work-under-storm.jpg?s=612x612&w=0&k=20&c=dE8yaxZHn2mgUwZ36fnswTbl7nk7fEr2Ykv54iqGcPc=") no-repeat center center fixed;
#         background-size: cover;
#     }

#     /* ✅ Transparent white card for content */
#     .main {
#         background: rgba(255, 255, 255, 0.9);
#         padding: 25px;
#         border-radius: 15px;
#         box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
#     }

#     h1 {
#         color: #2e7d32 !important;
#         text-align: center;
#         font-weight: 700;
#         padding: 10px;
#         background: rgba(255, 255, 255, 0.7);
#         border-radius: 12px;
#     }

#     h2, h3 {
#         color: #1b5e20 !important;
#         font-weight: 600;
#     }

#     /* ✅ Buttons */
#     .stButton>button {
#         background-color: #2e7d32;
#         color: white;
#         border-radius: 10px;
#         font-size: 18px;
#         padding: 0.6em 1.2em;
#         transition: 0.3s;
#         border: none;
#         box-shadow: 0px 3px 10px rgba(0,0,0,0.2);
#     }
#     .stButton>button:hover {
#         background-color: #1b5e20;
#         transform: scale(1.05);
#     }

#     /* ✅ Sidebar Styling - Black Theme */
#     [data-testid="stSidebar"] {
#         background: #000000;
#         color: #ffffff;
#     }
#     [data-testid="stSidebar"] h1, 
#     [data-testid="stSidebar"] h2, 
#     [data-testid="stSidebar"] h3, 
#     [data-testid="stSidebar"] p, 
#     [data-testid="stSidebar"] label {
#         color: #00e676 !important; /* Green text */
#     }
#     [data-testid="stSidebar"] .stRadio label {
#         color: #ffffff !important;
#     }

#     /* ✅ Progress bar color */
#     .stProgress .st-bo {
#         background-color: #81c784;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # -----------------------
# # Load ML Model
# # -----------------------
# model = joblib.load("crop_recommendation_model.joblib")
# label_encoder = joblib.load("label_encoder.joblib")

# # -----------------------
# # Crop Info
# # -----------------------
# crop_info = {
#     "rice": {
#         "desc": "Rice is a staple food crop grown in waterlogged fields. Requires high rainfall and humidity.",
#         "image": "https://unsplash.com/photos/closeup-of-mixed-rice-SWbjWvvubZM"
#     },
#     "wheat": {
#         "desc": "Wheat is a temperate climate crop requiring moderate rainfall and well-drained soil.",
#         "image": "https://unsplash.com/photos/wheat-field-y4xZxzN754M"
#     },
#     "maize": {
#         "desc": "Maize (corn) is versatile, grown in warm climate with moderate rainfall.",
#         "image": "https://unsplash.com/photos/a-close-up-of-a-corn-on-the-cob-gLDElXUHwNg"
#     },
#     "chickpea": {
#         "desc": "Chickpeas need dry and cool climate with minimal rainfall.",
#         "image": "https://unsplash.com/photos/a-bowl-of-popcorn-on-a-blue-cloth-SAOO2A95Vpc"
#     },
#     "mungbean": {
#         "desc": "Mung beans grow well in warm climate with moderate rainfall.",
#         "image": "https://unsplash.com/photos/plants-grow-on-a-sandy-beach-at-night-0P7-VRPKq4U"
#     },
#     "lentil": {
#         "desc": "Lentils prefer cooler climate and well-drained loamy soil.",
#         "image": "https://unsplash.com/photos/a-wooden-table-topped-with-bowls-of-food-MLp_rPW9G00"
#     },
# }

# # -----------------------
# # Title
# # -----------------------
# st.title("🌾 AI-Powered Crop Recommendation System")

# # -----------------------
# # Sidebar Navigation
# # -----------------------
# st.sidebar.title("📌 Navigation")
# st.sidebar.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=120)  
# st.sidebar.image("https://vodenglish.news/wp-content/uploads/2022/07/signal-2022-07-05-045019_002.jpeg", caption="Farmer", use_column_width=True)

# section = st.sidebar.radio("Go to", ["Dataset Explorer", "Model Information", "Crop Information", "Crop Recommendation"])

# # -----------------------
# # Dataset Explorer
# # -----------------------
# if section == "Dataset Explorer":
#     st.header("📊 Dataset Explorer")
#     try:
#         df = pd.read_csv("crop_recommendation.csv")
#         st.subheader("🔍 Preview of Dataset")
#         st.dataframe(df.head())

#         st.subheader("📈 Feature Distribution")
#         feature = st.selectbox("Choose a feature", df.columns[:-1])
#         fig, ax = plt.subplots()
#         sns.histplot(df[feature], kde=True, ax=ax)
#         st.pyplot(fig)
#     except FileNotFoundError:
#         st.warning("Dataset not found. Please upload crop_recommendation.csv")

# # -----------------------
# # Model Info
# # -----------------------
# elif section == "Model Information":
#     st.header("🤖 Model Information")

#     st.write("""
#     - **Algorithm**: Random Forest Classifier  
#     - **Features Used**:  
#         - Nitrogen (N)  
#         - Phosphorus (P)  
#         - Potassium (K)  
#         - Temperature (°C)  
#         - Humidity (%)  
#         - pH Value  
#         - Rainfall (mm)  
#     - **Output**: Recommended Crop  
#     """)

#     st.success("✅ Model and Label Encoder Loaded Successfully!")

# # -----------------------
# # Crop Information
# # -----------------------
# elif section == "Crop Information":
#     st.header("🌱 Crop Information")
#     for crop, details in crop_info.items():
#         st.subheader(crop.capitalize())
#         st.write(details["desc"])
#         st.image(details["image"], caption=crop.capitalize(), use_column_width=True)
#         st.markdown("---")

# # -----------------------
# # Crop Recommendation
# # -----------------------
# elif section == "Crop Recommendation":
#     st.header("🌱 Crop Recommendation")

#     col1, col2, col3 = st.columns(3)
#     with col1:
#         N = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0)
#         ph = st.number_input("pH Value", 0.0, 14.0, 6.5)
#     with col2:
#         P = st.number_input("Phosphorus (P)", 0.0, 200.0, 50.0)
#         humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
#     with col3:
#         K = st.number_input("Potassium (K)", 0.0, 200.0, 50.0)
#         rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

#     temperature = st.slider("🌡️ Temperature (°C)", 0.0, 60.0, 25.0)

#     if st.button("🔍 Recommend Crop"):
#         X = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=float)
#         pred_id = int(model.predict(X)[0])
#         proba = model.predict_proba(X)[0]
#         classes = label_encoder.classes_.tolist()
#         pred_label = classes[pred_id]

#         st.success(f"🌟 Recommended Crop: {pred_label.capitalize()}")

#         st.write("### 📊 Prediction Confidence")
#         top3_idx = np.argsort(proba)[::-1][:3]
#         for i in top3_idx:
#             st.progress(proba[i])
#             st.write(f"**{classes[i]}** → {proba[i]:.2%}")

#         if pred_label in crop_info:
#             st.subheader(f"ℹ️ About {pred_label.capitalize()}")
#             st.write(crop_info[pred_label]["desc"])
#             st.image(crop_info[pred_label]["image"], caption=pred_label.capitalize(), use_column_width=True)

# # -----------------------
# # Footer
# # -----------------------
# st.markdown("---")
# st.caption("🌱 AI for Agriculture ")







# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # -----------------------
# # Page Config
# # -----------------------
# st.set_page_config(
#     page_title="🌾 Crop Recommendation System",
#     page_icon="🌱",
#     layout="wide"
# )

# # -----------------------
# # Custom CSS
# # -----------------------
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

#     html, body, [class*="css"] {
#         font-family: 'Poppins', sans-serif;
#     }

#     /* ✅ Background Image */
#     .stApp {
#         background: url("https://media.istockphoto.com/id/1091385068/photo/farmer-at-work-under-storm.jpg?s=612x612&w=0&k=20&c=dE8yaxZHn2mgUwZ36fnswTbl7nk7fEr2Ykv54iqGcPc=") no-repeat center center fixed;
#         background-size: cover;
#     }

#     /* ✅ Main Content Card */
#     .main {
#         background: rgba(223, 208, 184, 0.95);
#         padding: 30px;
#         border-radius: 18px;
#         box-shadow: 0px 6px 18px rgba(0,0,0,0.25);
#     }

#     /* ✅ Headings */
#     h1, h2, h3 {
#         color: #DFD0B8 !important;
#         background: #222831;
#         padding: 10px 15px;
#         border-radius: 12px;
#         font-weight: 700;
#         margin: 25px 0 20px 0 !important;  /* added space above & below */
#     }

#     /* ✅ Buttons */
#     .stButton>button {
#         background-color: #393E46;
#         color: #DFD0B8;
#         border-radius: 12px;
#         font-size: 18px;
#         padding: 0.6em 1.2em;
#         transition: 0.3s;
#         border: none;
#         box-shadow: 0px 3px 10px rgba(0,0,0,0.2);
#     }
#     .stButton>button:hover {
#         background-color: #948979;
#         color: #222831;
#         transform: scale(1.05);
#     }

#     /* ✅ Sidebar */
#     [data-testid="stSidebar"] {
#         background: #222831;
#     }
#     [data-testid="stSidebar"] h1,
#     [data-testid="stSidebar"] h2,
#     [data-testid="stSidebar"] h3,
#     [data-testid="stSidebar"] p,
#     [data-testid="stSidebar"] label {
#         color: #DFD0B8 !important;
#     }

#     /* ✅ Crop Info Notes */
#     .note {
#         background: #DFD0B8;
#         padding: 20px;
#         border-radius: 12px;
#         box-shadow: 4px 6px 12px rgba(0,0,0,0.2);
#         border-left: 8px solid #393E46;
#         min-height: 150px;
#         margin-bottom: 20px;
#     }
#     .note h3 {
#         color: #222831 !important;
#         margin-bottom: 10px;
#     }
#     .note p {
#         color: #393E46;
#         font-size: 15px;
#     }

#     /* ✅ Custom Recommendation Box */
#     .recommend-box {
#         background: #393E46;
#         color: #DFD0B8;
#         padding: 20px;
#         border-radius: 14px;
#         font-size: 20px;
#         font-weight: bold;
#         text-align: center;
#         box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
#         margin: 20px 0;
#     }

#     /* ✅ Progress Bar */
#     .stProgress .st-bo {
#         background-color: #948979;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # -----------------------
# # Load Model
# # -----------------------
# model = joblib.load("crop_recommendation_model.joblib")
# label_encoder = joblib.load("label_encoder.joblib")

# # -----------------------
# # Crop Info
# # -----------------------
# crop_info = {
#     "rice": "Rice is a staple food crop grown in waterlogged fields. Requires high rainfall and humidity.",
#     "wheat": "Wheat is a temperate climate crop requiring moderate rainfall and well-drained soil.",
#     "maize": "Maize (corn) is versatile, grown in warm climate with moderate rainfall.",
#     "chickpea": "Chickpeas need dry and cool climate with minimal rainfall.",
#     "mungbean": "Mung beans grow well in warm climate with moderate rainfall.",
#     "lentil": "Lentils prefer cooler climate and well-drained loamy soil.",
# }

# # -----------------------
# # Title
# # -----------------------
# st.title("🌾 AI-Powered Crop Recommendation System")

# # -----------------------
# # Sidebar
# # -----------------------
# st.sidebar.title("📌 Navigation")
# st.sidebar.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=120)
# st.sidebar.image(
#     "https://vodenglish.news/wp-content/uploads/2022/07/signal-2022-07-05-045019_002.jpeg",
#     caption="Farmer",
#     use_container_width=True
# )

# section = st.sidebar.radio("Go to", ["Dataset Explorer", "Model Information", "Crop Information", "Crop Recommendation"])

# # -----------------------
# # Dataset Explorer
# # -----------------------
# if section == "Dataset Explorer":
#     st.header("📊 Dataset Explorer")
#     try:
#         df = pd.read_csv("crop_recommendation.csv")
#         st.subheader("🔍 Preview of Dataset")
#         st.dataframe(df.head(), use_container_width=True)

#         st.subheader("📈 Feature Distribution")
#         feature = st.selectbox("Choose a feature", df.columns[:-1])
#         fig, ax = plt.subplots()
#         sns.histplot(df[feature], kde=True, ax=ax)
#         st.pyplot(fig)
#     except FileNotFoundError:
#         st.warning("Dataset not found. Please upload crop_recommendation.csv")

# # -----------------------
# # Model Info
# # -----------------------
# elif section == "Model Information":
#     st.header("🤖 Model Information")

#     st.markdown("""
#     <div class='note'>
#         <h3>Model Details</h3>
#         <p>
#         - <b>Algorithm</b>: Random Forest Classifier <br>
#         - <b>Features</b>: N, P, K, Temperature, Humidity, pH, Rainfall <br>
#         - <b>Output</b>: Recommended Crop
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

# # -----------------------
# # Crop Information
# # -----------------------
# elif section == "Crop Information":
#     st.header("🌱 Crop Information")

#     # Display in 2-column grid
#     cols = st.columns(2)
#     i = 0
#     for crop, desc in crop_info.items():
#         with cols[i % 2]:
#             st.markdown(f"""
#             <div class='note'>
#                 <h3>{crop.capitalize()}</h3>
#                 <p>{desc}</p>
#             </div>
#             """, unsafe_allow_html=True)
#         i += 1

# # -----------------------
# # Crop Recommendation
# # -----------------------
# elif section == "Crop Recommendation":
#     st.header("🌱 Crop Recommendation")

#     col1, col2, col3 = st.columns(3)
#     with col1:
#         N = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0)
#         ph = st.number_input("pH Value", 0.0, 14.0, 6.5)
#     with col2:
#         P = st.number_input("Phosphorus (P)", 0.0, 200.0, 50.0)
#         humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
#     with col3:
#         K = st.number_input("Potassium (K)", 0.0, 200.0, 50.0)
#         rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

#     temperature = st.slider("🌡️ Temperature (°C)", 0.0, 60.0, 25.0)

#     if st.button("🔍 Recommend Crop"):
#         feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
#         X = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_names)

#         pred_id = int(model.predict(X)[0])
#         proba = model.predict_proba(X)[0]
#         classes = label_encoder.classes_.tolist()
#         pred_label = classes[pred_id]

#         # Custom styled recommendation box
#         st.markdown(f"""
#         <div class="recommend-box">
#             🌟 Recommended Crop: {pred_label.capitalize()}
#         </div>
#         """, unsafe_allow_html=True)

#         st.write("### 📊 Prediction Confidence")
#         top3_idx = np.argsort(proba)[::-1][:3]
#         for i in top3_idx:
#             st.progress(proba[i])
#             st.write(f"**{classes[i]}** → {proba[i]:.2%}")

#         if pred_label in crop_info:
#             st.markdown(f"""
#             <div class='note'>
#                 <h3>ℹ️ About {pred_label.capitalize()}</h3>
#                 <p>{crop_info[pred_label]}</p>
#             </div>
#             """, unsafe_allow_html=True)

# # -----------------------
# # Footer
# # -----------------------
# st.markdown("---")
# st.caption("🌱 AI for Agriculture ")









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
