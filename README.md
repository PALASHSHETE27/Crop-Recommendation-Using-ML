# 🌾 Crop Recommendation System using Machine Learning

## 📌 Overview

The Crop Recommendation System is a machine learning-based web application that helps farmers and agricultural professionals determine the most suitable crop to cultivate based on soil nutrients and environmental conditions.

The system analyzes key parameters such as Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH level, and rainfall, and recommends the most appropriate crop using a trained machine learning model.

---

## 🚀 Features

* Predicts the best crop based on soil and weather conditions.
* User-friendly Streamlit web interface.
* Machine Learning-powered recommendations.
* Real-time prediction results.
* Trained model saved using Joblib.
* Easy deployment on Streamlit Cloud.

---

## 🛠️ Technologies Used

### Frontend

* HTML
* CSS
* Streamlit

### Backend

* Python

### Machine Learning

* Scikit-learn
* Pandas
* NumPy
* Joblib

### Visualization

* Matplotlib
* Seaborn

---

## 📂 Project Structure

Crop-Recommendation-Using-ML/
│
├── frontend/
│   ├── index.html
│   └── style.css
│
├── api.py
├── streamlit_app.py
├── train.py
├── requirements.txt
├── Crop_recommendation.csv
├── crop_recommendation_model.joblib
├── label_encoder.joblib
└── README.md

---

## 📊 Dataset Features

The model uses the following input parameters:

| Feature     | Description           |
| ----------- | --------------------- |
| N           | Nitrogen Content      |
| P           | Phosphorus Content    |
| K           | Potassium Content     |
| Temperature | Temperature in °C     |
| Humidity    | Relative Humidity (%) |
| pH          | Soil pH Value         |
| Rainfall    | Rainfall in mm        |

---

## 🤖 Machine Learning Model

The crop recommendation model is trained using supervised machine learning techniques on agricultural datasets.

### Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Selection
4. Model Training
5. Model Evaluation
6. Model Serialization using Joblib
7. Deployment using Streamlit

---

## ⚙️ Installation

Clone the repository:


git clone https://github.com/PALASHSHETE27/Crop-Recommendation-Using-ML.git

Navigate to the project directory:


cd Crop-Recommendation-Using-ML

Create a virtual environment:

python -m venv .venv

Activate the environment:

### macOS/Linux

source .venv/bin/activate

### Windows

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

---

## ▶️ Run the Application

Start the Streamlit application:

streamlit run streamlit_app.py

The application will open in your browser automatically.

---

## 🌐 Deployment

The application is deployed using Streamlit Community Cloud.

To deploy:

1. Push the project to GitHub.
2. Connect GitHub with Streamlit Cloud.
3. Select the repository.
4. Set the main file path as:

streamlit_app.py

5. Deployed application.

https://crop-recommendation-using-ml-idegqdrcj5utnpekfyqtfa.streamlit.app/

---

## 📈 Future Enhancements

* Fertilizer Recommendation System
* Disease Detection Module
* Weather Forecast Integration
* Multi-language Support
* Mobile Application Version

---

## 👨‍💻 Author

**Palash Shete**

Pre-Final Year Engineering Student (2026)

Interested in:

* Machine Learning
* Android Development
* Full Stack Development
* Artificial Intelligence

---

## 📄 License

This project is intended for educational and learning purposes.





https://crop-recommendation-using-ml-idegqdrcj5utnpekfyqtfa.streamlit.app/


<img width="1427" height="679" alt="Screenshot 2026-06-01 at 7 13 21 PM" src="https://github.com/user-attachments/assets/496974ad-ba02-4e27-8101-031c43d3d21a" />

<img width="1417" height="614" alt="Screenshot 2026-06-01 at 7 12 39 PM" src="https://github.com/user-attachments/assets/01bfa3d1-3dfe-442e-b268-f2bd364ec98c" />
