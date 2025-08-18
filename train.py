import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Paths
DATA_PATH = Path("Crop_recommendation.csv")
MODEL_PATH = Path("crop_recommendation_model.joblib")
ENCODER_PATH = Path("label_encoder.joblib")

# Load dataset
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train model
print("🚀 Training model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2%}")

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model and label encoder
joblib.dump(model, MODEL_PATH)
joblib.dump(label_encoder, ENCODER_PATH)

print(f"\n💾 Model saved to: {MODEL_PATH}")
print(f"💾 Label encoder saved to: {ENCODER_PATH}")
