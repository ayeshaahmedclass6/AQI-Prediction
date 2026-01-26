import os
import hopsworks
import pandas as pd
import numpy as np
import joblib
import shutil
import time
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# TensorFlow for ANN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- 1. SETUP & LOGIN ---
load_dotenv()
project = hopsworks.login(project="ayeshaahmedAQI")
fs = project.get_feature_store()

# --- 2. FETCH DATA ---
print("📡 Fetching data from Feature Store...")
fg = fs.get_feature_group(name="aqi_features", version=2)
df = fg.read()

# --- 3. DATA PREPARATION (The NaN-Proof Version) ---
# Drop non-numeric and ensure aqi exists
training_df = df.select_dtypes(include=[np.number]).dropna(subset=["aqi"])

X = training_df.drop(columns=["aqi"])
y = training_df["aqi"]

# 🛠️ CRITICAL FIX: Fill missing values in Features (X) BEFORE scaling
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Scaling for Ridge and ANN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. EXPERIMENT WITH VARIOUS ML MODELS ---

# A. Ridge Regression
print("📈 Training Ridge Regression...")
ridge = Ridge(alpha=1.0).fit(X_train_scaled, y_train)

# B. Ensemble Models
print("🌲 Training Ensemble Models...")
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
gb = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)

# C. Artificial Neural Network (ANN)
print("🧠 Training ANN (TensorFlow)...")
def build_ann(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1) 
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

ann = build_ann(X_train.shape[1])
ann.fit(X_train_scaled, y_train, epochs=50, batch_size=8, verbose=0)

# --- 5. EVALUATE PERFORMANCE ---
models = {
    "Ridge": (ridge.predict(X_test_scaled), ridge),
    "RandomForest": (rf.predict(X_test), rf),
    "GradientBoosting": (gb.predict(X_test), gb),
    "ANN": (ann.predict(X_test_scaled).flatten(), ann)
}

results = {}
for name, (preds, model_obj) in models.items():
    results[name] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds),
        "object": model_obj
    }
    print(f"📊 {name} -> RMSE: {results[name]['RMSE']:.2f}, R2: {results[name]['R2']:.2f}")

# --- 6. STORE TRAINED MODEL ---
best_name = min(results, key=lambda k: results[k]["RMSE"])
print(f"🏆 Best Performing Model: {best_name}")

os.makedirs("model_artifact", exist_ok=True)
if best_name == "ANN":
    results[best_name]["object"].save("model_artifact/saved_model")
else:
    joblib.dump(results[best_name]["object"], "model_artifact/model.pkl")

# Save Preprocessing Tools (Vital for app.py!)
joblib.dump(scaler, "model_artifact/scaler.pkl")
joblib.dump(imputer, "model_artifact/imputer.pkl")

shutil.make_archive("aqi_model", "zip", "model_artifact")

mr = project.get_model_registry()
model_meta = mr.python.create_model(
    name="karachi_aqi_model",
    metrics={"RMSE": float(results[best_name]["RMSE"]), "R2": float(results[best_name]["R2"])},
    description=f"Experimentation winner: {best_name}. Includes Imputer and Scaler."
)
model_meta.save("aqi_model.zip")
print("🔥 SUCCESS: Advanced training complete!")