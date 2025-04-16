import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load dataset
unzip_dir = r"C:\Users\cscpr\Desktop\PAPER\ANOMALY DETECTION CONFERENCE 4\SGSMA_Competiton 2024_PMU_DATA"
for root, dirs, files in os.walk(unzip_dir):
    for file in files:
        if file.endswith(".csv"):
            data_path = os.path.join(root, file)
            break

df = pd.read_csv(data_path)

# Data Preprocessing
df.fillna(method='ffill', inplace=True)
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.iloc[:, 1:]), columns=df.columns[1:])  # Assuming first column is timestamp

# Create Time-Series Sequences for LSTM
sequence_length = 50
X = [df_scaled.iloc[i:i + sequence_length].values for i in range(len(df_scaled) - sequence_length)]
X = np.array(X)

# LSTM Autoencoder Model
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(sequence_length, X.shape[2])),
    Dropout(0.2),
    LSTM(32, activation='relu', return_sequences=False),
    Dense(32, activation='relu'),
    Dense(X.shape[2])  # Output layer same as input features
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, X[:, -1, :], epochs=1, batch_size=32, validation_split=0.1, verbose=1)

# Compute Reconstruction Errors
X_pred = model.predict(X)
reconstruction_errors = np.mean(np.abs(X[:, -1, :] - X_pred), axis=1)

# Apply Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(reconstruction_errors.reshape(-1, 1))
anomaly_labels = iso_forest.predict(reconstruction_errors.reshape(-1, 1))  # -1 = anomaly, 1 = normal

# Set threshold for anomalies
threshold = np.percentile(reconstruction_errors, 95)  # Top 5% as anomalies
predicted_anomalies = (reconstruction_errors > threshold).astype(int)  # 1 = anomaly, 0 = normal

# Convert Isolation Forest labels for comparison
true_anomalies = (anomaly_labels == -1).astype(int)  # Convert -1 (anomaly) to 1

# Calculate precision, recall, and F1-score
precision = precision_score(true_anomalies, predicted_anomalies)
recall = recall_score(true_anomalies, predicted_anomalies)
f1 = f1_score(true_anomalies, predicted_anomalies)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# ------------------------
# Step 1: Plot Reconstruction Errors with Anomalies
# ------------------------
plt.figure(figsize=(12, 6))
sns.lineplot(x=np.arange(len(reconstruction_errors)), y=reconstruction_errors, label='Reconstruction Error')
sns.scatterplot(x=np.where(predicted_anomalies == 1)[0], y=reconstruction_errors[predicted_anomalies == 1], color='red', label='Detected Anomalies')
plt.xlabel("Time")
plt.ylabel("Reconstruction Error")
plt.title("Reconstruction Error with Anomalies")
plt.legend()
plt.show()

# ------------------------
# Step 2: Plot Precision, Recall, and F1-score in a single graph
# ------------------------
plt.figure(figsize=(8, 5))
metrics = ["Precision", "Recall", "F1-score"]
scores = [precision, recall, f1]

plt.bar(metrics, scores, color=['blue', 'green', 'red'])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Performance Metrics")

# Display the values on top of the bars
for i, v in enumerate(scores):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)

plt.show()
