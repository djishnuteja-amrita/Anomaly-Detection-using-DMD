# Import necessary libraries
import numpy as np                              # For numerical operations
import pandas as pd                             # For data manipulation
import matplotlib.pyplot as plt                 # For plotting
import seaborn as sns                           # For advanced plotting
import os                                       # To interact with the operating system
from sklearn.preprocessing import MinMaxScaler  # To scale features to [0, 1] range
from sklearn.ensemble import IsolationForest    # Anomaly detection using Isolation Forest
from sklearn.metrics import precision_score, recall_score, f1_score  # Evaluation metrics

# TensorFlow and Keras for deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ------------------------------------
# Load dataset
# ------------------------------------
unzip_dir = r"C:\Users\cscpr\Desktop\PAPER\ANOMALY DETECTION CONFERENCE 4\SGSMA_Competiton 2024_PMU_DATA"
# Walk through directory to find CSV file
for root, dirs, files in os.walk(unzip_dir):
    for file in files:
        if file.endswith(".csv"):              # Check for CSV files
            data_path = os.path.join(root, file)  # Get full file path
            break

# Load CSV file into pandas DataFrame
df = pd.read_csv(data_path)

# ------------------------------------
# Data Preprocessing
# ------------------------------------
df.fillna(method='ffill', inplace=True)        # Forward-fill missing values
scaler = MinMaxScaler()                        # Initialize scaler
# Scale all features except the first column (assumed to be timestamp)
df_scaled = pd.DataFrame(scaler.fit_transform(df.iloc[:, 1:]), columns=df.columns[1:])

# ------------------------------------
# Create Time-Series Sequences for LSTM
# ------------------------------------
sequence_length = 50
# Create overlapping sequences of shape (sequence_length, num_features)
X = [df_scaled.iloc[i:i + sequence_length].values for i in range(len(df_scaled) - sequence_length)]
X = np.array(X)  # Convert to NumPy array for model training

# ------------------------------------
# Define LSTM Autoencoder Model
# ------------------------------------
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(sequence_length, X.shape[2])),  # First LSTM layer
    Dropout(0.2),                         # Dropout for regularization
    LSTM(32, activation='relu', return_sequences=False),  # Second LSTM layer
    Dense(32, activation='relu'),        # Fully connected layer
    Dense(X.shape[2])                    # Output layer to reconstruct last time step
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
# Predict the last time step of each sequence
model.fit(X, X[:, -1, :], epochs=1, batch_size=32, validation_split=0.1, verbose=1)

# ------------------------------------
# Compute Reconstruction Errors
# ------------------------------------
X_pred = model.predict(X)  # Reconstruct last time step
# Compute mean absolute error for each sequence
reconstruction_errors = np.mean(np.abs(X[:, -1, :] - X_pred), axis=1)

# ------------------------------------
# Apply Isolation Forest for Ground Truth Labeling
# ------------------------------------
iso_forest = IsolationForest(contamination=0.05, random_state=42)
# Fit and predict using reconstruction error
iso_forest.fit(reconstruction_errors.reshape(-1, 1))
anomaly_labels = iso_forest.predict(reconstruction_errors.reshape(-1, 1))  # -1 = anomaly

# ------------------------------------
# Threshold-based Anomaly Detection
# ------------------------------------
# Set 95th percentile of reconstruction error as threshold
threshold = np.percentile(reconstruction_errors, 95)
# Predict anomalies using threshold (1 = anomaly, 0 = normal)
predicted_anomalies = (reconstruction_errors > threshold).astype(int)

# Convert Isolation Forest output to binary anomaly labels for comparison
true_anomalies = (anomaly_labels == -1).astype(int)

# ------------------------------------
# Evaluate Model
# ------------------------------------
# Compute precision, recall, and F1 score
precision = precision_score(true_anomalies, predicted_anomalies)
recall = recall_score(true_anomalies, predicted_anomalies)
f1 = f1_score(true_anomalies, predicted_anomalies)

# Print metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# ------------------------------------
# Step 1: Plot Reconstruction Errors with Anomalies
# ------------------------------------
plt.figure(figsize=(12, 6))
sns.lineplot(x=np.arange(len(reconstruction_errors)), y=reconstruction_errors, label='Reconstruction Error')
sns.scatterplot(x=np.where(predicted_anomalies == 1)[0], y=reconstruction_errors[predicted_anomalies == 1], color='red', label='Detected Anomalies')
plt.xlabel("Time")
plt.ylabel("Reconstruction Error")
plt.title("Reconstruction Error with Anomalies")
plt.legend()
plt.show()

# ------------------------------------
# Step 2: Plot Performance Metrics
# ------------------------------------
plt.figure(figsize=(8, 5))
metrics = ["Precision", "Recall", "F1-score"]
scores = [precision, recall, f1]

# Bar plot for evaluation metrics
plt.bar(metrics, scores, color=['blue', 'green', 'red'])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Performance Metrics")

# Annotate bars with score values
for i, v in enumerate(scores):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)

plt.show()
