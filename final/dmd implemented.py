import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Load dataset
# -----------------------------
unzip_dir = r"C:\Users\cscpr\Desktop\PAPER\ANOMALY DETECTION CONFERENCE 4\SGSMA_Competiton 2024_PMU_DATA"
for root, dirs, files in os.walk(unzip_dir):
    for file in files:
        if file.endswith(".csv"):
            data_path = os.path.join(root, file)
            break

df = pd.read_csv(data_path)
df.fillna(method='ffill', inplace=True)

# -----------------------------
# Scale data (skip timestamp column)
# -----------------------------
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.iloc[:, 1:]), columns=df.columns[1:])
X_full = df_scaled.T.to_numpy()  # Shape: (features, time)

# -----------------------------
# Perform Dynamic Mode Decomposition (DMD)
# -----------------------------
def perform_dmd(X, r):
    X1 = X[:, :-1]
    X2 = X[:, 1:]

    # SVD and truncation
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vr = Vh.conj().T[:, :r]

    # Low-rank linear operator
    A_tilde = Ur.T @ X2 @ Vr @ np.linalg.inv(Sr)
    eigs, W = np.linalg.eig(A_tilde)
    Phi = X2 @ Vr @ np.linalg.inv(Sr) @ W

    # Continuous time dynamics (safe version)
    omega = np.log(eigs)
    time_points = X.shape[1]
    t = np.arange(time_points)

    # Initial condition
    x0 = X[:, 0]
    b = np.linalg.lstsq(Phi, x0, rcond=None)[0]

    # Time dynamics
    time_dynamics = np.array([
        b * np.exp(omega * ti) for ti in t
    ]).T

    X_dmd = (Phi @ time_dynamics).real
    return X_dmd

# -----------------------------
# Apply DMD and reconstruct
# -----------------------------
rank = 100
X_dmd_reconstructed = perform_dmd(X_full, rank)

# -----------------------------
# Compute Reconstruction Error
# -----------------------------
reconstruction_errors = np.mean(np.abs(X_full - X_dmd_reconstructed), axis=0)

# -----------------------------
# Anomaly Detection
# -----------------------------
threshold = np.percentile(reconstruction_errors, 95)
anomalies = (reconstruction_errors > threshold).astype(int)

# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(12, 6))
sns.lineplot(x=np.arange(len(reconstruction_errors)), y=reconstruction_errors, label='DMD Reconstruction Error')
sns.scatterplot(x=np.where(anomalies == 1)[0], y=reconstruction_errors[anomalies == 1], color='red', label='Anomalies')
plt.axhline(threshold, color='orange', linestyle='--', label='Threshold')
plt.xlabel("Time")
plt.ylabel("Reconstruction Error")
plt.title("DMD Reconstruction Error with Anomalies")
plt.legend()
plt.tight_layout()
plt.show()
