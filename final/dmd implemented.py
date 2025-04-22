# Import necessary libraries
import numpy as np                              # Numerical computing library
import pandas as pd                             # Data manipulation and analysis library
import matplotlib.pyplot as plt                 # Plotting library
import seaborn as sns                           # Statistical data visualization library
import os                                       # OS module to handle file paths and directory walking
from sklearn.preprocessing import MinMaxScaler  # Scaler to normalize features to [0, 1] range

# -----------------------------
# Load dataset
# -----------------------------
unzip_dir = r"C:\Users\cscpr\Desktop\PAPER\ANOMALY DETECTION CONFERENCE 4\SGSMA_Competiton 2024_PMU_DATA"
# Traverse directory tree to find first CSV file in folder
for root, dirs, files in os.walk(unzip_dir):
    for file in files:
        if file.endswith(".csv"):              # Check if file is a CSV
            data_path = os.path.join(root, file)  # Get full path to the file
            break

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(data_path)

# Forward-fill missing values (NaNs) to handle gaps in the data
df.fillna(method='ffill', inplace=True)

# -----------------------------
# Scale data (skip timestamp column)
# -----------------------------
scaler = MinMaxScaler()                        # Create MinMaxScaler instance
# Apply scaling to all columns except the first (timestamp), create new DataFrame with same column names
df_scaled = pd.DataFrame(scaler.fit_transform(df.iloc[:, 1:]), columns=df.columns[1:])
# Transpose the scaled data for DMD: rows = features, columns = time steps
X_full = df_scaled.T.to_numpy()

# -----------------------------
# Perform Dynamic Mode Decomposition (DMD)
# -----------------------------
def perform_dmd(X, r):
    # Split data matrix into X1 and X2 (time-shifted snapshots)
    X1 = X[:, :-1]
    X2 = X[:, 1:]

    # Perform Singular Value Decomposition (SVD)
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    # Truncate to rank-r
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vr = Vh.conj().T[:, :r]

    # Construct low-rank linear operator A_tilde
    A_tilde = Ur.T @ X2 @ Vr @ np.linalg.inv(Sr)
    # Compute eigenvalues and eigenvectors
    eigs, W = np.linalg.eig(A_tilde)
    # Compute DMD modes
    Phi = X2 @ Vr @ np.linalg.inv(Sr) @ W

    # Convert discrete eigenvalues to continuous-time frequencies
    omega = np.log(eigs)
    time_points = X.shape[1]                    # Total number of time steps
    t = np.arange(time_points)                  # Time vector

    # Initial condition (first snapshot)
    x0 = X[:, 0]
    # Solve for mode amplitudes b using least squares
    b = np.linalg.lstsq(Phi, x0, rcond=None)[0]

    # Reconstruct time dynamics using exponential evolution of modes
    time_dynamics = np.array([
        b * np.exp(omega * ti) for ti in t
    ]).T

    # Reconstruct the original data matrix using DMD approximation
    X_dmd = (Phi @ time_dynamics).real          # Take real part (imaginary due to numerical errors)
    return X_dmd

# -----------------------------
# Apply DMD and reconstruct
# -----------------------------
rank = 100                                      # Set rank for low-rank approximation
X_dmd_reconstructed = perform_dmd(X_full, rank) # Perform DMD and get reconstruction

# -----------------------------
# Compute Reconstruction Error
# -----------------------------
# Compute mean absolute reconstruction error across all features at each time step
reconstruction_errors = np.mean(np.abs(X_full - X_dmd_reconstructed), axis=0)

# -----------------------------
# Anomaly Detection
# -----------------------------
threshold = np.percentile(reconstruction_errors, 95)  # Set threshold at 95th percentile
anomalies = (reconstruction_errors > threshold).astype(int)  # Mark points exceeding threshold as anomalies

# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(12, 6))                     # Set plot size
# Plot reconstruction error over time
sns.lineplot(x=np.arange(len(reconstruction_errors)), y=reconstruction_errors, label='DMD Reconstruction Error')
# Mark anomalies with red dots
sns.scatterplot(x=np.where(anomalies == 1)[0], y=reconstruction_errors[anomalies == 1], color='red', label='Anomalies')
# Draw threshold line
plt.axhline(threshold, color='orange', linestyle='--', label='Threshold')
# Label axes and title
plt.xlabel("Time")
plt.ylabel("Reconstruction Error")
plt.title("DMD Reconstruction Error with Anomalies")
# Show legend and fit layout
plt.legend()
plt.tight_layout()
# Display the plot
plt.show()
