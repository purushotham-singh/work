import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, medfilt
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# Suppress Tkinter deprecation warning
os.environ['TK_SILENCE_DEPRECATION'] = '1'

# Step 1: Get user input for Tx and Tr frequencies
Tx_khz = float(input("Enter the transmitter (Tx) frequency in kHz: "))
Tr_khz = float(input("Enter the receiver (Tr) frequency in kHz: "))

# Step 2: Select CSV file
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select a CSV file", filetypes=[["CSV files", "*.csv"]])
if not file_path:
    raise ValueError("No file selected. Please select a CSV file.")

data = pd.read_csv(file_path)

# Step 3: Extract time and amplitude
try:
    time = data['time'].values  # Time in µs
    amplitude = data['amplitude'].values  # Amplitude in V
except KeyError:
    time = data.iloc[:, 0].values  # First column as time
    amplitude = data.iloc[:, 1].values  # Second column as amplitude

# Apply median filtering to reduce noise
amplitude = medfilt(amplitude, kernel_size=5)

N = len(amplitude)

# Step 4: Compute signal parameters
T0 = 0  # Initial time is always 0
Tp_idx = np.argmax(np.abs(amplitude))
Tp = time[Tp_idx]
Ap = amplitude[Tp_idx]

threshold = 0.05 * np.max(np.abs(amplitude))  # Adjusted threshold
peaks, _ = find_peaks(np.abs(amplitude), height=threshold)
T1 = time[peaks[0]] if len(peaks) > 0 else Tp
TOF = T1 - T0

post_peak_data = np.abs(amplitude[Tp_idx:])
Tr_idx_relative = np.where(post_peak_data < threshold)[0]
Tr = time[Tp_idx + Tr_idx_relative[0]] if len(Tr_idx_relative) > 0 else time[-1]
Ar = amplitude[Tp_idx + Tr_idx_relative[0]] if len(Tr_idx_relative) > 0 else amplitude[-1]

# Step 5: Compute FFT parameters
time_step = np.mean(np.diff(time)) * 1e-6  # Convert to seconds
if np.any(np.diff(time) <= 0):
    raise ValueError("Time values must be strictly increasing for FFT.")

fft_data = fft(amplitude)
fft_magnitude_values = np.abs(fft_data[:N//2])
frequencies = fftfreq(N, time_step)[:N//2] / 1e3  # Convert to kHz

Tx_idx = (np.abs(frequencies - Tx_khz)).argmin()
fft_frequency = frequencies[Tx_idx]
fft_magnitude = fft_magnitude_values[Tx_idx]

# Step 6: Feature Engineering
features = np.array([[T1, TOF, Tp, Ap, Tr, Ar, fft_frequency, fft_magnitude]])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 7: Machine Learning Models
svm_model = SVC()
rf_model = RandomForestClassifier()
rf_model.fit(features_scaled, [0])

defect_cnn = keras.Sequential([
    keras.Input(shape=(features_scaled.shape[1], 1)),
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
defect_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 8: Validation & Results
def check_range(value, min_val, max_val, name):
    status = "✅ Within Range" if min_val <= value <= max_val else "❌ Out of Range"
    return f"{name}: {value:.2f} µs {status}"

print("\nValidation Results:")
print(f"T0: {T0:.2f} µs (Expected: 0 µs)")
print(check_range(T1, 140, 260, "T1 (Rise Time)"))
print(f"TOF: {TOF:.2f} µs (T1 should be equal to TOF)")
print(check_range(Tp, 300, 400, "Tp (Peak Time)"))
print(f"Ap: {Ap:.3f} V")
print(check_range(Tr, 400, 600, "Tr (Ring-down Time)"))
print(f"Ar: {Ar:.3f} V")
print(f"FFT Frequency: {fft_frequency:.2f} kHz")
print(f"FFT Magnitude: {fft_magnitude:.2f} ({'✅ Good Zone' if fft_magnitude < 100 else '❌ Defective Zone'})")

# Step 9: Plot Signal
plt.figure(figsize=(10, 6))
plt.plot(time, amplitude, label='Signal')
plt.axvline(T0, color='r', linestyle='--', label='T0')
plt.axvline(T1, color='g', linestyle='--', label='T1')
plt.axvline(Tp, color='b', linestyle='--', label='Tp')
plt.axvline(Tr, color='m', linestyle='--', label='Tr')
plt.xlabel('Time (µs)')
plt.ylabel('Amplitude (V)')
plt.title('Ultrasonic Signal with Key Time Points')
plt.legend()
plt.grid()
plt.show()