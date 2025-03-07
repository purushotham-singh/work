import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog

# Ensure Tkinter works in PyCharm
import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'

# Step 1: Ask user for Tx and Tr frequencies in kHz
Tx_khz = float(input("Enter the transmitter (Tx) frequency in kHz: "))
Tr_khz = float(input("Enter the receiver (Tr) frequency in kHz: "))

# Step 2: Prompt user to select a CSV file
root = tk.Tk()
root.withdraw()  # Hide the main tkinter window
file_path = filedialog.askopenfilename(title="Select a CSV file", filetypes=[("CSV files", "*.csv")])
if not file_path:
    raise ValueError("No file selected. Please select a CSV file.")

data = pd.read_csv(file_path)  # Load CSV file

# Extract time and amplitude columns (adjust based on your CSV header)
try:
    time = data['time'].values  # Time in µs
    amplitude = data['amplitude'].values  # Amplitude in V
except KeyError:
    time = data.iloc[:, 0].values  # First column as time
    amplitude = data.iloc[:, 1].values  # Second column as amplitude

N = len(amplitude)  # Number of samples

# Step 3: Compute time-based parameters
T0 = time[0]  # In µs
T0_idx = 0  # Index corresponding to T0
max_amplitude = np.max(np.abs(amplitude))
threshold = 0.05 * max_amplitude  # Adjustable threshold

Tp_idx = np.argmax(np.abs(amplitude))
Tp = time[Tp_idx]  # In µs
Ap = amplitude[Tp_idx]  # In V

peaks, _ = find_peaks(np.abs(amplitude[T0_idx:]), height=threshold)
T1 = time[T0_idx + peaks[0]] if len(peaks) > 0 else Tp
TOF = T1 - T0  # In µs

post_peak_data = np.abs(amplitude[Tp_idx:])
Tr_idx_relative = np.where(post_peak_data < threshold)[0]
Tr = time[Tp_idx + Tr_idx_relative[0]] if len(Tr_idx_relative) > 0 else time[-1]
Ar = amplitude[Tp_idx + Tr_idx_relative[0]] if len(Tr_idx_relative) > 0 else amplitude[-1]

# Step 4: Compute Cycles
signal_segment = amplitude[T0_idx:Tp_idx]
zero_crossings = np.where(np.diff(np.sign(signal_segment)))[0]
cycles = len(zero_crossings) // 2  # Number of cycles

# Step 5: Compute FFT parameters
time_step = np.mean(np.diff(time)) * 1e-6  # Convert to seconds
if np.any(np.diff(time) <= 0):
    raise ValueError("Time values must be strictly increasing for FFT.")

fft_data = fft(amplitude)
fft_magnitude = np.abs(fft_data[:N//2])
frequencies = fftfreq(N, time_step)[:N//2] / 1e3  # Convert Hz to kHz

# Find FFT frequency closest to Tx and Tr
Tx_idx = (np.abs(frequencies - Tx_khz)).argmin()
Tr_idx = (np.abs(frequencies - Tr_khz)).argmin()

fft_frequency = frequencies[Tx_idx]  # Using Tx as reference
fft_magnitude = fft_magnitude[Tx_idx]

# Step 6: Validate and print results
def check_range(value, min_val, max_val, name):
    status = "Within Range" if min_val <= value <= max_val else "Out of Range"
    return f"{name}: {value:.2f} µs ({status})"

print("Validation Results:")
print(f"T0 (Initial Time): {T0:.2f} µs (Expected: 0 µs)")
print(check_range(T1, 140, 260, "T1 (Rise Time)"))
print(f"TOF (Time of Flight): {TOF:.2f} µs (T1 should be equal to TOF)")
print(check_range(Tp, 300, 400, "Tp (Peak Time)"))
print(f"Ap (Peak Amplitude): {Ap:.3f} V")
print(check_range(Tr, 400, 600, "Tr (Ring-down Time)"))
print(f"Ar (Ring-down Amplitude): {Ar:.3f} V")
print(f"Cycles: {cycles}")
print(f"FFT Frequency: {fft_frequency:.2f} kHz")
print(f"FFT Magnitude: {fft_magnitude:.2f} ({'Good Zone' if fft_magnitude < 100 else 'Defected Zone'})")

# Step 7: Plot the signal
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
