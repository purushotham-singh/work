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

# Extract time and amplitude columns (adjust based on CSV header)
try:
    time = data['time'].values  # Time in µs
    amplitude = data['amplitude'].values  # Amplitude in V
except KeyError:
    time = data.iloc[:, 0].values  # First column as time
    amplitude = data.iloc[:, 1].values  # Second column as amplitude

# Ensure time is strictly increasing
if np.any(np.diff(time) <= 0):
    raise ValueError("Time values must be strictly increasing for FFT.")

N = len(amplitude)  # Number of samples

# Step 3: Compute time-based parameters (using second code's logic)
T0 = time[0]  # In µs
T0_idx = 0  # Index corresponding to T0

# Find Tp (Peak Time) and Ap (Peak Amplitude)
Tp_idx = np.argmax(np.abs(amplitude))
Tp = time[Tp_idx]  # In µs
Ap = amplitude[Tp_idx]  # In V

# Define Adaptive Threshold (from second code: 10% of peak amplitude)
threshold = 0.1 * Ap  # 10% of Peak Amplitude

# Find T1 (First Peak After T0)
peaks, _ = find_peaks(np.abs(amplitude[T0_idx:]), height=threshold)
if len(peaks) > 0:
    T1_idx = T0_idx + peaks[0]
    T1 = time[T1_idx]
else:
    T1 = Tp  # If no early peak, set T1 to Tp

# Compute TOF
TOF = T1 - T0  # In µs

# Find Tr (Ring-down Time) - Picking the last valid drop below threshold (from second code)
min_Tr_range = Tp + 400  # Lower bound in µs
max_Tr_range = Tp + 600  # Upper bound in µs

post_peak_data = np.abs(amplitude[Tp_idx:])
time_post_peak = time[Tp_idx:]

Tr_idx_relative = np.where((post_peak_data < threshold) & (time_post_peak >= min_Tr_range) & (time_post_peak <= max_Tr_range))[0]

if len(Tr_idx_relative) > 0:
    Tr_idx = Tp_idx + Tr_idx_relative[-1]  # Pick the LAST drop within valid range
    Tr = time[Tr_idx]
    Ar = amplitude[Tr_idx]
else:
    Tr = max_Tr_range  # Default to upper bound if no valid Tr found
    Ar = 0

# Step 4: Compute Cycles (adjusted to T0 to Tr from second code)
signal_segment = amplitude[T0_idx:Tr_idx]
zero_crossings = np.where(np.diff(np.sign(signal_segment)))[0]
cycles = len(zero_crossings) // 2  # Count full cycles

# Step 5: Compute FFT parameters
time_step = np.mean(np.diff(time)) * 1e-6  # Convert to seconds
fft_data = fft(amplitude)
fft_magnitude = np.abs(fft_data[:N//2])

frequencies = fftfreq(N, time_step)[:N//2] / 1e3  # Convert Hz to kHz

# Find FFT frequency closest to Tx and Tr (retained from first code)
Tx_idx = (np.abs(frequencies - Tx_khz)).argmin()
Tr_idx = (np.abs(frequencies - Tr_khz)).argmin()

fft_frequency_tx = frequencies[Tx_idx]
fft_magnitude_tx = fft_magnitude[Tx_idx]

fft_frequency_tr = frequencies[Tr_idx]
fft_magnitude_tr = fft_magnitude[Tr_idx]

# Also include maximum frequency component (from second code)
max_freq_idx = np.argmax(fft_magnitude)
fft_frequency_max = frequencies[max_freq_idx]
fft_magnitude_max = fft_magnitude[max_freq_idx]

# Step 6: Print results (combined from both codes)
print(f"T0 (Initial Time): {T0:.2f} µs")
print(f"T1 (Rise Time): {T1:.2f} µs")
print(f"TOF (Time of Flight): {TOF:.2f} µs")
print(f"Tp (Peak Time): {Tp:.2f} µs")
print(f"Ap (Peak Amplitude): {Ap:.3f} V")
print(f"Tr (Ring-down Time): {Tr:.2f} µs")
print(f"Ar (Ring-down Amplitude): {Ar:.3f} V")
print(f"Cycles: {cycles}")
print(f"FFT Frequency (Tx): {fft_frequency_tx:.2f} kHz, Magnitude: {fft_magnitude_tx:.2f}")
print(f"FFT Frequency (Tr): {fft_frequency_tr:.2f} kHz, Magnitude: {fft_magnitude_tr:.2f}")
print(f"FFT Frequency (Max): {fft_frequency_max:.2f} kHz, Magnitude: {fft_magnitude_max:.2f} (arbitrary units)")

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
