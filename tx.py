import numpy as np
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import time

# === Transmitter Settings ===
center_freq = 864.1e6      # Center frequency (Hz) — e.g., 100 MHz
sample_rate = 2e6        # Sample rate (Hz)
gain = 50                # TX gain
tone_freq = 1e3          # Tone frequency (Hz)
duration = 92            # Duration to transmit (seconds)

# === Generate Signal ===
# num_samples = 2**14
# t = np.arange(num_samples) / sample_rate
# samples = 0.5 * np.exp(2j * np.pi * tone_freq * t).astype(np.complex64)  # Complex sine

# === Generate Constant Signal ===
num_samples = 2**14
constant_value = 1.5  # Constant value for the signal
samples = constant_value * np.ones(num_samples, dtype=np.complex64)  # Constant signal


# === Initialize SDR ===
sdr = SoapySDR.Device()
sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
sdr.setGain(SOAPY_SDR_TX, 0, gain)

# === Setup Stream ===
txStream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32)
sdr.activateStream(txStream)

# === Transmit Loop ===
print(f"Transmitting on {center_freq/1e6:.2f} MHz for {duration} seconds...")

start_time = time.time()
while time.time() - start_time < duration:
    sr = sdr.writeStream(txStream, [samples], len(samples))
    if sr.ret != len(samples):
        print("Warning: writeStream returned", sr)

# === Cleanup ===
print("Stopping transmission.")
sdr.deactivateStream(txStream)
sdr.closeStream(txStream)

