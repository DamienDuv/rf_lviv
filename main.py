import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import SoapySDR
from SoapySDR import *  # Constantjj

# === SDR Configuration ===
sdr = SoapySDR.Device()
sample_rate = 10e6  # Lowered for better performance
center_freq = 868e6  # FM radio
gain = 50  # More practical value
buf_len = 8192  # More samples for FFT resolution

# Set up SDR
sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
sdr.setGain(SOAPY_SDR_RX, 0, gain)
sdr.setDCOffsetMode(SOAPY_SDR_RX, 0, True)


rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rxStream)

# === Visualization Setup ===
plt.ion()
fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(12, 7))
line_I, = ax_time.plot([], [], label="I", color='blue')
line_Q, = ax_time.plot([], [], label="Q", color='red')
line_fft, = ax_freq.plot([], [], label="Magnitude (dB)")
ax_time.set_title("Time Domain")
ax_time.set_xlabel("Sample Index")
ax_time.set_ylabel("Amplitude")
ax_time.legend()
ax_freq.set_title("Frequency Domain")
ax_freq.set_xlabel("Frequency (Hz)")
ax_freq.set_ylabel("Power (dB)")
plt.tight_layout()

# === Streaming Loop ===
try:
    while True:
        buff = np.empty(buf_len, dtype=np.complex64)
        sr = sdr.readStream(rxStream, [buff], len(buff))

        if sr.ret > 0:
            data = buff[:sr.ret]

            # Time domain
            line_I.set_data(np.arange(len(data)), data.real)
            line_Q.set_data(np.arange(len(data)), data.imag)
            ax_time.set_xlim(0, len(data))
            ax_time.set_ylim(-0.1, 0.1)

            # Frequency domain
            fft = np.fft.fftshift(np.fft.fft(data * np.hanning(len(data))))
            freqs = np.fft.fftshift(np.fft.fftfreq(len(data), d=1/sample_rate))
            fft_mag = 20 * np.log10(np.abs(fft) + 1e-6)

            freqs_real = freqs + center_freq
            line_fft.set_data(freqs_real / 1e6, fft_mag)
            ax_freq.set_xlim((center_freq - sample_rate/2)/1e6, (center_freq + sample_rate/2)/1e6)
            ax_freq.set_ylim(np.max(fft_mag)-60, np.max(fft_mag)+5)
            ax_freq.set_xticks(np.linspace((center_freq - sample_rate / 2) / 1e6,
                         (center_freq + sample_rate / 2) / 1e6, num=18))

            fig.canvas.draw()
            fig.canvas.flush_events()

except KeyboardInterrupt:
    print("Stopping...")

# Cleanup
sdr.deactivateStream(rxStream)
sdr.closeStream(rxStream)

