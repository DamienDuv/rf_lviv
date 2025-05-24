import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import SoapySDR
from SoapySDR import *  # Constantjj

# === SDR Configuration ===
master = SoapySDR.Device("driver=hackrf,serial=0000000000000000437c63dc21857dd7")
slave  = SoapySDR.Device("driver=hackrf,serial=000000000000000066a062dc2b5a9b9f")

sample_rate = 2e6  # Lowered for better performance
center_freq = 863.5e6  # FM radio
gain = 50  # More practical value
buf_len = 4096  # More samples for FFT resolution

# Set up SDR
master.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
master.setFrequency(SOAPY_SDR_RX, 0, center_freq)
master.setGain(SOAPY_SDR_RX, 0, gain)
master.setDCOffsetMode(SOAPY_SDR_RX, 0, True)

slave.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
slave.setFrequency(SOAPY_SDR_RX, 0, center_freq)
slave.setGain(SOAPY_SDR_RX, 0, gain)
slave.setDCOffsetMode(SOAPY_SDR_RX, 0, True)
slave.setClockSource("external")

# Set up streams
masterStream = master.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
slaveStream = slave.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
master.activateStream(masterStream)
slave.activateStream(slaveStream)

# === Visualization Setup ===
plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(14, 8))

# Master plots
line_I_m, = axs[0, 0].plot([], [], label="I", color='blue')
line_Q_m, = axs[0, 0].plot([], [], label="Q", color='red')
line_fft_m, = axs[1, 0].plot([], [], label="Magnitude (dB)")

# Slave plots
line_I_s, = axs[0, 1].plot([], [], label="I", color='blue')
line_Q_s, = axs[0, 1].plot([], [], label="Q", color='red')
line_fft_s, = axs[1, 1].plot([], [], label="Magnitude (dB)")

# Titles & labels
axs[0, 0].set_title("Master - Time Domain")
axs[0, 1].set_title("Slave - Time Domain")
axs[1, 0].set_title("Master - Frequency Domain")
axs[1, 1].set_title("Slave - Frequency Domain")
for ax in axs[0]:
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    ax.legend()
for ax in axs[1]:
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power (dB)")

plt.tight_layout()

# === Streaming Loop ===
try:
    while True:
        buff_m = np.empty(buf_len, dtype=np.complex64)
        buff_s = np.empty(buf_len, dtype=np.complex64)

        sr_m = master.readStream(masterStream, [buff_m], len(buff_m))
        sr_s = slave.readStream(slaveStream, [buff_s], len(buff_s))

        if sr_m.ret > 0 and sr_s.ret > 0:
            data_m = buff_m[:sr_m.ret]
            data_s = buff_s[:sr_s.ret]

            # Master time domain
            line_I_m.set_data(np.arange(len(data_m)), data_m.real)
            line_Q_m.set_data(np.arange(len(data_m)), data_m.imag)
            axs[0, 0].set_xlim(0, len(data_m))
            axs[0, 0].set_ylim(-0.1, 0.1)

            # Slave time domain
            line_I_s.set_data(np.arange(len(data_s)), data_s.real)
            line_Q_s.set_data(np.arange(len(data_s)), data_s.imag)
            axs[0, 1].set_xlim(0, len(data_s))
            axs[0, 1].set_ylim(-0.1, 0.1)

            # Master frequency domain
            fft_m = np.fft.fftshift(np.fft.fft(data_m * np.hanning(len(data_m))))
            freqs_m = np.fft.fftshift(np.fft.fftfreq(len(data_m), d=1 / sample_rate))
            fft_mag_m = 20 * np.log10(np.abs(fft_m) + 1e-6)
            freqs_real_m = freqs_m + center_freq
            line_fft_m.set_data(freqs_real_m / 1e6, fft_mag_m)
            axs[1, 0].set_xlim((center_freq - sample_rate / 2) / 1e6,
                               (center_freq + sample_rate / 2) / 1e6)
            axs[1, 0].set_ylim(np.max(fft_mag_m) - 60, np.max(fft_mag_m) + 5)

            # Slave frequency domain
            fft_s = np.fft.fftshift(np.fft.fft(data_s * np.hanning(len(data_s))))
            freqs_s = np.fft.fftshift(np.fft.fftfreq(len(data_s), d=1 / sample_rate))
            fft_mag_s = 20 * np.log10(np.abs(fft_s) + 1e-6)
            freqs_real_s = freqs_s + center_freq
            line_fft_s.set_data(freqs_real_s / 1e6, fft_mag_s)
            axs[1, 1].set_xlim((center_freq - sample_rate / 2) / 1e6,
                               (center_freq + sample_rate / 2) / 1e6)
            axs[1, 1].set_ylim(np.max(fft_mag_s) - 60, np.max(fft_mag_s) + 5)

            fig.canvas.draw()
            fig.canvas.flush_events()

except KeyboardInterrupt:
    print("Stopping...")

# Cleanup
master.deactivateStream(masterStream)
master.closeStream(masterStream)
slave.deactivateStream(slaveStream)
slave.closeStream(slaveStream)
