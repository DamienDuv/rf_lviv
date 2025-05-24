import math
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import SoapySDR

class RFSource:
    def __init__(self, x, y, freq=2.45e9, power=1.0):
        self.x = x
        self.y = y
        self.freq = freq
        self.power = power
        self.speed_of_light = 3e8

    def calculate_amplitude(self, target_x, target_y):
        distance = math.sqrt((target_x - self.x)**2 + (target_y - self.y)**2)
        wavelength = self.speed_of_light / self.freq
        path_loss = (wavelength / (4 * math.pi * distance))**2
        received_power = self.power * path_loss
        amplitude = math.sqrt(received_power)
        return amplitude

class Antenna:
    def __init__(self, x, y, freq):
        self.x = x
        self.y = y
        self.freq = freq

    def process_signal_frequencies(self, signal_iq, sample_rate=4e6):
        #IQ signal
        fft_signal = np.fft.fft(signal_iq)
        n = len(signal_iq)
        frequencies = np.fft.fftfreq(n, d=1/sample_rate)
        amplitudes = np.abs(fft_signal)

        return frequencies, amplitudes

    def record_signal(self, rf_source):
        if rf_source.freq != self.freq:
            raise ValueError("Frequency of the source does not match our Antenna")

        amplitude = rf_source.calculate_amplitude(self.x, self.y)
        return amplitude

def plot_positions(rf_source, antennas):
    plt.figure()
    plt.scatter(rf_source.x, rf_source.y, color="red", label='RF Source')
    for i, antenna in enumerate(antennas):
        plt.scatter(antenna.x, antenna.y, color='blue', label=f"Antenna {i+1}" if i == 0 else "")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title('Positions of RF Source and Antennas')
    plt.legend()
    plt.grid(True)
    plt.show()

def determine_binant(antennas, amplitudes):
    max_amplitude = max(amplitudes, key=lambda x: x["amplitude"])
    strongest_antenna = max_amplitude["antenna"]

    if strongest_antenna.x < 50:
        return "Left"
    else:
        return "Right"

def main():
    parser = argparse.ArgumentParser(description="wow")
    parser.add_argument('--antennas', nargs='+', type=float, required=True)
    args = parser.parse_args()
    
    if len(args.antennas) % 2 != 0:
        parser.error('Number of antennas must be EVEN')

    rf_source = RFSource(x=50, y=50, freq=2.45e9, power=1.0)
    antennas = []

    amplitudes = []

    for i in range(0, len(args.antennas), 2):
        x = args.antennas[i]
        y = args.antennas[i+1]
        antenna = Antenna(x=x, y=y, freq=2.45e9)
        antennas.append(antenna)

    for i, antenna in enumerate(antennas):
        amplitude = antenna.record_signal(rf_source)
        print(f"Antenna at ({antenna.x}, {antenna.y}), receives: {amplitude:.2e} W^(1/2)")
        amplitudes.append({"antenna": antenna, "amplitude": amplitude})
    
    direction = determine_binant(antennas, amplitudes)
    print(f"The RF source is located to the {direction}")
    
    plot_positions(rf_source, antennas)


if __name__ == "__main__":
    main()
