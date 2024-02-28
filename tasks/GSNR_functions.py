
def calculate_GSNR_i_max(F_i, L_i, P_base, eth_nli_i, R_s):
    """
    Calculate the maximum GSNR for the ith amplifier using the given parameters.

    Parameters:
    - F_i: Factor for the ith amplifier
    - L_i: Length of the ith segment
    - P_base: Base power
    - eth_nli_i: Efficiency factor for nonlinear interference for the ith amplifier
    - R_s: Symbol rate

    Returns:
    - GSNR_i_max: Maximum GSNR for the ith amplifier
    """
    GSNR_i_max = 2/3 * (1 / (2 * eth_nli_i * R_s * (F_i * L_i * P_base)**2))**(1/3)
    return GSNR_i_max

# Example usage with reasonable values
F_i = 1.5
L_i = 200  # Assume length in kilometers
P_base = 1e-3  # Assume base power in watts
eth_nli_i = 0.9  # Assume efficiency factor for nonlinear interference
R_s = 10e9  # Assume symbol rate in Hz

GSNR_max = calculate_GSNR_i_max(F_i, L_i, P_base, eth_nli_i, R_s)
print(f"Maximum GSNR: {GSNR_max:.6f}")




def calculate_optimal_launch_power(F_i, L_i, P_base, R_s, etha_i):
    """
    Calculate the optimal launch power using the given parameters.

    Parameters:
    - F_i: Factor for the ith amplifier
    - L_i: Length of the ith segment
    - P_base: Base power
    - R_s: Symbol rate
    - etha_i: Efficiency factor for the ith amplifier

    Returns:
    - P_opt_i: Optimal launch power for the ith segment
    """
    P_opt_i = ((F_i * L_i * P_base) / (2 * R_s * etha_i))**(1/3)
    return P_opt_i

# Example usage with reasonable values
F_i = 1.5
L_i = 200  # Assume length in kilometers
P_base = 1e-3  # Assume base power in watts
R_s = 10e9  # Assume symbol rate in Hz
etha_i = 0.8  # Assume efficiency factor

P_optimal = calculate_optimal_launch_power(F_i, L_i, P_base, R_s, etha_i)
print(f"Optimal Launch Power: {P_optimal:.6f} watts")



import math
from scipy import constants

def calculate_eta_nli(alpha, beta2_abs, gamma, length, Rs, df, Nch):
    # Constants
    pi = math.pi

    # Calculate effective length
    Leff = length / (1 + (Rs * alpha / (2 * beta2_abs))**2)

    # Calculate eta_nli
    eta_nli = (16 / (27 * pi)) * math.log((pi**2 / 2) * (abs(beta2_abs) * Rs**2) / (alpha * Nch**(2 * Rs / df))) \
              * alpha / abs(beta2_abs) * gamma**2 * Leff**2 / Rs**3

    return eta_nli

class Line:

    def __init__(self, line_data, num_channels=10):
        self.label = line_data['label']
        self.length = line_data['length']
        self.successive = {}
        self.channels = [{'state': 1} for _ in range(num_channels)]  # Initialize all channels as 1

        # Constants for amplifiers
        self.amplifier_spacing = 80e3  # Amplifier spacing in kilometers
        self.gain = 16  # Gain of the optical amplifier in dB
        self.noise_figure = 5.5  # Noise figure of the optical amplifier in dB

        # Fiber characteristics
        self.alpha_dB = 0.2  # Attenuation coefficient in dB/km
        self.beta2_abs = 2.13e-26  # Absolute value of dispersion parameter in ps^2/km
        self.gamma = 1.27e-3  # Nonlinear coefficient in (W m)^−1
        # Convert alpha from dB/km to 1/m using the conversion factor log10(e)/10
        self.alpha = self.alpha_dB / (10 * math.log10(math.e))

        # Calculate the number of amplifiers based on the line length
        self.n_amplifiers = self.calculate_amplifiers()

    def calculate_eta_nli(self, alpha, beta2_abs, gamma, length, Rs, df, Nch):
        # Constants
        pi = math.pi

        # Calculate effective length
        Leff = length / (1 + (Rs * alpha / (2 * beta2_abs)) ** 2)

        # Calculate eta_nli
        eta_nli = (16 / (27 * pi)) * math.log((pi ** 2 / 2) * (abs(beta2_abs) / alpha * Rs ** 2)) \
                  * alpha / abs(beta2_abs) * gamma ** 2 * Leff ** 2 / Rs ** 3

        return eta_nli

    def fiber_characteristics(self):
        return f"Attenuation Coefficient (α): {self.alpha:.6f} 1/m, " \
               f"Dispersion Parameter (|β₂|): {self.beta2_abs:.6e} ps^2/km, " \
               f"Nonlinear Coefficient (γ): {self.gamma:.6e} (W m)^−1"

    def calculate_amplifiers(self):
        # Calculate the number of amplifiers needed based on the line length and amplifier spacing
        return int(self.length / self.amplifier_spacing)

    def ase_generation(self):
        # Calculate ASE generation based on the number of amplifiers, Plank constant, and other constants
        h = constants.Planck  # Plank constant
        f = 193.414e12  # Frequency in Hz (C-band center)
        B_n = 12.5e9  # Noise bandwidth in Hz

        ase = self.n_amplifiers * h * f * B_n * self.noise_figure * (10 ** (self.gain / 10 - 1))
        return ase

    def nli_generation(self, signal_power):
        """
        Calculate the total amount of nonlinear interference noise in linear units.

        Parameters:
        - signal_power: Signal power in watts.

        Returns:
        - Total nonlinear interference noise in linear units.
        """
        # Calculate the total amount of nonlinear interference noise using the provided formula
        Bn = 12.5e9  # Noise bandwidth in Hz
        Rs = 32e9
        Nch = 10

        df = 50e9
        eta_nli = self.calculate_eta_nli(self.alpha, self.beta2_abs, self.gamma, self.length, Rs, df, Nch)

        nli = signal_power ** 3 * eta_nli * self.n_amplifiers * Bn
        return nli

    def latency_generation(self):
        # Calculate the latency based on the length and the speed of light in fiber
        speed_of_light = constants.speed_of_light  # Speed of light in meters per second in vacuum
        latency = (3 / 2) * (self.length / speed_of_light)
        return latency

    def noise_generation(self, signal_power):
        # Calculate total noise generation including ASE and NLI
        total_noise = self.ase_generation() + self.nli_generation(signal_power)
        return total_noise

    def set_channel_state(self, channel, new_state):
        # Set the state of the specified channel (1 or 0)
        if 0 <= channel < len(self.channels):
            self.channels[channel]['state'] = new_state

    def get_channel_state(self, channel):
        # Get the state of the specified channel
        if 0 <= channel < len(self.channels):
            return self.channels[channel]['state']
        else:
            return None

    def get_free_channels(self):
        # Get a list of indices of free channels
        return [i for i, channel in enumerate(self.channels) if channel['state'] == 1]

    # Other methods and attributes
    def __str__(self):
        return f"Line: Label = {self.label}, Length = {self.length}, Channels = {self.channels}"

# Example usage with a Line instance
line_data = {'label': 'AB', 'length': 200}
line = Line(line_data)
signal_power = 1e-3  # Assume signal power in watts
nli_value = line.noise_generation(signal_power)
print(f"Nonlinear Interference Noise: {nli_value:.6e}")
