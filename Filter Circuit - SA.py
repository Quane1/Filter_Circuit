import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# Design Variables
r1 , r2, r3, r4 = 5, 30, 10, 30
n = 4
c1 = 10*10e-6
c2 = c1 * n

# Cutoff Freq
def cutoff(res, cap):
    # Give cutoff frequency based on components
    freq = 1 / (2 * np.pi * res * cap)
    return freq


#---------------------------Transfer Functions---------------------------

def high_trans_func(res1, cap1):
    # High pass filter transfer function
    b_hp, a_hp = signal.butter(1, 2 * np.pi * res1 * cap1, btype='highpass', analog=True)
    w, h = signal.freqs(b_hp, a_hp)
    return w, h

def low_trans_func(res2, cap2):
    # Low pass filter transfer function
    b_lp, a_lp = signal.butter(1, 2 * np.pi * res2 * cap2, btype='low', analog=True)
    w, h = signal.freqs(b_lp, a_lp)
    return w, h

def circuit_transfer_function(res1, res2, res3, res4, cap1, cap2):
    # Cascading transfer function

    # Amplifier transfer function
    d_amp = [1 + (res4/res3)]
    a_amp = [1 + (res4/res3)]

    # Lowpass
    d_lp, a_lp = low_trans_func(res2, cap2)

    # Highpass
    d_hp, a_hp = high_trans_func(res1, cap1)

    # Cascading
    d_total = np.convolve(d_hp, d_amp)
    a_total = np.convolve(a_hp, a_amp)
    d_total = np.convolve(d_total, d_lp)
    a_total = np.convolve(a_total, a_lp)

    w, h = signal.freqs(d_total, a_total)
    return w, h    

#---------------------------Bode Plots---------------------------
# Use log distribution 


# High Pass
wh, hh = high_trans_func(r1, c1)

plt.figure(1, figsize=(12,6))
plt.suptitle('High Pass Transfer Function')

plt.subplot(211)
plt.semilogx(wh, 20 * np.log10(abs(hh)), label='Gain')
plt.ylabel('Gain (dB)')
plt.grid(True)
plt.legend()

plt.subplot(212)
plt.semilogx(wh, np.angle(hh), label='Phase')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.grid(True)
plt.legend()


# Low Pass
wl, hl = low_trans_func(r2, c2)

plt.figure(2, figsize=(12,6))
plt.title('Low Pass Transfer Function')

plt.subplot(211)
plt.semilogx(wl, 20 * np.log10(abs(hl)), label='Gain')
plt.ylabel('Gain (dB)')
plt.grid(True)
plt.legend()

plt.subplot(212)
plt.semilogx(wl, np.angle(hl), label='Phase')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.grid(True)
plt.legend()




#---------------------------Range of Parameters---------------------------

# Values for N
n_vals = np.linspace(1, 10, 10)     # Range for n
# Values fpr C1
c_vals = np.linspace(1*10e-6, 100*10e-6,  100)

# Resistors +/- 10% tolerance
res1_values = [(r1 * 0.9), r1, (r1 * 1.1)]
res2_values = [(r2 * 0.9), r2, (r2 * 1.1)]
res3_values = [(r3 * 0.9), r3, (r3 * 1.1)]
res4_values = [(r4 * 0.9), r4, (r4 * 1.1)]


# FOM
# Bandwidth, Center
# bandw = hp_freq - lp_freq
# center = bandw / 2

# Arrays to store values
bandwdth_n_results = np.zeros(len(n_vals))
centerfreq_n_results = np.zeros(len(n_vals))
bandwdth_c_results = np.zeros(len(c_vals))
centerfreq_c_results = np.zeros(len(c_vals))
bandwdth_r_results = np.zeros((len(res1_values), len(res2_values), len(res3_values), len(res4_values)))
centerfreq_r_results = np.zeros((len(res1_values), len(res2_values), len(res3_values), len(res4_values)))

#---------------------------C1 Sensitivity Analysis---------------------------
# Preform sensitivity analysis for C1
for i, c_val in enumerate(c_vals):
    freq_high = cutoff(r1, c_val)
    freq_low = cutoff(r2, c_val * n)
    bandwdth_c_results[i] = freq_high - freq_low
    centerfreq_c_results[i] = bandwdth_c_results[i]/2
    

# Plot C1 Sensitivity Analysis 
plt.figure(3, figsize=(12,6))
plt.suptitle('Sensitivity Analysis regarding C1')

# Bandwidth
plt.subplot(211)
plt.plot(c_vals, bandwdth_c_results, label='Bandwidth')
plt.ylabel('Bandwidth (Hz)')
plt.grid(True)
plt.legend()

# Center Frequency
plt.subplot(212)
plt.plot(c_vals, centerfreq_c_results, label='Center Freq')
plt.xlabel('Value of C1 (Farads)')
plt.ylabel('Center Frequency (Hz)')
plt.grid(True)
plt.legend()


#---------------------------N Sensitivity Analysis---------------------------
# Preform sensitivity analysis for n
for i, N in enumerate(n_vals):
    freq_high = cutoff(r1, c1)
    freq_low = cutoff(r2, c1 * N)
    bandwdth_n_results[i] = freq_high - freq_low
    centerfreq_n_results[i] = bandwdth_n_results[i]/2
    

# Plot n Sensitivity Analysis 
plt.figure(4, figsize=(12,6))
plt.suptitle('Sensitivity Analysis regarding N')

# Bandwidth
plt.subplot(211)
plt.plot(n_vals, bandwdth_n_results, label='Bandwidth')
plt.ylabel('Bandwidth (Hz)')
plt.grid(True)
plt.legend()

# Center Frequency
plt.subplot(212)
plt.plot(n_vals, centerfreq_n_results, label='Center Freq')
plt.xlabel('Value of N')
plt.ylabel('Center Frequency (Hz)')
plt.grid(True)
plt.legend()


#---------------------------Resistor Sensitivity Analysis---------------------------
# Preform sensitivity analysis for resistors
for i, R1 in enumerate(res1_values):
    for j, R2 in enumerate(res2_values):
        for k, R3 in enumerate(res3_values):
            for l, R4 in enumerate(res4_values):
                freq_high = cutoff(R1, c1)
                freq_high = cutoff(R2, c2)
                bandwdth_r_results[i, j, k, l] = freq_high - freq_low
                centerfreq_r_results[i, j, k, l] = bandwdth_r_results[i, j, k, l]/2


# Plot n Sensitivity Analysis 
plt.figure(5, figsize=(12,6))

# Bandwidth
plt.subplot(211)
plt.contour(res1_values, res2_values, np.mean(bandwdth_r_results, axis=(2, 3)), cmap='viridis')
plt.xlabel('Resistor 1 (ohms)')
plt.ylabel('Resistor 2 (ohms)')
plt.grid(True)
plt.title('Bandwidth SA')
plt.colorbar()

# Center Frequency
plt.subplot(212)
plt.contour(res3_values, res4_values, np.mean(centerfreq_r_results, axis=(2, 3)), cmap='viridis')
plt.xlabel('Resistor 3 (ohms)')
plt.ylabel('Resistor 4 (ohms)')
plt.grid(True)
plt.title('Center Frequency SA')
plt.colorbar



plt.tight_layout()
plt.show()