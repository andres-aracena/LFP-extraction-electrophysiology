import os
import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neo
from scipy.signal import butter, filtfilt, hilbert, iirnotch, welch, find_peaks
from scipy.integrate import trapezoid
from scipy.fft import fft, ifft
import pywt
from matplotlib.ticker import FuncFormatter


# Directory and file configuration
data_dir = "C:\\Users\\laboratorio\\Downloads"
filename = "20230522-090311-001.ns2" 
file_path = os.path.join(data_dir, filename)

if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {filename} was not found in {data_dir}")

# Load continuous data using Neo BlackrockIO reader
reader = neo.io.BlackrockIO(file_path)
block = reader.read_block()
analog_signal = block.segments[0].analogsignals[0]

# Extract basic properties
fs = int(analog_signal.sampling_rate)  
num_channels = analog_signal.shape[1]
n_voltage_channels = 32  # Number of voltage channels 
offset_value = 600

# Load all channels as (channels, samples).
signal_data = np.asarray(analog_signal).T
channel_ids = np.asarray(analog_signal.array_annotations['channel_ids']).astype(str)
channel_names = [f'CH_{channel_id}' for channel_id in channel_ids]
channel_id_to_index = {channel_id: index for index, channel_id in enumerate(channel_ids)}

if len(channel_ids) != num_channels:
    raise ValueError('channel_ids does not match the number of signal channels.')

# Enter the time window to display (seconds).
start_time = 0 * 60.0
end_time = 5 * 60.0
recording_end = signal_data.shape[1] / fs
start_time = max(0.0, start_time)
end_time = min(end_time, recording_end)
if start_time >= end_time:
    raise ValueError("start_time must be smaller than end_time.")

plot_samples = np.arange(int(start_time * fs), int(end_time * fs))
plot_time = plot_samples / fs
n_proc_channels = min(n_voltage_channels, signal_data.shape[0])

segmented_data = signal_data[:n_proc_channels, plot_samples].T

# Design Notch Filters
b60, a60 = iirnotch(w0=60.0, Q=30.0, fs=fs)   # Standard aggressiveness
b180, a180 = iirnotch(w0=180.0, Q=90.0, fs=fs) # Less aggressive (Narrower)

# Pre-allocate array for filtered signals (only for the 32 real voltage channels)
notch_data = np.zeros((n_voltage_channels, segmented_data.shape[1]))

print("Applying Notch filters (60Hz and 180Hz) to all voltage channels...")
for ch_idx in range(n_voltage_channels):
    ch_filt60 = filtfilt(b60, a60, segmented_data[ch_idx, :])
    notch_data[ch_idx, :] = filtfilt(b180, a180, ch_filt60)

print("Filtering complete.")


# Welch periodogram parameters
window_seconds = 4
nperseg = int(window_seconds * fs)  # 4-second window
noverlap = int(nperseg / 2)         # 50% overlap

channel_indices = np.arange(n_voltage_channels)
ripple_powers = []
ripple_scores = []

print('Computing spectral metrics for 32 channels')

for channel_index in channel_indices:
    frequencies, psd = welch(
        notch_data[channel_index],
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap
    )
    ripple_mask = (frequencies >= 100) & (frequencies <= 250)
    surrounding_mask = (frequencies >= 70) & (frequencies <= 300)
    ripple_power = np.trapezoid(psd[ripple_mask], frequencies[ripple_mask])
    surrounding_power = np.trapezoid(
        psd[surrounding_mask], frequencies[surrounding_mask]
    )
    score = ripple_power / surrounding_power if surrounding_power > 0 else 0
    ripple_powers.append(ripple_power)
    ripple_scores.append(score)

ripple_powers = np.array(ripple_powers)
ripple_scores = np.array(ripple_scores)
highest_score_index = np.argmax(ripple_scores)
lowest_score_index = np.argmin(ripple_scores)
highest_score_channel_index = channel_indices[highest_score_index]
lowest_score_channel_index = channel_indices[lowest_score_index]

print('\n=== PYRAMIDAL LAYER CHANNEL DETERMINATION ===')
print(
    f"Green Bar -> Reference Channel: {channel_names[highest_score_channel_index]} "
    f"(Score: {ripple_scores[highest_score_index]:.4f})"
)
print(
    f"Red Bar -> Worst Quality Channel: {channel_names[lowest_score_channel_index]} "
    f"(Score: {ripple_scores[lowest_score_index]:.4f})"
)

best = notch_data[highest_score_channel_index]
worst = notch_data[lowest_score_channel_index]


plot_data = signal_data[:n_proc_channels, plot_samples].T

# Apply vertical offset to separate LFP channels cleanly.
lfp_offsets = np.arange(n_proc_channels) * offset_value
plot_data_offset = plot_data + lfp_offsets

# Compute normalized movement intensity from the accelerometer and gyroscope channels.
acc_channels = [channel_id_to_index[channel_id] for channel_id in ['34', '35', '33']]
gyro_channels = [channel_id_to_index[channel_id] for channel_id in ['38', '39', '37']]
acc_data = signal_data[acc_channels] - np.mean(signal_data[acc_channels], axis=1, keepdims=True)
gyro_data = signal_data[gyro_channels] - np.mean(signal_data[gyro_channels], axis=1, keepdims=True)
acc_z = acc_data / np.std(acc_data, axis=1, keepdims=True)
gyro_z = gyro_data / np.std(gyro_data, axis=1, keepdims=True)
acc_magnitude = np.linalg.norm(acc_z, axis=0)
gyro_magnitude = np.linalg.norm(gyro_z, axis=0)

# Plot LFP, differential LFP, and movement on one synchronized figure.
fig, (ax_lfp, ax_difference, ax_movement) = plt.subplots(
    3,
    1,
    figsize=(16, 12),
    sharex=True,
    gridspec_kw={'height_ratios': [6, 1.5, 1.5]}
)

# Use a continuous colormap to generate distinct colors for all 32 channels
colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_proc_channels))

for channel_idx in range(n_proc_channels):
    if channel_ids[channel_idx] == '31':
        continue
    ax_lfp.plot(
        plot_time,
        plot_data_offset[:, channel_idx],
        color=colors[channel_idx],
        linewidth=0.5
    )

# Layout adjustments
ax_lfp.set_title(
    f"Full Recording | Voltage Channels {channel_ids[0]} to {channel_ids[n_proc_channels - 1]}",
    fontsize=14,
    pad=15
)
ax_lfp.set_ylabel("LFP amplitude + offset", fontsize=11)
ax_lfp.set_xlim(start_time, end_time)
ax_lfp.grid(True, alpha=0.25, linestyle='--')

# Match Y-ticks with channel positions for instant identification.
ax_lfp.set_yticks(lfp_offsets)
ax_lfp.set_yticklabels(
    [f"CH_{channel_ids[channel_idx]}" if channel_ids[channel_idx] != '31'
     else 'CH_31 (muted)' for channel_idx in range(n_proc_channels)],
    fontsize=8
)

# Plot the differential signal below the LFP panel.
differential_channels = ('26', '28')
differential_signal = (
    signal_data[channel_id_to_index[differential_channels[0]]] -
    signal_data[channel_id_to_index[differential_channels[1]]]
)
ax_difference.plot(
    plot_time,
    differential_signal[plot_samples],
    color='black',
    linewidth=0.6
)
ax_difference.set_title(
    f'Differential LFP | CH_{differential_channels[0]} - CH_{differential_channels[1]}',
    fontsize=12
)
ax_difference.set_ylabel('Difference', fontsize=10)
ax_difference.grid(True, axis='y', alpha=0.15)
ax_difference.spines[['top', 'right']].set_visible(False)

ax_movement.plot(
    plot_time,
    acc_magnitude[plot_samples],
    color="tab:blue",
    linewidth=0.7,
    label="Acceleration"
)
ax_movement.plot(
    plot_time,
    gyro_magnitude[plot_samples],
    color="tab:orange",
    linewidth=0.7,
    label="Angular velocity"
)
ax_movement.set_title("Movement Intensity", fontsize=12)
ax_movement.set_ylabel("Normalized magnitude", fontsize=10)
ax_movement.set_xlabel("Time (mm:ss.mmm)", fontsize=11)
ax_movement.set_ylim(0, 20)
ax_movement.legend(frameon=False, ncol=2, loc="upper right", fontsize=8)
ax_movement.spines[["top", "right"]].set_visible(False)
ax_movement.grid(axis="y", alpha=0.15)

def format_time_axis(seconds, position):
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    if milliseconds == 1000:
        remaining_seconds += 1
        milliseconds = 0
    return f"{minutes:02d}:{remaining_seconds:02d}.{milliseconds:03d}"

ax_movement.xaxis.set_major_formatter(FuncFormatter(format_time_axis))

plt.tight_layout()
plt.show()