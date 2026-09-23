import os
import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import neo
from scipy.signal import butter, filtfilt, hilbert, iirnotch, welch, find_peaks
from scipy.integrate import trapezoid
from scipy.fft import fft, ifft
import pywt


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

# Transfer data to GPU using CuPy and transpose to get (channels, samples)
signal_data_gpu = cp.array(analog_signal).T
signal_data_cpu = np.array(analog_signal).T  # For CPU-based operations

# Bring a small segment to CPU for Pandas DataFrame exploration
# Keep this preview limited to the first 10 seconds.
max_samples_to_explore = min(10000, signal_data_gpu.shape[1])
time_array_cpu = np.arange(max_samples_to_explore) / fs
signal_segment_cpu = cp.asnumpy(signal_data_gpu[:, :max_samples_to_explore].T)

# Create channel names dynamically for all 32 channels
channel_names = [f"CH_{i}" for i in range(num_channels)]

# Construct exploration DataFrame
df_exploration = pd.DataFrame(signal_segment_cpu, columns=channel_names)
df_exploration.insert(0, 'Time (s)', time_array_cpu)
df_exploration.set_index('Time (s)', inplace=True)


# --- Visualization: Full Recording, First 32 Voltage Channels Stacked ---
n_plot_channels = min(32, signal_data_cpu.shape[0])
offset_value = 600
max_plot_points = 200_000

# Downsample only for display while preserving the complete recording duration.
step = max(1, int(np.ceil(signal_data_cpu.shape[1] / max_plot_points)))
plot_samples = np.arange(0, signal_data_cpu.shape[1], step)
plot_time = plot_samples / fs
plot_data = signal_data_cpu[:n_plot_channels, plot_samples].T

# Apply vertical offset to separate LFP channels cleanly.
lfp_offsets = np.arange(n_plot_channels) * offset_value
plot_data_offset = plot_data + lfp_offsets

# Prepare all channels after CH_31 as auxiliary channels.
n_auxiliary_channels = max(0, signal_data_cpu.shape[0] - n_plot_channels)
auxiliary_data = signal_data_cpu[n_plot_channels:, plot_samples].T
auxiliary_scale = np.nanpercentile(np.abs(auxiliary_data), 95, axis=0)
auxiliary_scale = np.maximum(auxiliary_scale, np.finfo(float).eps)
auxiliary_offsets = np.arange(n_auxiliary_channels) * 2.0
auxiliary_data_offset = auxiliary_data / auxiliary_scale + auxiliary_offsets

# Plot LFP and auxiliary channels on synchronized axes.
fig, (ax_lfp, ax_auxiliary) = plt.subplots(
    2,
    1,
    figsize=(14, 15),
    sharex=True,
    gridspec_kw={'height_ratios': [4, 1.5]}
)

# Use a continuous colormap to generate distinct colors for all 32 channels
colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_plot_channels))

for channel_idx in range(n_plot_channels):
    ax_lfp.plot(
        plot_time,
        plot_data_offset[:, channel_idx],
        color=colors[channel_idx],
        linewidth=0.5
    )

# Layout adjustments
ax_lfp.set_title(
    f"Full Recording | Voltage Channels 0 to {n_plot_channels - 1}",
    fontsize=14,
    pad=15
)
ax_lfp.set_ylabel("LFP amplitude + offset", fontsize=11)
ax_lfp.set_xlim(plot_time[0], plot_time[-1])
ax_lfp.grid(True, alpha=0.25, linestyle='--')

# Match Y-ticks with channel positions for instant identification
ax_lfp.set_yticks(lfp_offsets)
ax_lfp.set_yticklabels([f"CH_{channel_idx}" for channel_idx in range(n_plot_channels)], fontsize=8)

if n_auxiliary_channels:
    auxiliary_colors = plt.cm.tab20(np.linspace(0, 1, n_auxiliary_channels))
    for channel_idx in range(n_auxiliary_channels):
        ax_auxiliary.plot(
            plot_time,
            auxiliary_data_offset[:, channel_idx],
            color=auxiliary_colors[channel_idx % len(auxiliary_colors)],
            linewidth=0.5
        )

    ax_auxiliary.set_title(
        f"Synchronized Auxiliary Channels {n_plot_channels} to {signal_data_cpu.shape[0] - 1}",
        fontsize=12
    )
    ax_auxiliary.set_ylabel("Normalized + offset", fontsize=10)
    ax_auxiliary.set_yticks(auxiliary_offsets)
    ax_auxiliary.set_yticklabels(
        [f"CH_{n_plot_channels + channel_idx}" for channel_idx in range(n_auxiliary_channels)],
        fontsize=8
    )
else:
    ax_auxiliary.text(0.5, 0.5, "No auxiliary channels available", ha="center", va="center")
    ax_auxiliary.set_yticks([])

ax_auxiliary.set_xlabel("Time (s)", fontsize=11)
ax_auxiliary.grid(True, alpha=0.25, linestyle='--')

def format_time_axis(seconds, position):
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"

ax_auxiliary.xaxis.set_major_formatter(FuncFormatter(format_time_axis))

plt.tight_layout()
plt.show()