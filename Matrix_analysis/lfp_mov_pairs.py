import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import neo


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

# Load all channels as (channels, samples).
signal_data = np.asarray(analog_signal).T
channel_ids = np.asarray(analog_signal.array_annotations['channel_ids']).astype(str)
channel_id_to_index = {channel_id: index for index, channel_id in enumerate(channel_ids)}

if len(channel_ids) != num_channels:
    raise ValueError('channel_ids does not match the number of signal channels.')

# Enter the time window to display (seconds).
start_time = 6 * 60.0
end_time = 8 * 60.0
recording_end = signal_data.shape[1] / fs
start_time = max(0.0, start_time)
end_time = min(end_time, recording_end)
if start_time >= end_time:
    raise ValueError("start_time must be smaller than end_time.")

plot_samples = np.arange(int(start_time * fs), int(end_time * fs))
plot_time = plot_samples / fs
n_plot_channels = min(32, signal_data.shape[0])
offset_value = 400
plot_data = signal_data[:n_plot_channels, plot_samples].T

# Split the voltage channels by their recorded channel number.  This avoids
# assuming that an even-numbered channel is always stored at an even index.
voltage_channel_ids = channel_ids[:n_plot_channels]
muted_channel_ids = {'31'}
even_channel_indices = [
    index for index, channel_id in enumerate(voltage_channel_ids)
    if channel_id not in muted_channel_ids and int(channel_id) % 2 == 0
]
odd_channel_indices = [
    index for index, channel_id in enumerate(voltage_channel_ids)
    if channel_id not in muted_channel_ids and int(channel_id) % 2 == 1
]


def add_channel_offsets(channel_indices):
    """Return selected traces with offsets and their matching tick locations."""
    channel_data = plot_data[:, channel_indices]
    offsets = np.arange(len(channel_indices)) * offset_value
    return channel_data + offsets, offsets


even_plot_data, even_offsets = add_channel_offsets(even_channel_indices)
odd_plot_data, odd_offsets = add_channel_offsets(odd_channel_indices)

# Compute normalized movement intensity from the accelerometer and gyroscope channels.
acc_channels = [channel_id_to_index[channel_id] for channel_id in ['34', '35', '33']]
gyro_channels = [channel_id_to_index[channel_id] for channel_id in ['38', '39', '37']]
acc_data = signal_data[acc_channels] - np.mean(signal_data[acc_channels], axis=1, keepdims=True)
gyro_data = signal_data[gyro_channels] - np.mean(signal_data[gyro_channels], axis=1, keepdims=True)
acc_z = acc_data / np.std(acc_data, axis=1, keepdims=True)
gyro_z = gyro_data / np.std(gyro_data, axis=1, keepdims=True)
acc_magnitude = np.linalg.norm(acc_z, axis=0)
gyro_magnitude = np.linalg.norm(gyro_z, axis=0)

# Plot even and odd LFP channels separately, with the remaining signals synchronized.
fig, (ax_even, ax_odd, ax_difference, ax_movement) = plt.subplots(
    4,
    1,
    figsize=(16, 15),
    sharex=True,
    gridspec_kw={'height_ratios': [4, 4, 1.5, 1.5]}
)

# Use a continuous colormap to generate distinct colors for all voltage channels.
colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_plot_channels))

for axis, channel_indices, channel_plot_data, offsets, group_name in (
    (ax_even, even_channel_indices, even_plot_data, even_offsets, 'Even'),
    (ax_odd, odd_channel_indices, odd_plot_data, odd_offsets, 'Odd'),
):
    for plot_index, channel_idx in enumerate(channel_indices):
        axis.plot(
            plot_time,
            channel_plot_data[:, plot_index],
            color=colors[channel_idx],
            linewidth=0.5,
            label=f"CH_{channel_ids[channel_idx]}"
        )

    axis.set_title(f"{group_name}-Numbered Voltage Channels", fontsize=14, pad=15)
    axis.set_ylabel("LFP amplitude + offset", fontsize=11)
    axis.set_xlim(start_time, end_time)
    axis.set_yticks(offsets)
    axis.set_yticklabels(
        [f"CH_{channel_ids[channel_idx]}" for channel_idx in channel_indices],
        fontsize=8
    )
    axis.grid(True, alpha=0.25, linestyle='--')

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
