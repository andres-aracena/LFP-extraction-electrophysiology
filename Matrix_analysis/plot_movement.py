# =========================
# Blackrock motion analysis
# =========================
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import neo

# Configuration
DATA_DIR = r"C:\Users\laboratorio\Downloads"
FILENAME = "20230522-090311-001.ns2"
FILE_PATH = os.path.join(DATA_DIR, FILENAME)

ACC_CH = [33, 34, 32]   # X, Y, Z
GYRO_CH = [37, 38, 36]  # X, Y, Z
MAX_POINTS = 20000

# Load recording
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(FILE_PATH)

signal = neo.io.BlackrockIO(FILE_PATH).read_block().segments[0].analogsignals[0]
fs = float(signal.sampling_rate)
data = np.asarray(signal.magnitude).T
time = np.arange(data.shape[1]) / fs

print(f"Recording: {FILENAME}")
print(f"Sampling rate: {fs:.0f} Hz | Duration: {time[-1]/60:.2f} min")

# Extract motion signals
acc = data[ACC_CH]
gyro = data[GYRO_CH]

# Remove DC offsets for visualization
acc = acc - np.mean(acc, axis=1, keepdims=True)
gyro = gyro - np.mean(gyro, axis=1, keepdims=True)

# Standardize axes for 3D visualization
acc_z = acc / np.std(acc, axis=1, keepdims=True)
gyro_z = gyro / np.std(gyro, axis=1, keepdims=True)

# Motion magnitude
acc_mag = np.linalg.norm(acc_z, axis=0)
gyro_mag = np.linalg.norm(gyro_z, axis=0)

# Downsample for plotting
idx = np.linspace(0, len(time)-1, min(MAX_POINTS, len(time))).astype(int)
t = time[idx]

# =========================
# 1. 3D motion state
# =========================
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection="3d")
ax1.plot(acc_z[0, idx], acc_z[1, idx], acc_z[2, idx], lw=0.5)
ax1.scatter(acc_z[0, idx[0]], acc_z[1, idx[0]], acc_z[2, idx[0]], s=30, label="Start")
ax1.scatter(acc_z[0, idx[-1]], acc_z[1, idx[-1]], acc_z[2, idx[-1]], s=30, label="End")
ax1.set(xlabel="ACC X", ylabel="ACC Y", zlabel="ACC Z", title="Accelerometer Motion Space")
ax1.legend(fontsize=8)

ax2 = fig.add_subplot(122, projection="3d")
ax2.plot(gyro_z[0, idx], gyro_z[1, idx], gyro_z[2, idx], lw=0.5)
ax2.scatter(gyro_z[0, idx[0]], gyro_z[1, idx[0]], gyro_z[2, idx[0]], s=30, label="Start")
ax2.scatter(gyro_z[0, idx[-1]], gyro_z[1, idx[-1]], gyro_z[2, idx[-1]], s=30, label="End")
ax2.set(xlabel="GYRO X", ylabel="GYRO Y", zlabel="GYRO Z", title="Gyroscope Motion Space")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.show()

# =========================
# 2. Movement intensity
# =========================
# =========================
# Movement intensity
# =========================
fig, ax = plt.subplots(figsize=(14, 4.5))

ax.plot(t, acc_mag[idx], lw=0.7, label="Acceleration")
ax.plot(t, gyro_mag[idx], lw=0.7, label="Angular velocity")

ax.set(
    xlabel="Time (s)",
    ylabel="Normalized magnitude",
    title="Animal Movement"
)

ax.legend(frameon=False, ncol=2, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.15)

ax.xaxis.set_major_formatter(
    FuncFormatter(lambda x, _: f"{int(x//60):02d}:{int(x%60):02d}")
)

plt.tight_layout()
plt.show()
