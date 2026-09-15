# Configuration settings for LFP extraction tool
INPUT_FILE = r"path\to\your\dat" # Path to the input file
OUTPUT_FILE = r"path\to\your\lfp" # Path to the output file
SAMPLE_RATE_ORIGINAL = 30000  # Original sample rate in Hz
TARGET_SAMPLING_RATE = 1250      # Target sample rate in Hz

CUTOFF_FREQUENCY = 500   # Cutoff frequency for low-pass filter in Hz
CHUNK_SIZE = 134000000   # Bytes (~134 MB)
N_CHANNELS = 67  # Number of channels in the recordings

