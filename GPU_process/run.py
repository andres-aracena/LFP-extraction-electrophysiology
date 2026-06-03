from src.extract_lfp import extract_lfp
from src.config import INPUT_FILE, OUTPUT_FILE, CHUNK_SIZE, N_CHANNELS


def main():
    # Define variables directly here (or edit src/config.py)
    # Using defaults from config.py for now, but these can be overridden
    input_file = r"C:\Users\laboratorio\Documents\GitHub\LFP-extraction-electrophysiology\BlackRock\continuous_32ch.dat"
    output_file = r"C:\Users\laboratorio\Documents\GitHub\LFP-extraction-electrophysiology\LFP_output\output_32ch.lfp"
    chunk_size = 134000000
    num_channels = 32

    extract_lfp(input_file, output_file, chunk_size, num_channels)


if __name__ == "__main__":
    main()
