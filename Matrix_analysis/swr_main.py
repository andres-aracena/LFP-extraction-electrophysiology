"""Main script for SWR event detection using the first 32 channels."""

import numpy as np
import matplotlib.pyplot as plt
from swr_detection import SWRDetector
from filters import (
    apply_butterworth_filter,
    apply_harmonic_notch_filters_to_dataframe,
    compute_hilbert_envelope,
)
from grafics import (
    plot_raw_and_filtered,
    plot_psd_pair,
    plot_reference_power,
    plot_spectrogram_pair,
    plot_candidate_windows,
    plot_detection_results_with_zoom,
    plot_envelope_and_candidates,
)


def prompt_plot_options():
    options = [
        ('1', 'Raw and filtered ripple signal'),
        ('2', 'PSD ripple and control bands'),
        ('3', 'Reference power (all channels)'),
        ('4', 'Envelope and peak detection (debug)'),
        ('5', 'Spectrograms ripple and control'),
        ('6', 'Candidate ripple windows'),
        ('7', 'SWR detection results with zoom'),
    ]

    valid_keys = {opt[0] for opt in options}
    print("\nPlot options:")
    for key, text in options:
        print(f"  {key}. {text}")

    choice = input("Choose plots to show (comma-separated, default all): ").strip()
    if not choice:
        return valid_keys

    selected = {item.strip() for item in choice.split(',') if item.strip() in valid_keys}
    return selected or valid_keys


def main():
    LFP_FILE = r"C:\Users\laboratorio\Documents\GitHub\LFP-extraction-electrophysiology\LFP_output\output_32ch.lfp"
    FS = 1250
    FILE_CHANNELS = 32
    PYRAMIDAL_CHANNEL_INDEX = None
    NOISE_REFERENCE_CHANNEL_INDEX = None
    CLIP_DURATION_SECONDS = 5 * 60

    print("=" * 70)
    print("SWR Event Detection (Single-Channel Direct Analysis)")
    print("=" * 70)
    print('DEBUG: Starting SWR detection main script...')

    detector = SWRDetector(fs=FS, ripple_band=(80, 250), control_band=(200, 500), filter_order=4)

    print("\nLoading LFP data and keeping first 32 channels...")
    df = detector.load_lfp_dataframe(LFP_FILE, n_channels=FILE_CHANNELS)
    rename_dict = {f'CH_{i}': f'CH_{i+1}' for i in range(FILE_CHANNELS)}
    df = df.rename(columns=rename_dict)

    even_channels = [f'CH_{i}' for i in range(1, FILE_CHANNELS + 1) if i % 2 == 0]
    df = df[even_channels]

    max_samples = min(df.shape[0], int(CLIP_DURATION_SECONDS * FS))
    df = df.iloc[:max_samples]
    print(f"Clipped data shape: {df.shape}")
    print(f"Clipped duration: {df.shape[0] / FS:.2f} s")

    df = apply_harmonic_notch_filters_to_dataframe(df, FS)
    print("Applied sequential 60 Hz and 180 Hz notch filters to all channels.")

    scores = detector.compute_channel_scores(df)

    if PYRAMIDAL_CHANNEL_INDEX is None:
        pyramidal_channel = detector.choose_pyramidal_channel(scores)
    else:
        pyramidal_channel = f'CH_{PYRAMIDAL_CHANNEL_INDEX}'

    if NOISE_REFERENCE_CHANNEL_INDEX is None:
        noise_reference_channel = detector.choose_noise_reference_channel(scores)
    else:
        noise_reference_channel = f'CH_{NOISE_REFERENCE_CHANNEL_INDEX}'

    print(f"Selected pyramidal channel: {pyramidal_channel}")
    print(f"Reference channel (for score visualization only): {noise_reference_channel}")

    pyramidal_signal = df[pyramidal_channel].values

    ripple_filtered = apply_butterworth_filter(pyramidal_signal, FS, 80, 250)
    control_filtered = apply_butterworth_filter(pyramidal_signal, FS, 200, 500)

    # Compute envelope for debug visualization
    ripple_envelope, _ = compute_hilbert_envelope(ripple_filtered)

    cwt_start, cwt_stop = 211.0, 213.0
    idx_start = int(cwt_start * FS)
    idx_stop = min(int(cwt_stop * FS), len(pyramidal_signal))
    cwt_ripple = ripple_filtered[idx_start:idx_stop]
    cwt_control = control_filtered[idx_start:idx_stop]

    plot_choices = prompt_plot_options()
    print("\nGenerating selected plots...")

    if '1' in plot_choices:
        print('DEBUG: Starting raw + filtered signal plot...')
        plot_raw_and_filtered(
            pyramidal_signal,
            ripple_filtered,
            FS,
            raw_title=f'Raw pyramidal signal: {pyramidal_channel}',
            filtered_title=f'Filtered ripple signal: {pyramidal_channel}',
        )
        print('DEBUG: Raw + filtered signal plot shown.')

    if '2' in plot_choices:
        print('DEBUG: Starting PSD comparison plot...')
        plot_psd_pair(
            cwt_ripple,
            cwt_control,
            FS,
            ripple_title=f'PSD ripple band: {pyramidal_channel}',
            control_title=f'PSD control band: {pyramidal_channel}',
            lowcut_ripple=80,
            highcut_ripple=250,
            lowcut_control=200,
            highcut_control=500,
        )
        print('DEBUG: PSD comparison plot shown.')

    if '3' in plot_choices:
        print('DEBUG: Starting reference power plot...')
        plot_reference_power(scores, pyramidal_channel, noise_reference_channel)
        print('DEBUG: Reference power plot shown.')


    if '4' in plot_choices:
        print('DEBUG: Starting envelope and peak detection plot...')
        from filters import detect_peaks_with_threshold
        peaks = detect_peaks_with_threshold(ripple_envelope, threshold_factor=5.0, min_distance_samples=int(0.020 * FS))
        plot_envelope_and_candidates(ripple_filtered, ripple_envelope, peaks, FS, 
                             threshold_factor=5.0, time_window=None)
        print('DEBUG: Envelope and peak detection plot shown.')

    if '5' in plot_choices:
        print('DEBUG: Starting spectrogram pair plot...')
        plot_spectrogram_pair(
            cwt_ripple,
            cwt_control,
            FS,
            ripple_band=(80, 250),
            control_band=(200, 500),
            start_time=cwt_start,
            wavelet_name='cmor3.0-2.0',
        )
        print('DEBUG: Spectrogram pair plot shown.')

    print("\nDetecting SWR events...")
    events = detector.detect_swr_events(pyramidal_signal)
    detector.print_summary(events)

    if '6' in plot_choices:
        print('DEBUG: Starting candidate windows plot...')
        plot_candidate_windows(pyramidal_signal, events, FS,
                               title='Candidate ripple windows')
        print('DEBUG: Candidate windows plot shown.')

    valid_events = [e for e in events if e.is_valid]
    if valid_events:
        print("\nFirst valid events:")
        for i, event in enumerate(valid_events[:5], 1):
            duration_ms = (event.end_sample - event.start_sample) / FS * 1000
            print(f"  {i}. {event.start_sample/FS:.3f}s-{event.end_sample/FS:.3f}s, {duration_ms:.1f}ms, "
                  f"freq={event.mean_frequency:.1f}Hz, cycles={event.num_cycles:.1f}")

    if '7' in plot_choices:
        print("\nPlotting combined SWR detection results with zoom...")
        print('DEBUG: Starting combined detection + zoom plot...')
        plot_detection_results_with_zoom(pyramidal_signal, events, FS)
        print('DEBUG: Combined detection + zoom plot shown.')

    output_file = "swr_detection_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Pyramidal channel: {pyramidal_channel}\n")
        f.write(f"Reference channel (score visualization only): {noise_reference_channel}\n")
        f.write(f"Total candidates: {len(events)}\n")
        f.write(f"Valid events: {len(valid_events)}\n\n")
        if not valid_events:
            f.write("No valid SWR events were detected.\n\n")
        for i, event in enumerate(events, 1):
            duration_ms = (event.end_sample - event.start_sample) / FS * 1000
            status = 'VALID' if event.is_valid else 'REJECTED'
            f.write(
                f"{i}. {event.start_sample/FS:.3f}s-{event.end_sample/FS:.3f}s "
                f"({duration_ms:.1f}ms), freq={event.mean_frequency:.1f}Hz, "
                f"cycles={event.num_cycles:.1f}, valid={status}, notes={event.validation_notes}\n"
            )

    print(f"\nResults saved to: {output_file}")
    print("=" * 70)

    # Keep all generated figures open until the user closes them manually.
    plt.show(block=True)


if __name__ == "__main__":
    main()
