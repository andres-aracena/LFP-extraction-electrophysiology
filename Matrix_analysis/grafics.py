"""
Plotting utilities for SWR detection and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from filters import apply_butterworth_filter


def plot_raw_and_filtered(signal_data: np.ndarray, filtered_signal: np.ndarray, fs: int,
                          raw_title: str = 'Raw differential signal',
                          filtered_title: str = 'Filtered ripple signal'):
    time = np.arange(len(signal_data)) / fs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    ax1.plot(time, signal_data, color='black')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(time[0], time[-1])
    ax1.set_title(raw_title)
    ax1.grid(True, alpha=0.25)

    ax2.plot(time, filtered_signal, color='blue')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_xlim(time[0], time[-1])
    ax2.set_title(filtered_title)
    ax2.grid(True, alpha=0.25)

    fig.suptitle('Raw and Ripple-Filtered Differential Signal', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=False)
    plt.pause(0.001)



def plot_psd_pair(ripple_filtered: np.ndarray, control_filtered: np.ndarray, fs: int,
                  ripple_title: str = 'PSD ripple band',
                  control_title: str = 'PSD control band',
                  lowcut_ripple: int = 80, highcut_ripple: int = 250,
                  lowcut_control: int = 200, highcut_control: int = 500):
    freqs_r, psd_r = signal.welch(ripple_filtered, fs=fs, nperseg=min(1024, len(ripple_filtered)))
    freqs_c, psd_c = signal.welch(control_filtered, fs=fs, nperseg=min(1024, len(control_filtered)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(freqs_r, psd_r, color='darkgreen', label='PSD Signal')
    ax1.axvspan(lowcut_ripple, highcut_ripple, color='gray', alpha=0.2, label='Target Band (80-250 Hz)')
    ax1.set_title(ripple_title)
    ax1.set_ylabel('Power Spectral Density (V²/Hz)')
    ax1.grid(True, which='both', ls='--', lw=0.5, alpha=0.7)
    ax1.legend(loc='upper right', frameon=True)
    #ax1.set_ylim(1e-13, 5e3)
    ax1.set_xlim(0, fs / 2)

    ax2.plot(freqs_c, psd_c, color='purple', label='PSD Signal')
    ax2.axvspan(lowcut_control, highcut_control, color='gray', alpha=0.2, label='Control Band (200-500 Hz)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density (V²/Hz)')
    ax2.set_title(control_title)
    ax2.grid(True, which='both', ls='--', lw=0.5, alpha=0.7)
    ax2.legend(loc='upper right', frameon=True)
    #ax2.set_ylim(1e-13, 5e3)
    ax2.set_xlim(0, fs / 2)
    
    fig.suptitle('PSD Comparison: Ripple vs Control Band', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=False)
    plt.pause(0.001)


def plot_spectrogram_pair(signal_data_ripple: np.ndarray, signal_data_control: np.ndarray,
                          fs: int, ripple_band: tuple[int, int], control_band: tuple[int, int],
                          start_time: float = 0.0,
                          wavelet_name: str = 'cmor3.0-2.0'):
    dt = 1.0 / fs
    step_freq = 0.2
    min_r, max_r = ripple_band
    min_c, max_c = control_band
    
    freqs_r = np.linspace(min_r, max_r, int((max_r - min_r) / step_freq) + 1)
    freqs_c = np.linspace(min_c, max_c, int((max_c - min_c) / step_freq) + 1)
    wav = pywt.ContinuousWavelet(wavelet_name)

    scales_r = pywt.central_frequency(wav) / (freqs_r * dt)
    scales_c = pywt.central_frequency(wav) / (freqs_c * dt)

    W_r, _ = pywt.cwt(signal_data_ripple, scales_r, wav, method='fft', sampling_period=dt)
    W_c, _ = pywt.cwt(signal_data_control, scales_c, wav, method='fft', sampling_period=dt)

    W_r = np.abs(W_r) ** 2
    W_c = np.abs(W_c) ** 2

    cwt_freqs_r = pywt.scale2frequency(wav, scales_r) / dt
    cwt_freqs_c = pywt.scale2frequency(wav, scales_c) / dt

    time_r = np.arange(len(signal_data_ripple)) / fs + start_time
    time_c = np.arange(len(signal_data_control)) / fs + start_time

    # --- Compute Shared Shared Color Limits (Global VMAX) ---
    global_vmax = max(np.percentile(W_r, 99), np.percentile(W_c, 99))
    global_vmin = 0.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Top Subplot: Ripple Band
    im1 = ax1.imshow(W_r,
                     extent=[time_r[0], time_r[-1], cwt_freqs_r.min(), cwt_freqs_r.max()],
                     aspect='auto', cmap='jet', vmax=global_vmax, vmin=global_vmin,
                     origin='lower')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title(f'Ripple Band Spectrogram: {min_r}-{max_r} Hz')
    ax1.set_ylim(min_r, max_r)
    fig.colorbar(im1, ax=ax1, label='Power')

    # Bottom Subplot: Control Band
    im2 = ax2.imshow(W_c,
                     extent=[time_c[0], time_c[-1], cwt_freqs_c.min(), cwt_freqs_c.max()],
                     aspect='auto', cmap='jet', vmax=global_vmax, vmin=global_vmin,
                     origin='lower')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title(f'Control Band Spectrogram: {min_c}-{max_c} Hz')
    ax2.set_ylim(min_c, max_c)
    fig.colorbar(im2, ax=ax2, label='Power')

    # Global Formatting
    fig.suptitle(f'Normalized Wavelet Spectrogram Comparison ({wavelet_name})')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

def plot_candidate_windows(signal_data: np.ndarray, events: list, fs: int,
                           max_plots: int = 16, half_window_sec: float = 0.1,
                           title: str = 'Candidate ripple windows'):
    if not events:
        print('No ripple candidates available for candidate window plot.')
        return

    display_events = events[:max_plots]
    n = len(display_events)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 2 * n_rows), squeeze=False)
    axes = axes.reshape(-1)

    for ax, event in zip(axes, display_events):
        start = max(0, event.peak_sample - int(half_window_sec * fs))
        stop = min(len(signal_data), event.peak_sample + int(half_window_sec * fs))
        time = (np.arange(start, stop) - event.peak_sample) / fs
        segment = signal_data[start:stop]
        ax.plot(time, segment, color='black')
        ax.axvline(0.0, color='red', linestyle='--')
        ax.set_title(f'Event @ {event.peak_sample / fs:.3f}s', fontsize=10)
        ax.set_xlabel('Time from event peak (s)', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, alpha=0.2)
        ax.label_outer()

    for extra_ax in axes[n:]:
        fig.delaxes(extra_ax)

    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(hspace=0.45, wspace=0.35, top=0.92)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show(block=False)
    plt.pause(0.001)


def plot_reference_power(scores, pyramidal_channel: str, noise_reference_channel: str):
    """Plot ripple band score and power for pyramidal and noise reference channels."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    
    # Up: Ripple Score (paper metric: ripple power / surrounding power)
    scores_sorted = scores.sort_values('ripple_score', ascending=False)
    bars1 = ax1.bar(range(len(scores_sorted)), scores_sorted['ripple_score'], color='lightgray')
    for i, (bar, channel) in enumerate(zip(bars1, scores_sorted['channel'])):
        if channel == pyramidal_channel:
            bar.set_color('green')
        elif channel == noise_reference_channel:
            bar.set_color('red')
    ax1.set_xticks(range(len(scores_sorted)))
    ax1.set_xticklabels(scores_sorted['channel'], rotation=90)
    ax1.set_ylabel('Ripple Score (100–250 Hz / 70–300 Hz)')
    ax1.set_title('Ripple Band Score (Paper Metric)')
    ax1.grid(axis='y', alpha=0.3)
    
    # Down: Raw Ripple Power (100-250 Hz)
    scores_power = scores.sort_values('ripple_power', ascending=False)
    bars2 = ax2.bar(range(len(scores_power)), scores_power['ripple_power'], color='lightgray')
    for i, (bar, channel) in enumerate(zip(bars2, scores_power['channel'])):
        if channel == pyramidal_channel:
            bar.set_color('green')
        elif channel == noise_reference_channel:
            bar.set_color('red')
    ax2.set_xticks(range(len(scores_power)))
    ax2.set_xticklabels(scores_power['channel'], rotation=90)
    ax2.set_ylabel('Ripple Power (100–250 Hz)')
    ax2.set_title('Raw Ripple Power')
    ax2.grid(axis='y', alpha=0.3)
    
    # Legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color='green', label='Pyramidal channel (high ripple score)'),
                       plt.Rectangle((0, 0), 1, 1, color='red', label='Noise reference (low ripple score)')]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2)
    plt.tight_layout()
    plt.show(block=False)


def plot_detection_results_with_zoom(signal_data: np.ndarray, events: list,
                                     fs: int, zoom_duration: float = 8.0,
                                     title: str = 'SWR Event Detection Results with Zoom'):
    time = np.arange(len(signal_data)) / fs
    idx_full_start, idx_full_end = 0, len(signal_data)

    zoom_start = 0.0
    zoom_end = min(zoom_duration + 0.5, len(signal_data) / fs)
    if events:
        event_times = np.array([e.peak_sample / fs for e in events])
        best_start = 0.0
        best_count = 0
        for t in event_times:
            count = np.sum((event_times >= t) & (event_times < t + zoom_duration))
            if count > best_count:
                best_count = int(count)
                best_start = t
        zoom_start = max(0.0, best_start - 0.5)
        zoom_end = min(len(signal_data) / fs, best_start + zoom_duration + 0.5)

    idx_zoom_start = int(zoom_start * fs)
    idx_zoom_end = min(int(zoom_end * fs), len(signal_data))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=False)

    ax1.plot(time[idx_full_start:idx_full_end], signal_data[idx_full_start:idx_full_end], color='black')
    valid_count = 0
    invalid_count = 0
    for event in events:
        if event.start_sample >= idx_full_start and event.end_sample <= idx_full_end:
            event_time = np.arange(event.start_sample, event.end_sample + 1) / fs
            if event.is_valid:
                ax1.axvspan(event_time[0], event_time[-1], alpha=0.35, color='green', label='Valid SWR' if valid_count == 0 else '')
                ax1.plot(time[event.peak_sample], signal_data[event.peak_sample], 'r*', markersize=6)
                valid_count += 1
            else:
                ax1.axvspan(event_time[0], event_time[-1], alpha=0.15, color='red', label='Candidate (rejected)' if invalid_count == 0 else '')
                invalid_count += 1

    if valid_count == 0 and invalid_count == 0:
        ax1.text(0.5, 0.5, 'No events found in the full recording', transform=ax1.transAxes,
                 ha='center', va='center', color='gray', fontsize=12)

    ax1.set_ylabel('Amplitude (μV)')
    ax1.set_title('SWR Event Detection Results')
    ax1.set_xlim(time[idx_full_start], time[idx_full_end - 1])
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)

    ax2.plot(time[idx_zoom_start:idx_zoom_end], signal_data[idx_zoom_start:idx_zoom_end], color='black')
    valid_count = 0
    invalid_count = 0
    for event in events:
        if event.start_sample >= idx_zoom_start and event.end_sample <= idx_zoom_end:
            event_time = np.arange(event.start_sample, event.end_sample + 1) / fs
            if event.is_valid:
                ax2.axvspan(event_time[0], event_time[-1], alpha=0.35, color='green', label='Valid SWR' if valid_count == 0 else '')
                ax2.plot(time[event.peak_sample], signal_data[event.peak_sample], 'r*', markersize=8)
                valid_count += 1
            else:
                ax2.axvspan(event_time[0], event_time[-1], alpha=0.15, color='red', label='Candidate (rejected)' if invalid_count == 0 else '')
                invalid_count += 1

    if valid_count == 0 and invalid_count == 0:
        ax2.text(0.5, 0.5, 'No events found in the zoom window', transform=ax2.transAxes,
                 ha='center', va='center', color='gray', fontsize=12)

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude (μV)')
    ax2.set_title(f'Zoomed SWR Detection Results ({zoom_start:.2f}s - {zoom_end:.2f}s)')
    ax2.set_xlim(time[idx_zoom_start], time[idx_zoom_end])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=False)
    plt.pause(0.001)


def plot_envelope_and_candidates(ripple_filtered: np.ndarray, envelope: np.ndarray, 
                                peaks: np.ndarray, fs: int, 
                                threshold_factor: float = 5.0,
                                time_window: tuple[float, float] = None):
    """Visualize the ripple band envelope, peaks, and threshold for debugging detection."""
    total_samples = len(envelope)
    time = np.arange(total_samples) / fs
    
    # Dynamically resolve indices based on data bounds
    if time_window is not None:
        idx_start = max(0, int(time_window[0] * fs))
        idx_end = min(total_samples, int(time_window[1] * fs))
    else:
        idx_start, idx_end = 0, total_samples
        
    t_min, t_max = time[idx_start], time[idx_end - 1]
    
    median_value = np.median(envelope)
    threshold = threshold_factor * median_value
    half_threshold = threshold / 2
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    # --- Top Subplot: Filtered Signal ---
    ax1.plot(time[idx_start:idx_end], ripple_filtered[idx_start:idx_end], 
             label='Ripple filtered (80-250 Hz)', color='darkblue', linewidth=0.5)
    ax1.set_ylabel('Amplitude (uV)', fontsize=10)
    ax1.set_title('Ripple Band Filtered Signal', fontsize=11, loc='left', weight='bold')
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, alpha=0.3, ls=':')
    ax1.set_xlim(t_min, t_max)
    
    # --- Bottom Subplot: Envelope and Detection ---
    ax2.plot(time[idx_start:idx_end], envelope[idx_start:idx_end], 
             label='Envelope (Hilbert)', color='darkgreen', linewidth=1)
    ax2.axhline(y=threshold, color='crimson', linestyle='--', linewidth=1.5, 
                label=f'Detection threshold ({threshold_factor}x median)')
    ax2.axhline(y=half_threshold, color='orange', linestyle='--', linewidth=1.2,
                label=f'Event boundary ({threshold_factor/2}x median)')
    ax2.axhline(y=median_value, color='gray', linestyle=':', linewidth=1,
                label='Median value')
    
    # Safely filter peaks inside the selected window bounds
    valid_peaks = peaks[(peaks >= idx_start) & (peaks < idx_end)]
    ax2.plot(time[valid_peaks], envelope[valid_peaks], 'r.', markersize=5, 
             label=f'Detected peaks ({len(valid_peaks)})')
    
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Envelope magnitude', fontsize=10)
    ax2.set_title('Envelope with Detection Thresholds and Peaks', fontsize=11, loc='left', weight='bold')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, alpha=0.3, ls=':')
    ax2.set_xlim(t_min, t_max)
    
    plt.tight_layout()
    plt.show(block=False)
