"""
Filtering and envelope-based utilities for SWR detection.
"""

import numpy as np
from scipy import signal
from scipy.signal import iirnotch, filtfilt


def apply_butterworth_filter(signal_data: np.ndarray, fs: int, lowcut: int, highcut: int,
                             filter_order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter."""
    if len(signal_data) < 3:
        return signal_data.astype(np.float32)

    b, a = signal.butter(filter_order, [lowcut, highcut], btype='band', fs=fs)
    max_padlen = len(signal_data) - 1
    default_padlen = 3 * max(len(b), len(a))
    padlen = min(default_padlen, max_padlen)
    padlen = max(padlen, 1)
    return signal.filtfilt(b, a, signal_data, padlen=padlen)


def apply_harmonic_notch_filters(signal_data: np.ndarray, fs: int) -> np.ndarray:
    """Apply sequential 60 Hz and 180 Hz notch filtering with zero-phase filtering."""
    if len(signal_data) < 3:
        return signal_data.astype(np.float32)

    b60, a60 = iirnotch(w0=60.0, Q=30.0, fs=fs)
    b180, a180 = iirnotch(w0=180.0, Q=60.0, fs=fs)

    max_padlen = len(signal_data) - 1
    default_padlen = 3 * max(len(b60), len(a60), len(b180), len(a180))
    padlen = min(default_padlen, max_padlen)
    padlen = max(padlen, 1)

    cleaned = filtfilt(b60, a60, signal_data, padlen=padlen)
    cleaned = filtfilt(b180, a180, cleaned, padlen=padlen)
    return cleaned.astype(np.float32)


def apply_harmonic_notch_filters_to_dataframe(df, fs: int):
    """Apply sequential notch filters to each channel in a DataFrame."""
    cleaned = df.copy()
    for channel in cleaned.columns:
        cleaned[channel] = apply_harmonic_notch_filters(cleaned[channel].values, fs)
    return cleaned


def compute_hilbert_envelope(signal_data: np.ndarray):
    """Compute analytic signal envelope and unwrapped phase in radians."""
    analytic = signal.hilbert(signal_data)
    envelope = np.abs(analytic)
    phase = np.angle(analytic)  # radians, range -pi..pi
    phase_unwrapped = np.unwrap(phase)  # unwrap in radians
    return envelope, phase_unwrapped


def detect_peaks_with_threshold(envelope: np.ndarray, threshold_factor: float = 5.0,
                                min_distance_samples: int = 25) -> np.ndarray:
    """Detect peaks above a median-based threshold."""
    median_value = np.median(envelope)
    threshold = threshold_factor * median_value
    peaks, _ = signal.find_peaks(envelope, height=threshold, distance=min_distance_samples)
    return peaks


def find_event_boundaries(envelope: np.ndarray, peaks: np.ndarray,
                          threshold_factor: float = 5.0, window_size: int = 100,
                          min_duration: int = 20) -> list[tuple[int, int, int]]:
    """Find onset and offset boundaries for detected ripple peaks."""
    median_value = np.median(envelope)
    threshold = threshold_factor * median_value
    half_threshold = threshold / 2

    events = []
    for peak in peaks:
        # Find onset: search backward from peak until envelope falls below half_threshold
        onset = max(0, peak - window_size)
        for i in range(peak - 1, max(-1, peak - window_size - 1), -1):
            if envelope[i] < half_threshold:
                onset = i + 1
                break

        # Find offset: search forward from peak until envelope falls below half_threshold
        offset = min(len(envelope) - 1, peak + window_size)
        for i in range(peak + 1, min(len(envelope), peak + window_size + 1)):
            if envelope[i] < half_threshold:
                offset = i - 1
                break

        if onset < peak < offset and (offset - onset) >= min_duration:
            events.append((onset, peak, offset))

    return events


def merge_overlapping_events(events: list[tuple[int, int, int]], 
                            envelope: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Merge events with overlapping or very close boundaries.
    Per paper: if multiple peaks occur within 20ms, only the highest peak is retained.
    
    Args:
        events: List of (onset, peak, offset) tuples
        envelope: Ripple envelope array for peak height comparison
    
    Returns:
        List of merged (onset, peak, offset) tuples
    """
    if not events:
        return events
    
    # Sort events by onset time
    sorted_events = sorted(events, key=lambda x: x[0])
    
    merged = []
    current_group = [sorted_events[0]]
    
    for i in range(1, len(sorted_events)):
        event = sorted_events[i]
        last_event = current_group[-1]
        
        # Check if events overlap or are within 20ms (converted to samples: 0.020 * fs)
        overlap_threshold = int(0.020 * 1250)  
        
        if event[0] <= last_event[2] + overlap_threshold:
            # Events overlap or are very close - add to current group
            current_group.append(event)
        else:
            # Events are far apart - finalize current group and start new one
            # For the current group, keep only the event with highest peak
            best_event = max(current_group, key=lambda x: envelope[x[1]])
            merged.append(best_event)
            current_group = [event]
    
    # Finalize last group
    if current_group:
        best_event = max(current_group, key=lambda x: envelope[x[1]])
        merged.append(best_event)
    
    return merged


def compute_cycle_count_and_frequency(phase: np.ndarray, onset: int,
                                      offset: int, duration_sec: float) -> tuple[float, float]:
    """
    Estimate cycle count and mean frequency from unwrapped phase changes.

    Cycle count = phase difference / 2π radians.
    Example: 10π radians difference = 5 cycles.
    """
    phase_diff = phase[offset] - phase[onset]
    num_cycles = np.abs(phase_diff) / (2.0 * np.pi)
    mean_frequency = num_cycles / duration_sec if duration_sec > 0 else 0.0
    return num_cycles, mean_frequency


def compute_power_from_signal(signal_data: np.ndarray) -> float:
    """Compute power from an already filtered signal segment."""
    if len(signal_data) < 30:
        return 0.0
    return np.mean(np.abs(signal_data)) ** 2
