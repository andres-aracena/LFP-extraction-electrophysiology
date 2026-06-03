"""
Sharp Wave Ripple (SWR) Event Detection
Methodology based on ripple band filtering and envelope analysis using Hilbert transform.
"""

import numpy as np
import pandas as pd
from scipy import signal
from dataclasses import dataclass
from typing import Tuple, List, Optional
from filters import (
    apply_butterworth_filter,
    compute_hilbert_envelope,
    compute_power_from_signal,
    compute_cycle_count_and_frequency,
    detect_peaks_with_threshold,
    find_event_boundaries,
    merge_overlapping_events,
)


@dataclass
class SWREvent:
    """Data class to store SWR event information."""
    start_sample: int
    end_sample: int
    peak_sample: int
    peak_amplitude: float
    num_cycles: float
    mean_frequency: float
    ripple_power: float
    ripple_power_ref: float
    control_power: float
    is_valid: bool
    validation_notes: str


class SWRDetector:
    """
    Sharp Wave Ripple detector with filtering and event validation.
    """
    
    def __init__(self, fs: int = 1250, ripple_band: Tuple[int, int] = (80, 250),
                 control_band: Tuple[int, int] = (200, 500), filter_order: int = 4):
        """
        Initialize SWR detector with filter parameters.
        
        Args:
            fs: Sampling frequency (Hz)
            ripple_band: Ripple frequency band (Hz)
            control_band: Control high-frequency band (Hz)
            filter_order: Butterworth filter order
        """
        self.fs = fs
        self.ripple_band = ripple_band
        self.control_band = control_band
        self.filter_order = filter_order
        self.nyquist = fs / 2
        
    def load_lfp_data(self, filepath: str, n_channels: int = 67, dtype: str = 'int16') -> np.ndarray:
        """
        Load binary LFP data and reshape to (samples, channels).
        
        Args:
            filepath: Path to .lfp binary file
            n_channels: Number of channels
            dtype: Data type
            
        Returns:
            LFP data array (samples, channels)
        """
        data = np.fromfile(filepath, dtype=dtype)
        n_samples = len(data) // n_channels
        data = data.reshape((n_samples, n_channels))
        return data.astype(np.float32)

    def load_lfp_dataframe(self, filepath: str, n_channels: int = 67, dtype: str = 'int16') -> pd.DataFrame:
        """
        Load binary LFP data into a pandas DataFrame with time index.
        
        Args:
            filepath: Path to .lfp binary file
            n_channels: Number of channels in the file
            dtype: Data type of the binary file
        Returns:
            DataFrame with columns CH_0..CH_{n_channels-1} and index in seconds
        """
        data = self.load_lfp_data(filepath, n_channels=n_channels, dtype=dtype)
        n_samples = data.shape[0]
        columns = [f'CH_{i}' for i in range(data.shape[1])]
        index = np.arange(n_samples) / self.fs
        df = pd.DataFrame(data, columns=columns, index=index)
        df.index.name = 'Time (s)'
        return df

    def select_first_channels(self, df: pd.DataFrame, n_keep: int = 64) -> pd.DataFrame:
        """
        Select the first top channels of a DataFrame and discard the rest.

        Args:
            df: Full LFP DataFrame
            n_keep: Number of channels to keep

        Returns:
            DataFrame with only the first n_keep channels
        """
        return df.iloc[:, :n_keep]

    def compute_welch_band_power(self, signal_data: np.ndarray, lowcut: int, highcut: int) -> float:
        """Estimate power in a frequency band using Welch's method."""
        if len(signal_data) < self.fs * 2:
            return 0.0
        nperseg = min(len(signal_data), 4 * self.fs)
        noverlap = nperseg // 2
        freqs, psd = signal.welch(signal_data, fs=self.fs, window='hann', nperseg=nperseg,
                                  noverlap=noverlap, scaling='density')
        band_mask = (freqs >= lowcut) & (freqs <= highcut)
        if not np.any(band_mask):
            return 0.0
        x = freqs[band_mask]
        y = psd[band_mask]
        return np.sum((y[:-1] + y[1:]) * (x[1:] - x[:-1]) * 0.5)

    def compute_channel_ripple_score(self, signal_data: np.ndarray) -> float:
        """Compute the ripple band score for a full channel using Welch power."""
        ripple_power = self.compute_welch_band_power(signal_data, 100, 250)
        surround_power = self.compute_welch_band_power(signal_data, 70, 300)
        return ripple_power / surround_power if surround_power > 0 else 0.0

    def compute_channel_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score each channel by ripple band power ratio."""
        rows = []
        for channel in df.columns:
            signal_data = df[channel].values
            rows.append((channel,
                         self.compute_channel_ripple_score(signal_data),
                         self.compute_welch_band_power(signal_data, 100, 250)))
        return pd.DataFrame(rows, columns=['channel', 'ripple_score', 'ripple_power']).sort_values(
            by='ripple_score', ascending=False
        )

    def choose_pyramidal_channel(self, scores: pd.DataFrame) -> str:
        """Choose the pyramidal reference channel with highest ripple band score."""
        
        #return 'CH_49'
        return scores.sort_values('ripple_score', ascending=False)['channel'].iloc[0]

    def choose_noise_reference_channel(self, scores: pd.DataFrame) -> str:
        """Choose the noise reference channel with lowest ripple band score."""
        
        #return 'CH_14'
        #return scores.sort_values('ripple_power', ascending=True)['channel'].iloc[0]
        return scores.sort_values('ripple_score', ascending=True)['channel'].iloc[0]

    def validate_swr_event(self, event: SWREvent, ripple_power_ref: float,
                          control_power: float) -> Tuple[bool, str]:
        """
        Apply paper validation criteria:
        (1) Ripple band power in pyramidal channel > 2x noise reference channel
            Confirms events have stronger presence in pyramidal layer
        (2) Mean frequency > 100 Hz
            Ensures events are within ripple frequency range
        (3) At least 4 complete ripple cycles
            Ensures sufficient ripple content
        (4) Ripple band power > 2x control band power (200-500 Hz)
            Confirms ripple specificity
        """
        notes = []
        is_valid = True

        # Criterion 1: Power ratio between pyramidal and noise reference
        if event.ripple_power <= 2 * ripple_power_ref:
            is_valid = False
            notes.append(f"Power ratio (pyram/noise) failed: {event.ripple_power:.2f} <= 2*{ripple_power_ref:.2f}")

        # Criterion 2: Mean frequency > 100 Hz
        if event.mean_frequency <= 100:
            is_valid = False
            notes.append(f"Mean frequency too low: {event.mean_frequency:.2f} Hz (< 100 Hz)")

        # Criterion 3: At least 4 complete cycles
        if event.num_cycles < 4:
            is_valid = False
            notes.append(f"Insufficient cycles: {event.num_cycles:.2f} < 4")

        # Criterion 4: Ripple power > 2x control band power
        if event.ripple_power <= 2 * control_power:
            is_valid = False
            notes.append(f"Ripple vs control power failed: {event.ripple_power:.2f} <= 2*{control_power:.2f}")

        return is_valid, "; ".join(notes) if notes else "Valid"

    def detect_swr_events(self, pyramidal_channel: np.ndarray,
                         noise_reference_signal: np.ndarray) -> List[SWREvent]:
        """
        Full SWR detection pipeline.

        Args:
            pyramidal_channel: LFP channel from pyramidal layer (1D array)
            noise_reference_signal: Reference channel with low ripple content for differential referencing.

        Returns:
            List of SWREvent objects
        """
        # Differential signal: pyramidal minus noise reference
        differential_signal = pyramidal_channel - noise_reference_signal

        ripple_filtered = apply_butterworth_filter(
            differential_signal,
            self.fs,
            self.ripple_band[0],
            self.ripple_band[1],
            self.filter_order,
        )
        control_filtered = apply_butterworth_filter(
            differential_signal,
            self.fs,
            self.control_band[0],
            self.control_band[1],
            self.filter_order,
        )

        # Reference filtered signal for validation (noise reference)
        ripple_filtered_ref = apply_butterworth_filter(
            noise_reference_signal,
            self.fs,
            self.ripple_band[0],
            self.ripple_band[1],
            self.filter_order,
        )

        ripple_envelope, ripple_phase = compute_hilbert_envelope(ripple_filtered)
        envelope_median = np.median(ripple_envelope)
        threshold = 5.0 * envelope_median
        min_distance_samples = int(0.020 * self.fs)
        peaks = detect_peaks_with_threshold(
            ripple_envelope, threshold_factor=5.0, min_distance_samples=min_distance_samples
        )
        events_boundaries = find_event_boundaries(
            ripple_envelope,
            peaks,
            threshold_factor=5.0,
            window_size=int(0.5 * self.fs),
        )
        events_before_merge = len(events_boundaries)
        
        # Merge overlapping or very close events, keeping only the highest peak in each group
        events_boundaries = merge_overlapping_events(events_boundaries, ripple_envelope)
        events_after_merge = len(events_boundaries)

        print(f"[DEBUG] Envelope median={envelope_median:.6f}, threshold=5.0*median={threshold:.6f}")
        print(f"[DEBUG] Peak candidates found={len(peaks)}, events before merge={events_before_merge}, events after merge={events_after_merge}")

        swr_events: List[SWREvent] = []
        for onset, peak, offset in events_boundaries:
            duration_sec = (offset - onset) / self.fs
            num_cycles, mean_freq = compute_cycle_count_and_frequency(
                ripple_phase, onset, offset, duration_sec
            )

            ripple_power_det = compute_power_from_signal(ripple_filtered[onset:offset + 1])
            ripple_power_ref = compute_power_from_signal(ripple_filtered_ref[onset:offset + 1])
            control_power = compute_power_from_signal(control_filtered[onset:offset + 1])

            event = SWREvent(
                start_sample=onset,
                end_sample=offset,
                peak_sample=peak,
                peak_amplitude=ripple_envelope[peak],
                num_cycles=num_cycles,
                mean_frequency=mean_freq,
                ripple_power=ripple_power_det,
                ripple_power_ref=ripple_power_ref,
                control_power=control_power,
                is_valid=False,
                validation_notes="",
            )

            is_valid, notes = self.validate_swr_event(event, ripple_power_ref, control_power)
            event.is_valid = is_valid
            event.validation_notes = notes
            swr_events.append(event)

        return swr_events

    def print_summary(self, events: List[SWREvent]):
        """
        Print summary of detected events with detailed breakdown.

        Args:
            events: List of SWR events
        """
        valid_events = [e for e in events if e.is_valid]
        total_events = len(events)

        print(f"\n{'='*70}")
        print(f"SWR Event Detection Summary")
        print(f"{'='*70}")
        print(f"Total candidate events: {total_events}")
        print(f"Valid events: {len(valid_events)} ({100*len(valid_events)/max(total_events, 1):.1f}%)")
        
        if total_events > 0:
            invalid_events = [e for e in events if not e.is_valid]
            if invalid_events:
                print(f"\nCandidate Rejection Summary:")
                rejection_reasons = {}
                for e in invalid_events:
                    reasons = e.validation_notes.split("; ")
                    for reason in reasons:
                        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                for reason, count in rejection_reasons.items():
                    print(f"  - {reason}: {count}")

            print(f"\nFirst 10 candidate event details:")
            for i, e in enumerate(events[:10], 1):
                duration_ms = (e.end_sample - e.start_sample) / self.fs * 1000
                status = 'VALID' if e.is_valid else 'REJECTED'
                print(
                    f"  {i}. {e.start_sample/self.fs:.3f}s-{e.end_sample/self.fs:.3f}s, "
                    f"dur={duration_ms:.1f}ms, freq={e.mean_frequency:.1f}Hz, "
                    f"cycles={e.num_cycles:.1f}, pow={e.ripple_power:.4f}, "
                    f"ref_pow={e.ripple_power_ref:.4f}, "
                    f"control_pow={e.control_power:.4f}, {status}, notes={e.validation_notes}"
                )

        if valid_events:
            print(f"\nValid Event Statistics:")
            print(f"  Mean frequency: {np.mean([e.mean_frequency for e in valid_events]):.2f} Hz")
            print(f"  Median frequency: {np.median([e.mean_frequency for e in valid_events]):.2f} Hz")
            print(f"  Mean cycles: {np.mean([e.num_cycles for e in valid_events]):.2f}")
            print(f"  Mean ripple power: {np.mean([e.ripple_power for e in valid_events]):.4f}")
            print(f"  Mean duration (ms): {np.mean([(e.end_sample - e.start_sample) / self.fs * 1000 for e in valid_events]):.2f}")
        print(f"{'='*70}\n")
