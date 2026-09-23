"""Compatibility entry point plus the notebook-compatible ripple detector."""

# Preserve the pre-existing detector API, which is maintained in the sibling
# module, while adding the exact sleep_exploration notebook workflow below.
from swr_detection import *  # noqa: F401,F403


class NotebookRippleDetector(SWRDetector):
    """Detector configured exactly as in ``sleep_exploration.ipynb``."""

    def detect(self, lfp_channel):
        lfp_channel = np.asarray(lfp_channel, dtype=float)
        if lfp_channel.ndim != 1:
            raise ValueError('lfp_channel must be one-dimensional.')
        if self.fs <= 900:
            raise ValueError('The notebook 250--450 Hz control band requires fs > 900 Hz.')

        ripple_b, ripple_a = signal.butter(4, [100, 250], btype='bandpass', fs=self.fs)
        ripple_band = signal.filtfilt(ripple_b, ripple_a, lfp_channel)
        control_b, control_a = signal.butter(4, [250, 450], btype='bandpass', fs=self.fs)
        control_band = signal.filtfilt(control_b, control_a, lfp_channel)
        analytic = signal.hilbert(ripple_band)
        envelope = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        control_envelope = np.abs(signal.hilbert(control_band))
        median = np.median(envelope)
        threshold = 3 * median
        half_threshold = 1.5 * median
        peaks, _ = signal.find_peaks(envelope, height=threshold, distance=int(.020 * self.fs))
        time = np.arange(len(lfp_channel)) / self.fs
        rows = []
        for peak in peaks:
            onset = offset = peak
            while onset > 0 and envelope[onset] >= half_threshold:
                onset -= 1
            while offset < len(envelope) - 1 and envelope[offset] >= half_threshold:
                offset += 1
            duration = (offset - onset) / self.fs
            if duration <= 0:
                continue
            cycles = np.degrees(phase[offset] - phase[onset]) / 360.0
            mean_frequency = cycles / duration
            ripple_power = np.mean(envelope[onset:offset]) ** 2
            control_power = np.mean(control_envelope[onset:offset]) ** 2
            rows.append({
                'Peak_Sample': peak, 'Onset_Time(s)': time[onset],
                'Offset_Time(s)': time[offset], 'Duration(s)': duration,
                'Cycles': cycles, 'Mean_Freq(Hz)': mean_frequency,
                'Ripple_Power': ripple_power, 'Ctrl_Power': control_power,
                'Is_Valid_SWR': mean_frequency > 100 and cycles >= 4
                and ripple_power >= 1.5 * control_power,
            })
        columns = ['Peak_Sample', 'Onset_Time(s)', 'Offset_Time(s)', 'Duration(s)',
                   'Cycles', 'Mean_Freq(Hz)', 'Ripple_Power', 'Ctrl_Power', 'Is_Valid_SWR']
        return pd.DataFrame(rows, columns=columns), ripple_band, envelope
