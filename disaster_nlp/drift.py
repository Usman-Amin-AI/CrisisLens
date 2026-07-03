"""Simple drift monitoring utilities.

Tracks a rolling window of recent predictions to compute the predicted-disaster
rate and average confidence. Flags when recent metrics deviate from a
configured baseline by a configurable relative threshold.
"""
from collections import deque
from typing import Deque, List, Tuple, Optional


class DriftMonitor:
    """Maintain rolling stats and detect drift relative to a baseline.

    Parameters
    - window_size: number of recent predictions to keep
    - disaster_rate_baseline: baseline disaster rate (0-1)
    - avg_confidence_baseline: baseline mean confidence (0-1)
    - threshold: relative deviation required to flag drift (e.g., 0.2 = 20%)
    """

    def __init__(
        self,
        window_size: int = 500,
        disaster_rate_baseline: float = 0.5,
        avg_confidence_baseline: float = 0.75,
        threshold: float = 0.2,
    ):
        self.window_size = window_size
        self.disaster_rate_baseline = disaster_rate_baseline
        self.avg_confidence_baseline = avg_confidence_baseline
        self.threshold = threshold

        self._labels: Deque[int] = deque(maxlen=window_size)
        self._confs: Deque[float] = deque(maxlen=window_size)

    def add_batch(self, labels: List[int], confidences: List[float]) -> None:
        """Add a batch of predictions to the rolling window.

        labels: list of 0/1 ints where 1 indicates 'disaster'
        confidences: list of floats in [0,1]
        """
        for l in labels:
            self._labels.append(int(l))
        for c in confidences:
            self._confs.append(float(c))

    def current_stats(self) -> Tuple[Optional[float], Optional[float], int]:
        """Return (disaster_rate, avg_confidence, n_samples) or (None, None, 0) if empty."""
        n = len(self._labels)
        if n == 0:
            return None, None, 0
        disaster_rate = sum(self._labels) / n
        avg_conf = sum(self._confs) / len(self._confs) if len(self._confs) > 0 else None
        return disaster_rate, avg_conf, n

    def check_drift(self) -> dict:
        """Check whether current stats deviate from baseline.

        Returns a dict with keys: `drift_detected` (bool), `disaster_rate`,
        `avg_confidence`, `n_samples`, and `reasons` (list of strings).
        """
        disaster_rate, avg_conf, n = self.current_stats()
        if n == 0:
            return {
                'drift_detected': False,
                'disaster_rate': None,
                'avg_confidence': None,
                'n_samples': 0,
                'reasons': []
            }

        reasons = []
        drift = False

        # Check disaster rate deviation
        if self.disaster_rate_baseline is not None:
            if abs(disaster_rate - self.disaster_rate_baseline) / max(self.disaster_rate_baseline, 1e-6) > self.threshold:
                drift = True
                reasons.append('disaster_rate_change')

        # Check avg confidence deviation
        if self.avg_confidence_baseline is not None and avg_conf is not None:
            if abs(avg_conf - self.avg_confidence_baseline) / max(self.avg_confidence_baseline, 1e-6) > self.threshold:
                drift = True
                reasons.append('avg_confidence_change')

        return {
            'drift_detected': drift,
            'disaster_rate': disaster_rate,
            'avg_confidence': avg_conf,
            'n_samples': n,
            'reasons': reasons
        }
