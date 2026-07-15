import numpy as np
import math
from collections import deque

class RobustStatistics:
    def __init__(self, baseline_size: int = 50):
        self.baseline = deque(maxlen=baseline_size)

    def update(self, value: float):
        self.baseline.append(float(value))

    def ready(self) -> bool:
        return len(self.baseline) >= 10

    def compute_zscore(self, value: float) -> float:
        if not self.ready():
            return 0.0

        baseline = np.asarray(self.baseline, dtype=np.float32)
        median = np.median(baseline)
        mad = np.median(np.abs(baseline - median))
        robust_std = 1.4826 * mad + 1e-6
        z = (value - median) / robust_std
        return float(z)


class ExponentialSmoother:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.initialized = False
        self.previous = 0.0

    def update(self, value: float) -> float:
        if not self.initialized:
            self.previous = value
            self.initialized = True
            return value

        smoothed = self.alpha * value + (1 - self.alpha) * self.previous
        self.previous = smoothed
        return float(smoothed)


class HysteresisDetector:
    def __init__(self, enter_threshold: float = 2.5, exit_threshold: float = 1.2):
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.active = False

    def update(self, z: float) -> bool:
        if not self.active:
            if z >= self.enter_threshold:
                self.active = True
                return True
            return False
        else:
            if z <= self.exit_threshold:
                self.active = False
                return False
            return True


class ConsecutiveConfirmation:
    def __init__(self, required_confirmations: int):
        self.required = required_confirmations
        self.counter = 0

    def update(self, state: bool) -> bool:
        if state:
            self.counter += 1
        else:
            self.counter = 0
        return self.counter >= self.required
