from copy import deepcopy
import numpy as np
try:
    from filterpy.common import Saver
except ImportError:
    Saver = None

try:
    from scipy.spatial.distance import cdist
except ImportError:
    cdist = None


def track_objects():
    """Track objects in a video."""
    pass


class Tracker:
    """Track objects in a video."""

    def __init__(self):
        pass


class Tracklet:
    def __init__(self, initial_detection, kf=None, funcs=None):
        self.kf = kf
        self.initial_px = getattr(initial_detection, 'px', 0)
        self.initial_py = getattr(initial_detection, 'py', 0)

        self.saver = Saver(kf) if (Saver and kf) else None
        self._detections = [initial_detection]
        self._funcs = funcs or []

    def predict(self):
        if self.kf:
            self.kf.predict()

    def update(self, candidate_detection):
        candidate_detection = deepcopy(candidate_detection)
        xy = np.array([candidate_detection.px, candidate_detection.py])
        if self.kf:
            self.kf.update(xy)

        self._detections.append(candidate_detection)
        self.save()

    def save(self):
        if self.saver:
            self.saver.save()

    def associate(self, detections, return_cost=False):
        if not cdist:
            return (None, []) if return_cost else None
        cost_vec = cdist(detections, [self], self._funcs)

        if len(cost_vec) == 0:
            if return_cost:
                return None, cost_vec
            return None
        if return_cost:
            return detections[cost_vec.argmin()], cost_vec
        return detections[cost_vec.argmin()]

    @property
    def detections(self):
        return self._detections

    @property
    def _feature_xy(self):
        if len(self.detections) == 0:
            return np.array([self.initial_px, self.initial_py])
        return np.array([self.detections[-1].px, self.detections[-1].py])

    @property
    def _feature_motion(self):
        pass

    @property
    def _feature_appearance(self):
        pass
