import time
import numpy as np
from wrapped_kf import WrappedKalmanFilter1D, wrap


class Track:
    """A single tracked source: its KF plus bookkeeping."""

    def __init__(self, track_id, dt, initial_angle, initial_conf):
        self.id = track_id
        self.kf = WrappedKalmanFilter1D(dt=dt)
        self.kf.update(initial_angle, initial_conf)
        self.misses = 0
        self.hits = 1
        self.age = 1
        self.confirmed = False

    @property
    def angle(self):
        return self.kf.angle

    @property
    def angle_std(self):
        return self.kf.angle_std


class MultiTracker:
    """
    Greedy nearest-neighbour multi-source DOA tracker.

    Call .step(detections) each frame.
    detections: list of (angle_rad, confidence) tuples from pick_peaks().
    Returns: list of confirmed Track objects.
    """

    def __init__(self, dt,
                 gate_deg=20.0,
                 max_misses=15,
                 min_hits=3,
                 spawn_conf=0.4,
                 inherit_ttl=30.0):
        self.dt = dt
        self.gate = np.deg2rad(gate_deg)
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.spawn_conf = spawn_conf
        self.tracks = []
        self.next_id = 0
        self._graveyard = []        # (angle_rad, id, death_time) of recently-dead confirmed tracks
        self._inherit_ttl = inherit_ttl   # seconds to remember a dead track's ID

    def step(self, detections):
        # 1) Predict every track forward
        for t in self.tracks:
            t.kf.predict()
            t.age += 1

        # 2) Greedy association, best-confidence detections pick first
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        matches = []

        det_order = sorted(unmatched_dets,
                           key=lambda i: detections[i][1],
                           reverse=True)

        for di in det_order:
            ang, _ = detections[di]
            best_ti, best_d = None, np.inf
            for ti in unmatched_tracks:
                d = abs(wrap(ang - self.tracks[ti].angle))
                if d < best_d and d < self.gate:
                    best_d, best_ti = d, ti
            if best_ti is not None:
                matches.append((best_ti, di))
                unmatched_tracks.remove(best_ti)
                unmatched_dets.remove(di)

        # 3) Update matched tracks
        for ti, di in matches:
            ang, conf = detections[di]
            self.tracks[ti].kf.update(ang, conf)
            self.tracks[ti].misses = 0
            self.tracks[ti].hits += 1
            if self.tracks[ti].hits >= self.min_hits:
                self.tracks[ti].confirmed = True

        # 4) Unmatched tracks: increment miss, drop stale
        now = time.monotonic()
        for ti in unmatched_tracks:
            self.tracks[ti].misses += 1
        for t in self.tracks:
            if t.misses >= self.max_misses and t.confirmed:
                self._graveyard.append((t.angle, t.id, now))
        self.tracks = [t for t in self.tracks if t.misses < self.max_misses]
        self._graveyard = [(a, i, ts) for a, i, ts in self._graveyard
                           if now - ts < self._inherit_ttl]

        # 5) Unmatched detections: spawn if confident enough
        for di in unmatched_dets:
            ang, conf = detections[di]
            if conf >= self.spawn_conf:
                track_id = self._inherit_id(ang, now)
                self.tracks.append(Track(track_id, self.dt, ang, conf))

        return [t for t in self.tracks if t.confirmed]

    def _inherit_id(self, angle, now):
        """Return a dead track's ID if one is nearby, else allocate a new one."""
        best_gi, best_d = None, np.inf
        for gi, (ga, gid, _) in enumerate(self._graveyard):
            d = abs(wrap(angle - ga))
            if d < best_d and d < self.gate:
                best_d, best_gi = d, gi
        if best_gi is not None:
            inherited_id = self._graveyard[best_gi][1]
            del self._graveyard[best_gi]
            return inherited_id
        new_id = self.next_id
        self.next_id += 1
        return new_id