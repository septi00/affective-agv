"""
proxemics.py
Affect -> personal-space radius: the policy that turns a person's emotion into
the distance the robot must keep clear around them, and assembles that into a
ground-plane circle the planner can treat as a soft cost / keep-out.

Grounded in Hall's proxemic zones (intimate <0.45 m, personal 0.45-1.2 m,
social 1.2-3.6 m, public >3.6 m), modulated by affect: negative, high-arousal
states (anger, fear) push the robot further out; positive affect (happy) lets
it approach closer; neutral sits at the personal/social boundary.

The radius is a probability-weighted expectation over the WHOLE fused vector,
not a switch on the argmax label -- so it moves continuously as affect shifts
and never jitters when two emotions sit close together. Feed it
PersonResult.fused.

IMPORTANT: these are a principled PRIOR, not validated distances. The human
study is what tunes BASE_RADIUS_M -- every number here is meant to be moved.

Pure module: numpy only, no torch / depthai. Runs on either Pi.
"""
from dataclasses import dataclass

import numpy as np

# Class order MUST match late_fusion.SIX and the SER / face heads.
SIX = ["anger", "disgust", "fear", "happy", "neutral", "sad"]

# Per-emotion comfort radius, metres. Hall-grounded, affect-modulated, tunable.
BASE_RADIUS_M = {
    "anger":   2.4,   # de-escalate: keep well clear (upper social zone)
    "fear":    2.0,   # high space need: don't loom
    "disgust": 1.6,   # mild avoidance
    "sad":     1.2,   # personal zone: present but not crowding
    "neutral": 1.2,   # default: personal/social boundary
    "happy":   0.9,   # positive affect: closer approach tolerated
}

# Absolute clamps (Hall personal .. social), metres. Safety floor + ceiling.
R_MIN = 0.45
R_MAX = 3.6

_BASE_VEC = np.array([BASE_RADIUS_M[e] for e in SIX])


def radius_from_probs(fused) -> float:
    """Personal-space radius, metres, from a 6-class fused probability vector
    (order = SIX). Probability-weighted expectation over per-emotion comfort
    distances, clamped to Hall's [personal, social] band.

    A near-uniform (uncertain) vector blends toward the mean comfort distance,
    which sits in the personal/social zone -- a sensible default for when the
    robot cannot read the affect confidently.
    """
    p = np.asarray(fused, dtype=float)
    s = p.sum()
    if s <= 0 or not np.isfinite(s):
        return float(np.clip(_BASE_VEC.mean(), R_MIN, R_MAX))
    r = float((p / s) @ _BASE_VEC)
    return float(np.clip(r, R_MIN, R_MAX))


def radius_from_label(label: str) -> float:
    """Fallback for callers that only have the argmax label, not the vector.
    Prefer radius_from_probs -- the bare label throws away the confidence that
    makes the radius move smoothly."""
    return float(np.clip(BASE_RADIUS_M.get(label, BASE_RADIUS_M["neutral"]),
                         R_MIN, R_MAX))


class RadiusSmoother:
    """Per-person temporal smoothing of the proxemics radius.

    Affect is recomputed from scratch every frame, so a raw radius pops -- a
    blink or a head turn flips the read and the circle jumps between, say,
    1.2 m and 2.4 m several times a second while the person hasn't moved.
    The planner would translate that into visible forward/back twitching.
    This glides the radius instead.

    ASYMMETRIC BY DESIGN: growing is fast, shrinking is slow. If someone
    becomes agitated the robot should back off promptly; but it must not lunge
    inward the instant one noisy frame reads calm. Erring toward the larger
    radius is the safe direction for personal space.
    """

    def __init__(self, alpha_grow: float = 0.5, alpha_shrink: float = 0.1):
        self.alpha_grow = alpha_grow
        self.alpha_shrink = alpha_shrink
        self._r: float | None = None

    def update(self, radius: float) -> float:
        r = float(radius)
        if self._r is None:          # first sighting: adopt it, nothing to blend
            self._r = r
            return self._r
        a = self.alpha_grow if r > self._r else self.alpha_shrink
        self._r = a * r + (1 - a) * self._r
        return self._r

    @property
    def value(self) -> float | None:
        return self._r

    def reset(self):
        self._r = None


@dataclass
class ProxemicsCircle:
    """One person's keep-clear zone, on the robot base-frame ground plane."""
    track_id: int
    cx: float       # base-frame ground x (forward), metres
    cy: float       # base-frame ground y (left), metres
    radius: float   # keep-clear radius, metres
    label: str      # dominant emotion, for logging / debug only


def circle_for(track_id, ground_xy, fused, label, smoother=None) -> ProxemicsCircle | None:
    """Assemble one person's proxemics circle for the planner.

    ground_xy: (x_fwd, y_left) base-frame position from robot_frame.ground_xy,
        or None when this person has no valid metric position this frame.
    fused:     6-class probability vector (order = SIX).
    label:     dominant-emotion string, carried through for debug.
    smoother:  this person's RadiusSmoother (from PerTrackState), or None for
        the raw per-frame radius. Pass one in any live loop -- without it the
        circle pops frame to frame.

    Returns a ProxemicsCircle, or None if position is unknown (the planner
    should hold the person's last good circle or skip it, never invent one).
    """
    if ground_xy is None:
        return None
    r = radius_from_probs(fused)
    if smoother is not None:
        r = smoother.update(r)
    return ProxemicsCircle(
        track_id=int(track_id),
        cx=float(ground_xy[0]),
        cy=float(ground_xy[1]),
        radius=r,
        label=str(label),
    )