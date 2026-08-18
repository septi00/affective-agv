import numpy as np

from wrapped_kf import wrap


def pick_peaks(angles, scores,
               min_separation_deg=20.0,
               min_relative_height=0.5,
               max_peaks=4,
               noise_floor_percentile=50):
    """
    Find prominent peaks in a DOA spatial spectrum.

    angles: 1D array of candidate angles in radians (length G)
    scores: 1D array of spectrum scores at each angle (length G)

    Returns: list of (angle_rad, confidence) tuples, confidence in [0, 1].
    """
    angles = np.asarray(angles)
    scores = np.asarray(scores)

    # 1) Normalize to [0, 1] using a percentile as the floor
    floor = np.percentile(scores, noise_floor_percentile)
    peak = scores.max()
    if peak - floor < 1e-9:
        return []
    normed = np.clip((scores - floor) / (peak - floor), 0.0, 1.0)

    # 2) Local maxima on a circular grid
    G = len(angles)
    candidates = []
    for i in range(G):
        l, r = (i - 1) % G, (i + 1) % G
        if (normed[i] >= min_relative_height and
            normed[i] >= normed[l] and
            normed[i] >= normed[r]):
            candidates.append((normed[i], angles[i]))

    # 3) Sort high to low
    candidates.sort(reverse=True)

    # 4) Non-maximum suppression
    min_sep = np.deg2rad(min_separation_deg)
    kept = []
    for score, ang in candidates:
        if all(abs(wrap(ang - k_ang)) >= min_sep for _, k_ang in kept):
            kept.append((score, ang))
        if len(kept) >= max_peaks:
            break

    return [(ang, float(score)) for score, ang in kept]