"""
pose_emotion.py
Body channel: openness + activity  ->  soft emotion PRIOR over your 6 classes.

This is NOT a classifier. The body signal is coarse, so it outputs a gentle
prior (a 6-vector centred on 1.0) that MULTIPLIES into the face+audio fusion:
 > 1.0 nudges a class up, < 1.0 nudges it down, 1.0 leaves it untouched.

Grid (your mapping), from the two thresholds:
    OPEN  + ACTIVE  -> happy, anger      (expansive + high energy)
    OPEN  + CALM    -> neutral, happy    (expansive + low energy)
    CLOSED+ ACTIVE  -> anger, fear       (contracted + agitated)
    CLOSED+ CALM    -> sad, neutral      (contracted + still)
    disgust -> never touched (no body signature)

Grounding: expansive posture -> positive/high-arousal, contracted -> sad;
velocity/energy separates high- vs low-arousal (Poyo Solanas 2020; Roether 2009;
Kleinsmith & Bianchi-Berthouze 2013).
"""

SIX = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']

# thresholds (from your live readings -- tune here)
OPEN_THRESH   = 1.20    # openness > this -> open, else closed
BURST_THRESH  = 1.10   # burst > this    -> active (a sharp movement happened)
MEAN_THRESH   = 0.20   # mean  > this    -> sustained movement (steady energy)

# how hard the prior pushes, by intensity:
#   calm            -> no boost
#   active (1 high) -> BOOST_LOW      (a flick, or steady-but-not-explosive)
#   active (2 high) -> BOOST_HIGH     (sustained AND bursty = really worked up)
BOOST_LOW  = 0.4
BOOST_HIGH = 0.8

# which classes each grid cell boosts
GRID = {
    ('open',   'active'): ['happy', 'anger'],
    ('open',   'calm'):   ['neutral', 'happy'],
    ('closed', 'active'): ['anger', 'fear'],
    ('closed', 'calm'):   ['sad', 'neutral'],
}


def body_prior(openness, mean, burst):
    """openness / mean / burst: floats or None (mean & burst from PoseFeatures).
    Returns a 6-vector prior aligned to SIX. All 1.0 (no nudge) if the body
    can't be read. Boost scales with how much energy is present:
      - one of {mean, burst} high  -> BOOST_LOW  (a flick / mild activity)
      - both high                  -> BOOST_HIGH (sustained + explosive = really worked up)"""
    prior = {c: 1.0 for c in SIX}
    if openness is None or mean is None or burst is None:
        return [prior[c] for c in SIX]

    hot = (burst > BURST_THRESH) + (mean > MEAN_THRESH)   # 0, 1, or 2 energy signals
    o = 'open' if openness > OPEN_THRESH else 'closed'
    a = 'active' if hot >= 1 else 'calm'
    boost = 0.0 if hot == 0 else (BOOST_LOW if hot == 1 else BOOST_HIGH)

    for c in GRID[(o, a)]:
        prior[c] = 1.0 + boost                    # disgust never in a cell -> stays 1.0
    return [prior[c] for c in SIX]


def fuse_with_body(face_audio_probs, openness, mean, burst):
    """Multiply the face+audio fused 6-vector by the body prior and renormalise.
    face_audio_probs: list/array of 6 probs in SIX order."""
    import numpy as np
    p = np.asarray(face_audio_probs, dtype=float)
    prior = np.asarray(body_prior(openness, mean, burst), dtype=float)
    out = p * prior
    s = out.sum()
    return (out / s) if s > 0 else p


if __name__ == "__main__":
    # sanity: show the prior for each grid cell
    import numpy as np
    # (openness, mean, burst)
    cases = [("OPEN calm",        1.5, 0.05, 0.3),
             ("OPEN flick",       1.5, 0.05, 2.0),   # burst only  -> low boost
             ("OPEN worked-up",   1.5, 0.40, 2.0),   # both high   -> high boost
             ("CLOSED worked-up", 0.5, 0.40, 2.0),   # anger+fear, strong
             ("CLOSED calm",      0.5, 0.05, 0.3),
             ("no read",          None, None, None)]
    print("class order:", SIX)
    for name, o, m, b in cases:
        print(f"{name:16}", [round(x, 2) for x in body_prior(o, m, b)])