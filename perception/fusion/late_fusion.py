"""
late_fusion.py
Confidence-weighted late fusion over the emotion channels.

THE CONTRACT: every channel emits {probs6, available, quality}.
  probs6    6-class probability vector in SIX order. Never a label -- no
            channel ever makes a decision, only fusion does.
  available whether the channel could read anything at all this frame
  quality   [0,1] how much to trust this particular reading

TWO COMBINATION RULES, DELIBERATELY DIFFERENT:

  PEERS (face, audio) join as a weighted SUM. Kittler et al. (1998) show the
  sum rule is markedly more robust to estimation error in the individual
  posteriors than the product rule, because a product lets one near-zero
  estimate veto the whole decision. Face and audio are noisy peers; one bad
  frame must not zero out a class.

  BODY joins as a multiplicative PRIOR. Pose openness and activity are not a
  class-discriminative likelihood -- they constrain the arousal/valence region
  rather than voting for a class. A soft veto is what is wanted there, so the
  product rule is correct precisely because it can suppress.

WEIGHTS ARE DERIVED, NOT LEARNED. w_m comes from the quality each channel
reports this frame, not from trained parameters. That is the deployment
argument: when a channel degrades, its weight drops automatically. Grounded in
reliability-weighted cue integration (Ernst & Banks 2002), where cues are
weighted by inverse variance; `quality` is the practical proxy for precision.

CALIBRATION CAVEAT: weighting by quality assumes the incoming posteriors are
comparable across channels. Raw softmax outputs are systematically
overconfident, so temperature scaling per channel is the outstanding
prerequisite before quality weighting can be claimed as principled rather than
heuristic. Noted here so it is not forgotten at writing-up time.
"""
from __future__ import annotations

import numpy as np

SIX = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']
N = 6

# Floor on a channel's weight. A channel that is available but reports quality
# 0.0 should still whisper, not be silently deleted -- and it stops the weight
# vector summing to zero when every channel is poor.
MIN_WEIGHT = 1e-3


def uniform() -> np.ndarray:
    """Maximum-entropy fallback: nothing was readable this frame."""
    return np.full(N, 1.0 / N)


def fuse(peers, body_prior=None) -> np.ndarray:
    """peers: [(probs6, quality), ...] for channels that are AVAILABLE now.

    The caller filters on availability -- an unavailable channel is simply
    absent from the list. That is what modality dropout is, mechanically:
    not a zero vector, not an imputed guess, just one fewer term in the sum.

    body_prior: 6-vector centred on 1.0 from pose_emotion.body_prior(), or None
    when the body cannot be read. All-ones is the identity, so an unreadable
    body changes nothing.

    Returns a normalised 6-vector in SIX order.
    """
    if peers:
        w = np.array([max(float(q), MIN_WEIGHT) for _, q in peers], dtype=float)
        w /= w.sum()
        post = np.zeros(N, dtype=float)
        for wi, (p, _q) in zip(w, peers):
            post += wi * np.asarray(p, dtype=float)
    else:
        post = uniform()

    if body_prior is not None:
        post = post * np.asarray(body_prior, dtype=float)

    s = post.sum()
    return post / s if s > 0 else uniform()


def label_of(probs) -> str:
    return SIX[int(np.argmax(probs))]


def weights_of(peers) -> list[float]:
    """The normalised weights fuse() would apply. Exposed for the overlay and
    for logging -- being able to SHOW that the face weight collapsed when
    someone turned away is the evidence for the whole design."""
    if not peers:
        return []
    w = np.array([max(float(q), MIN_WEIGHT) for _, q in peers], dtype=float)
    return list(w / w.sum())


if __name__ == "__main__":
    face = np.array([0.05, 0.05, 0.05, 0.70, 0.10, 0.05])   # says happy
    audio = np.array([0.60, 0.05, 0.10, 0.10, 0.10, 0.05])  # says anger

    print("class order:", SIX)
    print("face only        ", np.round(fuse([(face, 0.9)]), 3))
    print("both, face better", np.round(fuse([(face, 0.9), (audio, 0.3)]), 3))
    print("both, audio better", np.round(fuse([(face, 0.2), (audio, 0.9)]), 3))
    print("face dropped out ", np.round(fuse([(audio, 0.9)]), 3))
    print("nothing readable ", np.round(fuse([]), 3))

    closed_calm = [1.0, 1.0, 1.0, 1.0, 1.8, 1.8]   # sad/neutral boosted
    print("face + body prior", np.round(fuse([(face, 0.9)], closed_calm), 3))
