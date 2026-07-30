"""
video_perception.py
I/O-free video core: one frame plus this frame's pose detections in, one
PersonResult per tracked person out.

No camera, no depthai, no drawing, no printing -- run_video.py owns all of
that. Mirrors perception.py on the audio side.

WHY THIS FILE HAD TO EXIST: pose_live.py and emotion_live_oakd.py are separate
processes. Each opens the OAK-D, each builds its own pipeline, each gets its
own frames. The device can only be claimed once, so they cannot even run
together -- and if they could, they would be looking at different frames, so
comparing a face box against a person box would be comparing coordinates from
different moments. Cross-channel association needs the SAME pixels: one frame,
both detectors, one coordinate space, one instant. That is what this provides.

TWO ASSOCIATIONS, DIFFERENT PROBLEMS, DIFFERENT METRICS:
  across TIME  -- is this the same person as last frame?  -> IoU (PersonTracker)
  across CHANNELS -- whose face is this?                  -> containment
                     (face_pose_match; IoU is useless here because a face is a
                     few percent of a body box, so even a perfect match scores
                     near zero)

AUDIO SLOTS IN LATER. process() takes an optional `audio` channel and passes it
to fuse() as a second peer alongside face. Nothing else changes when the mic
arrives -- that is the point of the {probs6, available, quality} contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from person_tracker import PersonTracker, iou
from per_track_state import PerTrackState
from target_selector import TargetSelector
from face_pose_match import match_face
from face_detector import detect_faces_scored
from face_perception import FacePerception
from pose_features import PoseFeatures
from pose_emotion import body_prior
import late_fusion
from late_fusion import SIX


@dataclass
class PersonResult:
    track_id: int
    box: tuple                              # person box, (x1,y1,x2,y2)

    # --- face channel: the {probs6, available, quality} contract ---
    face_probs: np.ndarray | None = None
    face_quality: float | None = None
    face_box: tuple | None = None
    face_available: bool = False

    # --- body channel: a multiplicative prior, not a peer likelihood ---
    body_prior: list | None = None
    body_available: bool = False
    openness: float | None = None
    activity: float | None = None
    burst: float | None = None

    # --- fusion output ---
    fused: np.ndarray = field(default_factory=late_fusion.uniform)
    label: str = 'neutral'
    weights: list = field(default_factory=list)   # normalised peer weights, for display

    @property
    def any_channel(self) -> bool:
        return self.face_available or self.body_available


@dataclass
class VideoResult:
    people: list                            # list[PersonResult], everyone tracked
    target_id: int | None                   # attended person

    @property
    def target(self):
        for p in self.people:
            if p.track_id == self.target_id:
                return p
        return None


def _detection_index(tbox, boxes, used):
    """Which pose detection a track matched to. The tracker hands back the exact
    box it was given, so the right one scores IoU 1.0; `used` keeps it
    one-to-one so two identical boxes cannot claim the same person."""
    best_i, best_v = None, 0.0
    for i, b in enumerate(boxes):
        if i in used:
            continue
        v = iou(tbox, b)
        if v > best_v:
            best_i, best_v = i, v
    return best_i


class VideoPerception:
    """Owns the face model and all per-track state. One instance per stream.

    face_min_size: Haar's minimum face box. The default (48,48) is tuned for
        full-resolution frames; the pose passthrough is 512x288, where a face
        at ~2m is roughly 40px and would be missed entirely. Lower it for the
        NN-sized stream or the face channel silently never fires.
    """

    def __init__(self, intrinsics, model_name="enet_b0_8_best_vgaf",
                 face_min_size=(32, 32), min_contain=0.6):
        self.face = FacePerception(model_name=model_name)
        self.tracker = PersonTracker()
        self.selector = TargetSelector()
        self.pose_state = PerTrackState(lambda: PoseFeatures(intrinsics=intrinsics))
        self.face_min_size = face_min_size
        self.min_contain = min_contain

    def process(self, frame_bgr, people, audio=None) -> VideoResult:
        """frame_bgr: the frame BOTH detectors see. Must be the same array the
            person boxes were computed from, or the coordinates do not line up.
        people: [(person_box, pts), ...] for this frame, extracted from the pose
            NN by the runner. pts is the COCO-17 keypoint dict pose_features
            expects.
        audio: optional (probs6, available, quality) for the audio channel --
            one reading for the scene, since the mic array cannot yet be
            attributed to a specific tracked person. Applied to the target only.

        Returns a VideoResult. Only CONFIRMED, currently-visible tracks appear.
        """
        boxes = [b for b, _pts in people]
        pts_by_det = [pts for _b, pts in people]

        tracks = self.tracker.update(boxes)
        target = self.selector.update(tracks)

        # all faces in this frame, once -- then attributed to owners below
        faces = detect_faces_scored(frame_bgr, min_size=self.face_min_size)

        results = []
        used = set()
        for tid, tbox in tracks:
            di = _detection_index(tbox, boxes, used)
            if di is None:
                continue
            used.add(di)

            r = PersonResult(track_id=tid, box=tbox)

            # ---- body channel: this person's own temporal state ----
            op, mean, burst = self.pose_state.get(tid).update(pts_by_det[di])
            r.openness, r.activity, r.burst = op, mean, burst
            if op is not None and mean is not None and burst is not None:
                r.body_prior = body_prior(op, mean, burst)
                r.body_available = True

            # ---- face channel: only a face INSIDE this person's box ----
            fbox, fweight = match_face(tbox, faces, min_contain=self.min_contain)
            if fbox is not None:
                fr = self.face.process(frame_bgr, box=fbox, det_weight=fweight)
                if fr.found:
                    r.face_probs = fr.probs
                    r.face_quality = fr.quality
                    r.face_box = fr.box
                    r.face_available = True

            # ---- fusion ----
            peers = []
            if r.face_available:
                peers.append((r.face_probs, r.face_quality if r.face_quality is not None else 0.5))
            if audio is not None and tid == target:
                a_probs, a_avail, a_quality = audio
                if a_avail and a_probs is not None:
                    peers.append((a_probs, a_quality))

            r.weights = late_fusion.weights_of(peers)
            r.fused = late_fusion.fuse(peers, r.body_prior)
            r.label = late_fusion.label_of(r.fused)
            results.append(r)

        # without this, every person who ever passed the robot keeps a
        # PoseFeatures with a 15-frame deque alive for the whole run
        self.pose_state.evict(self.tracker.dead_ids)

        return VideoResult(people=results, target_id=target)
