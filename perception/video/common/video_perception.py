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

AUDIO: process() takes an optional `audio` channel (scene-level {probs6,
available, quality}) and an optional `doa_tracks` (AudioBridge.tracks()).
With doa_tracks supplied, audio is attributed to whichever tracked person's
on-screen bearing matches the currently-active DOA track -- see bearing.py.
Without it (doa_tracks=None, the default), audio falls back to the old
behaviour of joining the attended person only, for callers that haven't
wired DOA through yet.
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
from pose_features import PoseFeatures, torso_centroid
from pose_emotion import body_prior
from emotion_smoother import EmaSmoother
import proxemics
import robot_frame
import late_fusion
import bearing
from late_fusion import SIX


@dataclass
class PersonResult:
    track_id: int
    box: tuple                              # person box, (x1,y1,x2,y2)
    centroid_xyz: np.ndarray | None = None  # person anchor, CAMERA frame, metres (torso mean)

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
    fused: np.ndarray = field(default_factory=late_fusion.uniform)   # TEMPORALLY SMOOTHED -- use this
    fused_raw: np.ndarray = field(default_factory=late_fusion.uniform)  # this frame only, pre-EMA (debug)

    # --- navigation output: affect-scaled personal space ---
    circle: object | None = None            # ProxemicsCircle, base frame. None if no metric position.
    label: str = 'neutral'
    weights: list = field(default_factory=list)   # normalised peer weights, for display

    @property
    def any_channel(self) -> bool:
        return self.face_available or self.body_available


@dataclass
class VideoResult:
    people: list                            # list[PersonResult], everyone tracked
    target_id: int | None                   # attended person
    audio_target_id: int | None = None      # who the audio channel was attributed to this frame
    audio_cam_deg: float | None = None      # active DOA track, converted to camera frame -- calibration debug
    person_bearings: dict = field(default_factory=dict)  # {track_id: camera_bearing_deg} -- calibration debug

    @property
    def circles(self) -> list:
        """Every person's proxemics circle this frame, base frame, skipping
        anyone without a valid metric position. This is the ONLY thing the
        navigation layer needs from perception -- publish this list."""
        return [p.circle for p in self.people if p.circle is not None]

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
    mic_offset_deg, mic_sign: mic-to-camera bearing calibration -- see
        bearing.py's module docstring for what each means and how to measure
        them. Only matter when process() is called with doa_tracks; physical
        mounting constants, calibrate on the real hardware, not guessable
        from here. mic_sign defaults to +1 (array and camera bearings
        increase the same rotational direction); flip to -1 if attribution
        works dead-centre but fails as the speaker moves off-axis in EITHER
        direction -- that pattern means the mount is rotated backwards, not
        just offset, and no gate_deg widening fixes it.
    """

    def __init__(self, intrinsics, model_name="enet_b0_8_best_vgaf",
                 face_min_size=(32, 32), min_contain=0.6, mic_offset_deg=0.0,
                 mic_sign=1.0):
        self.face = FacePerception(model_name=model_name)
        self.tracker = PersonTracker()
        self.selector = TargetSelector()
        self.intrinsics = intrinsics
        self.mic_offset_deg = mic_offset_deg
        self.mic_sign = mic_sign
        self.pose_state = PerTrackState(lambda: PoseFeatures(intrinsics=intrinsics))
        # one EMA per tracked person: an average is only meaningful over ONE
        # person's readings. Sharing a smoother across people blends strangers.
        self.fused_ema = PerTrackState(EmaSmoother)
        # safety ratchet on top of the EMA: expand fast, shrink slow.
        self.radius_smooth = PerTrackState(proxemics.RadiusSmoother)
        self.face_min_size = face_min_size
        self.min_contain = min_contain

    def process(self, frame_bgr, people, audio=None, doa_tracks=None) -> VideoResult:
        """frame_bgr: the frame BOTH detectors see. Must be the same array the
            person boxes were computed from, or the coordinates do not line up.
        people: [(person_box, pts), ...] for this frame, extracted from the pose
            NN by the runner. pts is the COCO-17 keypoint dict pose_features
            expects.
        audio: optional (probs6, available, quality) for the audio channel --
            one reading for the scene, since the SER classifies a single
            beamformed channel.
        doa_tracks: optional AudioBridge.tracks() for this frame. When given,
            `audio` is attributed to whichever tracked person's on-screen
            bearing matches the currently-active DOA track (bearing.py),
            which may not be the attended person -- that is the point: a
            person off to the side who is actually speaking gets the audio,
            not whoever the camera likes best. When None (the default),
            falls back to the old behaviour of joining the attended person
            only, for callers that have not wired DOA through yet.

        Returns a VideoResult. Only CONFIRMED, currently-visible tracks appear.
        """
        boxes = [b for b, _pts in people]
        pts_by_det = [pts for _b, pts in people]

        tracks = self.tracker.update(boxes)
        target = self.selector.update(tracks)

        # person_bearings and audio_cam_deg ride along on VideoResult purely
        # for field calibration -- see run.py's debug line. Neither
        # feeds back into fusion; audio_target (below) is the only decision
        # that matters there.
        person_bearings = {}
        audio_cam_deg = None
        if doa_tracks is None:
            audio_target = target
        else:
            if self.intrinsics:
                person_bearings = {tid: bearing.camera_bearing_deg(tbox, self.intrinsics)
                                   for tid, tbox in tracks}
            active = bearing.active_doa_track(doa_tracks)
            if active is not None:
                audio_cam_deg = bearing.doa_to_camera_deg(
                    active.angle_deg, self.mic_offset_deg, self.mic_sign)
            audio_target = bearing.match_person(doa_tracks, self.mic_offset_deg,
                                                person_bearings, mic_sign=self.mic_sign)

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

            # ---- metric position: person anchor in camera frame ----
            # torso mean, metres. The nav layer turns this into a proxemics
            # circle after the static camera->base transform. None when no
            # torso joint has valid depth this frame (planner should hold the
            # last good position or skip the circle rather than trust a guess).
            r.centroid_xyz = torso_centroid(pts_by_det[di])

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
            if audio is not None and tid == audio_target:
                a_probs, a_avail, a_quality = audio
                if a_avail and a_probs is not None:
                    peers.append((a_probs, a_quality))

            r.weights = late_fusion.weights_of(peers)
            r.fused_raw = late_fusion.fuse(peers, r.body_prior)

            # ---- temporal smoothing: the root fix for frame-to-frame popping --
            # Affect is recomputed from scratch every frame, so a blink or head
            # turn can flip the argmax and, downstream, make the proxemics
            # radius jump ~1 m several times a second. Smoothing HERE (not at
            # the radius) fixes the label and every consumer at once.
            # Weighted by this frame's best peer quality, so a confident read
            # pulls the average quickly and a poor one barely nudges it; with
            # no peer at all (body-only or nothing) the reading is untrusted
            # and merely drifts the average instead of resetting it.
            conf = max((q for _p, q in peers), default=0.0)
            r.fused = self.fused_ema.get(tid).update(r.fused_raw, conf)
            r.label = late_fusion.label_of(r.fused)

            # ---- navigation: affect -> personal-space circle ----------------
            # Camera-frame anchor -> robot base frame -> radius from the
            # SMOOTHED affect. None when this person has no valid depth this
            # frame; the planner holds their last circle rather than guessing.
            r.circle = proxemics.circle_for(
                track_id=tid,
                ground_xy=robot_frame.ground_xy(r.centroid_xyz),
                fused=r.fused,
                label=r.label,
                smoother=self.radius_smooth.get(tid),
            )
            results.append(r)

        # without this, every person who ever passed the robot keeps a
        # PoseFeatures with a 15-frame deque alive for the whole run
        self.pose_state.evict(self.tracker.dead_ids)
        self.fused_ema.evict(self.tracker.dead_ids)
        self.radius_smooth.evict(self.tracker.dead_ids)

        return VideoResult(people=results, target_id=target, audio_target_id=audio_target,
                          audio_cam_deg=audio_cam_deg, person_bearings=person_bearings)