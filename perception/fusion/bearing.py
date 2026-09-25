"""
bearing.py
Attributes a DOA audio track to a specific tracked PERSON by camera-relative
angle, instead of gluing the scene's audio reading onto whichever person the
robot currently attends to.

THE GAP THIS CLOSES: AudioBridge.tracks() returns confirmed DOA tracks with
bearings in the MIC ARRAY's frame (0 degrees is the ReSpeaker's silkscreen
mark). A person's box lives in the CAMERA's frame (pixels). The two are only
comparable once the mounting offset between the array and the camera is
known -- this file is that comparison, plus the offset as a named constant
the caller supplies.

CALIBRATION NEEDS TWO NUMBERS, NOT ONE. mic_offset_deg is the DOA angle (mic
frame) that points straight down the camera's optical axis -- measure it by
having someone speak dead-centre of frame and reading the reported track
angle. But that alone assumes the mic array's azimuth increases in the SAME
rotational sense as the camera's bearing (positive = to the camera's right).
If the array is mounted rotated 180 degrees about its normal relative to the
camera (a very easy thing to get backwards -- there is no visual cue for it
the way there is for a forward/back flip), azimuth increases the OPPOSITE
way, and a single additive offset cannot fix that: the mismatch will look
fine near centre (where displacement is small either way) and grow steadily
as the speaker moves off-axis, in either direction, because the two angles
are diverging rather than tracking each other. That is mic_sign: +1 if
moving right on camera makes the DOA angle increase the same way past the
offset, -1 if it goes the other way. Calibrate both from two stances (centre,
then one side), reading the "cam"/"brg" values run.py's Reporter prints
every frame (cam = doa_to_camera_deg() below, brg = each tracked person's own
camera_bearing_deg) and solving for the sign and constant that make cam
track the speaking person's own brg.

WHY "ACTIVE" TRACK, NOT ALL TRACKS: the SER classifies ONE beamformed channel
per window, so there is exactly one emotion reading to place -- the question
is which DOA track produced the sound it just read. A track with misses==0
was matched to a fresh detection THIS block, i.e. it is where the current
audio actually came from; a track that's merely coasting (age growing,
misses>0) is a past speaker still being remembered, not the source of the
current SER result. See active_doa_track() for the (rare) multi-peak case.
"""
from __future__ import annotations


def wrap_deg(a: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


def camera_bearing_deg(box, intrinsics) -> float:
    """Horizontal angle of a person's box centre off the camera's optical
    axis, in degrees. intrinsics: (fx, fy, cx, cy), the same tuple
    PoseFeatures already takes (from calib.getCameraIntrinsics()).

    Positive = right of centre, matching image-x increasing rightward --
    this is a bearing convention, not a rotation, so there is no need to
    match SRP-PHAT's sign convention here; doa_to_camera_deg() below is
    where the two frames actually get reconciled, via mic_offset_deg."""
    import math
    x1, _y1, x2, _y2 = box
    fx, _fy, cx, _cy = intrinsics
    cx_box = (x1 + x2) / 2.0
    return math.degrees(math.atan2(cx_box - cx, fx))


def active_doa_track(tracks):
    """Which confirmed DOA track produced the sound just classified, or None
    if nothing is currently live.

    tracks: TrackView list from AudioBridge.tracks() (.id/.angle_deg/
    .std_deg/.age/.misses).

    pick_peaks() can report up to 4 simultaneous spatial peaks, so more than
    one track can have misses==0 in the same block -- a speaker plus a loud
    reflection or a second noise source, for instance. There is no way to
    tell from a single beamformed-channel SER result which peak was the
    speech, so this is a documented ambiguity, not a solved one: the
    tie-break is the tightest angle estimate (lowest std_deg), on the theory
    that a stable, well-formed track is more likely a real sustained speaker
    than a transient reflection."""
    fresh = [t for t in tracks if t.misses == 0]
    if not fresh:
        return None
    return min(fresh, key=lambda t: t.std_deg)


def doa_to_camera_deg(doa_angle_deg, mic_offset_deg, mic_sign=1.0):
    """Convert one DOA angle (mic frame) into the camera's bearing frame.
    See the module docstring for what mic_sign is and why a plain additive
    offset isn't always enough. Exposed standalone (not just inlined in
    match_person) so a caller can print it next to a person's own
    camera_bearing_deg for field calibration -- that comparison is the whole
    point, see run.py's debug line."""
    return wrap_deg(mic_sign * (doa_angle_deg - mic_offset_deg))


def match_person(tracks, mic_offset_deg, person_bearings, gate_deg=20.0, mic_sign=1.0):
    """Which tracked person the currently-active DOA track points at.

    tracks: TrackView list from AudioBridge.tracks().
    mic_offset_deg, mic_sign: mounting calibration, see module docstring.
    person_bearings: {track_id: camera_bearing_deg} for this frame's tracked
        people (from camera_bearing_deg() above, keyed by PersonTracker id).
    gate_deg: how close the DOA bearing and a person's bearing must be to
        count as a match. Loose default -- SRP-PHAT's practical angular
        resolution is coarser than the 2-degree AZ_GRID suggests, and two
        candidate people rarely stand within 20 degrees of each other AND
        both plausibly be the source of the current speech. Note this only
        papers over noise, not a wrong mic_sign -- a sign error grows with
        displacement from centre and no reasonable gate absorbs that.

    Returns the matched track_id, or None -- meaning either nobody is
    currently speaking, or the speaker isn't any of this frame's tracked
    people (off-camera, or behind someone). None is correct in both cases:
    attaching audio to the nearest visible person regardless of direction is
    exactly the bug this module exists to fix."""
    active = active_doa_track(tracks)
    if active is None or not person_bearings:
        return None
    cam_deg = doa_to_camera_deg(active.angle_deg, mic_offset_deg, mic_sign)
    best_tid, best_d = None, gate_deg
    for tid, bearing_deg in person_bearings.items():
        d = abs(wrap_deg(cam_deg - bearing_deg))
        if d < best_d:
            best_tid, best_d = tid, d
    return best_tid


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class _T:
        id: int
        angle_deg: float
        std_deg: float
        age: int
        misses: int

    intr = (500.0, 500.0, 320.0, 240.0)   # fx, fy, cx, cy for a 640-wide frame
    box_left  = (20, 100, 180, 400)       # left side of frame  -> negative bearing
    box_right = (460, 100, 620, 400)      # right side of frame -> positive bearing

    print("camera_bearing left ", round(camera_bearing_deg(box_left, intr), 1))
    print("camera_bearing right", round(camera_bearing_deg(box_right, intr), 1))

    offset = 90.0   # mic-frame angle that happens to point down the camera axis

    # DOA track pointing at the RIGHT person, camera is attending the LEFT one
    tracks = [_T(id=7, angle_deg=offset + 15.0, std_deg=4.0, age=10, misses=0)]
    bearings = {1: camera_bearing_deg(box_left, intr),
                2: camera_bearing_deg(box_right, intr)}
    print("attended=1, speaker bearing matches ->", match_person(tracks, offset, bearings))
    assert match_person(tracks, offset, bearings) == 2

    # speaker is off-camera entirely -- must match nobody, not fall back to target
    tracks_offcam = [_T(id=7, angle_deg=offset + 90.0, std_deg=4.0, age=10, misses=0)]
    print("off-camera speaker ->", match_person(tracks_offcam, offset, bearings))
    assert match_person(tracks_offcam, offset, bearings) is None

    # nobody currently speaking (only a coasting track) -> no attribution
    tracks_stale = [_T(id=7, angle_deg=offset + 15.0, std_deg=4.0, age=40, misses=6)]
    print("stale-only track ->", match_person(tracks_stale, offset, bearings))
    assert match_person(tracks_stale, offset, bearings) is None

    # two simultaneous fresh tracks -> tie-break on tightest std_deg
    tracks_multi = [
        _T(id=7, angle_deg=offset + 15.0, std_deg=6.0, age=10, misses=0),   # noisy
        _T(id=8, angle_deg=offset - 15.0, std_deg=1.5, age=10, misses=0),   # confident
    ]
    print("multi-peak tie-break ->", match_person(tracks_multi, offset, bearings))
    assert match_person(tracks_multi, offset, bearings) == 1

    # mic_sign: array's azimuth increases OPPOSITE the camera's bearing sense.
    # Single tracked person, standing 25 degrees right of camera centre. The
    # mount is backwards, so the true reading is offset MINUS the true
    # bearing, not plus it -- reproducing the field symptom: near centre a
    # constant-offset-only model still looks fine (small either way), but at
    # 25 degrees off-axis the wrong-sign conversion misses the gate entirely.
    off2 = 40.0
    true_bearing = 25.0
    bearings2 = {1: true_bearing}
    doa_true = off2 - true_bearing
    tracks_flipped = [_T(id=9, angle_deg=doa_true, std_deg=2.0, age=10, misses=0)]
    print("wrong sign ->", match_person(tracks_flipped, off2, bearings2, mic_sign=1.0))
    assert match_person(tracks_flipped, off2, bearings2, mic_sign=1.0) is None
    print("correct sign ->", match_person(tracks_flipped, off2, bearings2, mic_sign=-1.0))
    assert match_person(tracks_flipped, off2, bearings2, mic_sign=-1.0) == 1

    print("all checks passed")