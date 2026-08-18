"""
face_pose_match.py
Decide which detected face belongs to which tracked person.

THE BUG THIS PREVENTS: with two people in frame, the face channel and the pose
channel each pick a person independently -- `detect_largest_face` takes the
biggest face, the pose loop takes whichever detection it happens to use. There
is nothing forcing those to be the same human. Fuse them and you get person A's
expression multiplied by person B's posture prior. No exception, no warning,
just a wrong answer that looks like model error.

Association rule: a face belongs to a person if the face box sits inside that
person's box. Cheap, and reliable here because YOLOv8-pose boxes are full-body
and faces are small relative to them, so containment is unambiguous except when
people physically overlap.

COORDINATE SPACE: pose boxes come from det.xmin*NN_W etc. and faces come from
Haar on the passthrough frame, which is the same NN_W x NN_H image. Same space,
no scaling needed. If the face detector is ever moved to a full-resolution
stream, scale one of them before calling this.
"""
from __future__ import annotations


def _area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def containment(inner, outer) -> float:
    """Fraction of `inner` that lies inside `outer`, in [0, 1].

    Not IoU: a face is a tiny fraction of a full-body box, so IoU would be near
    zero even for a perfect match. What matters is whether the face is INSIDE
    the body, which is asymmetric."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inner_area = _area(inner)
    return float(iw * ih / inner_area) if inner_area > 0 else 0.0


def match_face(person_box, faces, min_contain: float = 0.6):
    """person_box: (x1, y1, x2, y2) for one tracked person.
    faces: [((x1,y1,x2,y2), det_weight), ...] from face_detector.detect_faces_scored.

    Returns (box, det_weight) for this person's face, or (None, None) if no
    face is confidently inside their box.

    (None, None) is a NORMAL outcome, not an error -- the person is turned away,
    or Haar missed a profile. The caller passes it through as
    FaceResult(probs=None), the face channel reports available=False, and
    fusion proceeds on audio and body alone. That path is the modality-dropout
    design doing its job.
    """
    best = None
    best_area = -1.0
    for box, weight in faces:
        if containment(box, person_box) < min_contain:
            continue
        a = _area(box)
        if a > best_area:
            best_area = a
            best = (box, weight)
    return best if best is not None else (None, None)


def match_faces_to_people(people, faces, min_contain: float = 0.6):
    """Greedy one-to-one assignment across everyone at once.

    people: [(track_id, person_box), ...]
    Returns {track_id: (face_box, det_weight)} containing only matched people.

    One face cannot serve two people, which matters when boxes overlap: without
    the exclusivity, a face in the overlap region would be handed to both and
    two tracks would report the same expression.

    Only needed if you classify every person's face. If you only ever run
    HSEmotion on the attention target -- the cheaper and more likely path on
    the CM5 -- call match_face() for that one person and skip this.
    """
    scored = []
    for tid, pbox in people:
        for fi, (fbox, weight) in enumerate(faces):
            c = containment(fbox, pbox)
            if c >= min_contain:
                scored.append((c, _area(fbox), tid, fi, fbox, weight))
    scored.sort(key=lambda s: (-s[0], -s[1]))

    out: dict[int, tuple] = {}
    used_faces: set[int] = set()
    for _c, _a, tid, fi, fbox, weight in scored:
        if tid in out or fi in used_faces:
            continue
        out[tid] = (fbox, weight)
        used_faces.add(fi)
    return out


if __name__ == "__main__":
    person_a = (50, 20, 200, 400)
    person_b = (250, 30, 400, 410)
    faces = [((90, 40, 150, 110), 4.2),    # inside A
             ((290, 50, 350, 120), 3.8),   # inside B
             ((600, 50, 660, 120), 2.1)]   # nobody

    print("A ->", match_face(person_a, faces))
    print("B ->", match_face(person_b, faces))
    print("all ->", match_faces_to_people([(7, person_a), (9, person_b)], faces))
