# Circle relay wire format (UDP, CM5 -> Pi4)

One datagram per perception frame. Little-endian:

  [uint32  n_circles]
  then n_circles times:
    [int32   person_id]
    [float32 x]        # base_footprint, metres
    [float32 y]
    [float32 radius]
    [float32 peak_cost]

Empty frame (n_circles = 0) is valid and sent, so the receiver knows
"nobody here now" and can clear stale circles.

Header frame is always base_footprint (perception already transforms to it).
