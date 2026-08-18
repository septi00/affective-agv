"""
oak_device.py
Open the OAK-D with retries and an explicit USB speed cap.

WHY THIS EXISTS: on the CM5 the OAK-D Lite enumerates intermittently when the
pipeline includes stereo depth plus NN inference. Symptom is all XLink streams
dying at once at startup -- X_LINK_ERROR on every queue, then "Closed
connection / attempting to reconnect". Face-only pipelines (one camera, no
stereo, no VPU inference) almost never hit it; the stereo path does, because it
roughly doubles the device's power draw and bandwidth.

Two mitigations, both here:

  USB SPEED CAP. Forcing USB2 (UsbSpeed.HIGH) is counter-intuitive but is the
  standard workaround for a marginal link: a stable USB2 connection beats a
  USB3 one that renegotiates and drops. The bandwidth works out -- at 30fps and
  512x288, depth (16-bit) is ~9 MB/s and the BGR passthrough ~13 MB/s, roughly
  22 MB/s against USB2's ~35 MB/s practical ceiling.

  RETRY. Enumeration is intermittent, not permanently broken, so a second
  attempt usually succeeds. Retrying costs seconds; a demo that fails to start
  in front of the supervisor costs more.

If retries still fail consistently, the remaining suspect is power rather than
the link -- the CM5's USB ports are stingy and stereo plus NN roughly doubles
the draw. A powered hub or a 5V-injecting Y-cable is the fix there.

No pipeline, no perception -- just device acquisition.
"""
from __future__ import annotations

import time

import depthai as dai

# UsbSpeed.HIGH  = USB2, capped and stable (default here -- see docstring)
# UsbSpeed.SUPER = USB3, faster but the source of the intermittent failures
DEFAULT_MAX_SPEED = dai.UsbSpeed.HIGH


def open_device(attempts: int = 5, delay: float = 2.0,
                max_speed=DEFAULT_MAX_SPEED, verbose: bool = True):
    """Returns a dai.Device, or raises after `attempts` failures.

    The returned device is NOT yet in a context manager -- use it as
    `with open_device() as device:` exactly like `dai.Device()`.

    max_speed=None lets depthai negotiate freely (original behaviour), which is
    worth trying once the cable and power situation is known good.
    """
    last = None
    for i in range(1, attempts + 1):
        try:
            device = dai.Device() if max_speed is None else dai.Device(maxUsbSpeed=max_speed)
            if verbose:
                try:
                    print(f"OAK-D open on attempt {i} | USB speed: {device.getUsbSpeed()}")
                except Exception:
                    print(f"OAK-D open on attempt {i}")
            return device
        except RuntimeError as e:
            last = e
            if verbose:
                print(f"device open failed ({i}/{attempts}): {e}")
            if i < attempts:
                time.sleep(delay)
    raise RuntimeError(f"OAK-D not available after {attempts} attempts: {last}")


def probe(n: int = 5, delay: float = 1.0):
    """Diagnostic: open and close the device n times, reporting USB speed and
    chip temperature each time.

    Run this FIRST. If the reported speed varies between runs -- SUPER
    sometimes, HIGH others, or an outright failure -- the link is marginal and
    the problem is the cable or the port, not the pipeline. A consistent speed
    with consistent failures points at power instead.
    """
    for i in range(1, n + 1):
        try:
            with dai.Device() as d:
                temp = d.getChipTemperature().average
                print(f"{i}: speed={d.getUsbSpeed()}  chip={temp:.1f}C")
        except Exception as e:
            print(f"{i}: FAILED  {e}")
        time.sleep(delay)


if __name__ == "__main__":
    print("Probing OAK-D enumeration (unconstrained USB speed)…\n")
    probe()