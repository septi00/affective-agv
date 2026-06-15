import numpy as np


def wrap(angle):
    """Wrap an angle (radians) to (-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


class WrappedKalmanFilter1D:
    """
    1D azimuth tracker with wrap-aware innovation.
    State: [theta, theta_dot]   (radians, rad/s)
    Measurement: theta only.
    """

    def __init__(self,
                 dt,                       # seconds between frames
                 process_noise_std=2.0,    # rad/s^2, std of angular accel
                 R_min_deg=2.0,            # best-case measurement std
                 R_max_deg=30.0,           # worst-case measurement std
                 conf_skip=0.1,            # below this -> predict only
                 conf_bootstrap=0.3,       # need this to initialize
                 gate_sigmas=3.0):         # outlier reject threshold
        self.dt = dt

        # State and covariance
        self.x = np.zeros(2)
        self.P = np.diag([np.pi**2, 1.0])

        # Constant-velocity model
        self.F = np.array([[1.0, dt],
                           [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])

        # Process noise from white angular acceleration
        q = process_noise_std ** 2
        self.Q = q * np.array([[dt**4 / 4, dt**3 / 2],
                               [dt**3 / 2, dt**2]])

        # Confidence -> measurement variance mapping (rad^2)
        self.R_min = np.deg2rad(R_min_deg) ** 2
        self.R_max = np.deg2rad(R_max_deg) ** 2

        self.conf_skip = conf_skip
        self.conf_bootstrap = conf_bootstrap
        self.gate = gate_sigmas
        self.initialized = False

    def predict(self):
        self.x = self.F @ self.x
        self.x[0] = wrap(self.x[0])
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, confidence):
        """
        z: measured azimuth in radians
        confidence: scalar in [0, 1] from your DOA module
        Returns True if measurement was used, False otherwise.
        """
        # Wait for a confident frame before trusting any state
        if not self.initialized:
            if confidence >= self.conf_bootstrap:
                self.x[0] = wrap(z)
                self.x[1] = 0.0
                self.P = np.diag([np.deg2rad(10.0)**2, 1.0])
                self.initialized = True
                return True
            return False

        # Silence / garbage frame: coast on prediction
        if confidence < self.conf_skip:
            return False

        # Map confidence to variance: low conf -> large R -> small K
        R = self.R_min + (self.R_max - self.R_min) * (1.0 - confidence)

        # Wrap the innovation
        y = wrap(z - (self.H @ self.x)[0])
        S = float((self.H @ self.P @ self.H.T)[0, 0]) + R

        # Mahalanobis gate: reject outliers
        if y * y / S > self.gate ** 2:
            return False

        K = (self.P @ self.H.T).flatten() / S
        self.x = self.x + K * y
        self.x[0] = wrap(self.x[0])
        self.P = (np.eye(2) - np.outer(K, self.H)) @ self.P
        return True

    @property
    def angle(self):
        return self.x[0]

    @property
    def angle_std(self):
        return np.sqrt(self.P[0, 0])