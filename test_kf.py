# External imports
import numpy as np
import filterpy.kalman
import filterpy.stats
from pytest import approx
from scipy.linalg import block_diag, norm
from filterpy.common import Q_discrete_white_noise

# Local imports
from kalman import Kalman

"""
Test my Kalman Filter implementation by comparing its output to the filterpy implementation.
"""


def test_noisy_1d():

    # Initial state (location and velocity)
    x = np.array([[2.0], [0.0]])
    
    #Current state covariance matrix
    P = np.eye(2) * 1000

    # State transition matrix
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    
    # Measurement function
    H = np.array([[1.0, 0.0]])
    
    # state uncertainty
    R = np.identity(1) * 5
    
    # process uncertainty
    Q = 0.0001
    
    my_kf = Kalman(F=F, H=H, B=None, R=R, Q=Q, x_prev=x, P_prev=P)

    filterpy_kf = filterpy.kalman.KalmanFilter(dim_x=2, dim_z=1)
    filterpy_kf.x = x
    filterpy_kf.F = F
    filterpy_kf.H = H
    filterpy_kf.P = P
    filterpy_kf.R = R
    filterpy_kf.Q = Q

    for t in range(100):
        
        # Create measurement. measurement = t plus white noise
        z = t + np.random.randn() * 20

        # Perform kalman filtering
        filterpy_kf.predict()
        filterpy_kf.update(z)

        my_kf.predict()
        my_kf.update(z)

        assert norm(filterpy_kf.x - my_kf.x_posterior) < 1.e-12
        assert norm(filterpy_kf.P - my_kf.P_posterior) < 1.e-12


def test_noisy_3d():
    """
    """
    
    N = 100
    dt = 0.02
    R_std = 1.0

    x = np.zeros([6, 1])
   
    p = np.diag([1., 1.])
    P = block_diag(p, p, p)
   
    F1 = np.array(
        [[1., dt],
         [0., 1.]]
    )
    F = block_diag(F1, F1, F1)
   
    h = np.array([[1., 0.]])
    H = block_diag(h, h, h)
   
    R = np.eye(3) * (R_std**2)
   
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.0001)
    Q = block_diag(q, q, q)

    filterpy_kf = filterpy.kalman.KalmanFilter(dim_x=6, dim_z=3)
    filterpy_kf.x = x
    filterpy_kf.P = P
    filterpy_kf.F = F
    filterpy_kf.H = H
    filterpy_kf.R = R
    filterpy_kf.Q = Q

    my_kf = Kalman(F=F, H=H, B=None, R=R, Q=Q, x_prev=x, P_prev=P)
   
    # velocity vector
    v = np.array([3.0, 4.0, 5.0])

    z = np.zeros([3, 1])
    for t in range(N):
        # create measurement = t plus white noise
        z[0, 0] = v[0] * t + np.random.randn() * 20
        z[1, 0] = v[1] * t + np.random.randn() * 20
        z[2, 0] = v[2] * t + np.random.randn() * 20

        filterpy_kf.predict()
        filterpy_kf.update(z)

        my_kf.predict()
        my_kf.update(z)

        assert norm(filterpy_kf.x - my_kf.x_posterior) < 1.e-12
        assert norm(filterpy_kf.P - my_kf.P_posterior) < 1.e-12


if __name__ == "__main__":
    test_noisy_1d()
    test_noisy_3d()