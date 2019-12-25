import numpy as np
import win32gui 
import cv2
import matplotlib.pyplot as plt

class Kalman:
    """ 
    Kalman filters assume linear system dinamics.

    Trick: run the Kalman filter for the x, y and z values independently, so you have smaller matrices.
    """
    def __init__(self, A, B, C, Q, R, x_prev=None, P_prev=None):
        # Belief at previous time step
        self.x_prev = x_prev
        # Error covariance of the state estimate at previous time step
        self.P_prev = P_prev

        # State transition model or motion model (equations of motion) that predict the new state
        # matrix of size n x n where n is the dimension of x (the state vector)
        self.A = A

        # Control input model. Model that predicts what changes based on the commands to the vehicle E.g. differential model, ackermann, etc 
        # Matrix of size n x m, where m is the dimension of u
        # If you don't have B and u, you can simply not use them:
        # https://www.youtube.com/watch?v=18TKA-YWhX0?t=31m42s
        self.B = B

        # Measurement matrix
        # How to map from the sensor reading to the state vector.
        # Matrix of size k x n, where k is the size of the measurement vector z.
        self. C = C

        # Process noise
        # Covariance of the multivariate gaussian noise that models the randomness in the state transition
        # As a starting point, put the standard deviation of the noise of the sensor given from the manufacturer.
        # THIS IS GOING TO BE UPDATE! HOW? WHO KNOWS!
        # Matrix size is n x n
        self.Q = Q

        # Measurement noise covariance
        # Error from the sensor readings
        # Covariance matrix of the multivariate gaussian noise of the sensor measurement
        # As a starting point, put the standard deviation of the noise of the sensor given from the manufacturer.
        # Matrix size is k x k where k is the size of the measurement vector
        self.R = R

        self.setMode()

    def setMode(self):
        parameters = [self.A, self.C, self.Q, self.R, self.x_prev, self.P_prev]
        
        if all(map(lambda x: np.isscalar(x), parameters)):
            self.mode = '1D'
        else:
            self.mode = '2D'
        print(f"Mode: {self.mode}")
    
    def predict(self, u=None):
        """ 
        *Predict* the belief (the state and the error covariance matrix) about the state at time t.
        (Also called control update step)
        This step increases the uncertainty of the robot belief.
        
        INPUT:
        u: Control at time t. Column vector.

        OUTPUT:
        x_predicted: State at time t. Column vector.
        P_predicted: Error covariance of the state estimate. Quadratic matrix.

        x_prev: State at time t-1. Column vector.
        """
        print("\nPrediction step...")

        if self.mode == '1D':
            x_predicted = self.A * self.x_prev 
            if self.B:
                x_predicted += self.B * u
            P_predicted = self.A * self.P_prev + self.Q
        else:
            # Estimate the state at time t
            x_predicted = self.A @ self.x_prev 
            
            if self.B:
                x_predicted += self.B @ u

            # Now predict how much noise will be in the measurements
            # Error covariance matrix: Variance of the a priori estimate
            P_predicted = self.A @ self.P_prev @ self.A.transpose() + self.Q

        self.x_predicted = x_predicted
        self.P_predicted = P_predicted
    
    def __calculate_Kalman_gain(self):
        """
        INPUT
        P: Error covariance of the state estimate. Quadratic matrix.

        OUTPUT: 
        K: Kalman gain. It specifies the degree to which the measurement is incorporated into the new state estimate.
        Or how much to trust this sensor
        
        """
        if self.mode == '1D':
            S = self.C * self.P_predicted + self.R
            K = self.P_predicted / S
        else:
            # Innovation covariance
            S = self.C @ self.P_predicted @ self.C.transpose() + self.R
            # Kalman gain
            K = self.P_predicted @ self.C.transpose() @ np.linalg.inv(S)
        return K

    def update(self, z):
        """ 
        This step reduces the uncertainty of the robot belief.

        INPUT:
        x_predicted: obtained during the prediction step
        K: Kalman gain
        z: measurement vector of dimension k at time t (sensor data)

        OUTPUT:
        x: posterior belief
        P: covariance of the posterior belief
        """
        print("\nUpdate step...")

        K = self.__calculate_Kalman_gain()
        
        if self.mode == '1D':
            y = z - self.C * self.x_predicted
            x = self.x_predicted + K * y
            P = (1 - K * self.C) * self.P_predicted
            self.P_prev = P
            self.x_prev = x

        else:
            # Innovation
            y = z - self.C @ self.x_predicted
            # Update the estimate with measurements from sensor
            x = self.x_predicted + K @ y
            
            I = np.identity(K.shape[0])
            P = (I - K @ self.C) @ self.P_predicted

            self.P_prev = P.copy()
            self.x_prev = x.copy()
        return (x, P)

def enum_handler(hwnd, results):
    if win32gui.GetWindowText(hwnd) == "image":
        win_rect = win32gui.GetWindowRect(hwnd)
        clientRect = win32gui.GetClientRect(hwnd)
        
        win_h_with_title = (win_rect[-1] - win_rect[1])
        win_h_with_no_title = (clientRect[-1] - clientRect[1]) - 1
        title_bar_size = win_h_with_title - win_h_with_no_title

        x_top_left_corner = win_rect[0]
        y_top_left_corner = win_rect[1] + title_bar_size
        
        results.append((x_top_left_corner, y_top_left_corner))

def example_2D_constant_velocity_mouse():
    # 2D Constant velocity model
    #
    # state = [x; vx; y; vy]
    # measurement vector = [x; y]

    A = np.array([[1, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 1],
                  [0, 0, 0, 1]])

    n = A.shape[0]
    k = 2
    
    # No Control model
    B = None

    # Measurement matrix.
    # our sensors only know the position x and y
    C = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])
    
    process_noise = 1e-4
    Q = np.identity(n) * process_noise

    measurement_noise = 1 #std dev
    R = np.identity(2) * (measurement_noise ** 2)
    
    (x_win, y_win) = win32gui.GetCursorPos()
    initial_state = np.array([[x_win], [0], [y_win], [0]])
    initial_P = np.identity(n) * 1e-6 # perfectly known initial position
    kalman = Kalman(A, B, C, Q, R, initial_state, initial_P)

    img = np.zeros((512,512,3), np.uint8)

    mouse_prev = (0, 0)
    x_kalman_prev = (0,0,0,0)
    while True:
        kalman.predict()

        enumerated_windows = [[0,0]]
        win32gui.EnumWindows(enum_handler, enumerated_windows)
        (x_window, y_window) = enumerated_windows[-1][:2]

        # Sense
        mouse = win32gui.GetCursorPos()
        z = np.array([[mouse[0]], [mouse[1]]])
        
        x, P = kalman.update(z)

        # Plot mouse
        cv2.line(img, (mouse_prev[0] - x_window, mouse_prev[1] - y_window),
                      (mouse[0] - x_window, mouse[1] - y_window),
                      (0,255,0), 1)
        # Plot Kalman
        cv2.line(img, (x_kalman_prev[0] - x_window, x_kalman_prev[2] - y_window),
                      (x[0] - x_window, x[2] - y_window),
                      (0,0,255), 1)
        
        mouse_prev = mouse
        x_kalman_prev = x.copy()

        cv2.imshow('image', img)
        
        k = cv2.waitKey(1) & 0xFF
        if k==27:
            break

def example_2D_constant_velocity():
    A = 1.0
    B = None
    C = 1.0
    Q = 0.01
    R = 0.01

    initial_state_x = 5.3
    initial_state_y = 3.6
    initial_P = 0.01
    
    kalman_x = Kalman(A, B, C, Q, R, initial_state_x, initial_P)
    kalman_y = Kalman(A, B, C, Q, R, initial_state_y, initial_P)

    vx = 0.2
    vy = 0.1
    T = 0.5
    x = np.arange(0, 2, vx*T)
    y = np.arange(5, 6, vy*T)
    
    states = np.zeros((x.shape[0], 2))

    for k,_ in enumerate(x):
        kalman_x.predict()
        kalman_y.predict()
        new_state_x, P = kalman_x.update(x[k])
        new_state_y, P = kalman_y.update(y[k])
        states[k, :] = [new_state_x, new_state_y]

    plt.plot(x, y, 'b.-', label='Object position')
    plt.plot(states[:, 0], states[:, 1], 'r.-', label='Corrected position')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    example_2D_constant_velocity_mouse()
    #example_2D_constant_velocity()


