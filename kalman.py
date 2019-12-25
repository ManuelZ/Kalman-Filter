import numpy as np
import win32gui 
import cv2

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

        # Estimate the state at time t
        # If you don't have B and u, you can simply not use them:
        # https://www.youtube.com/watch?v=18TKA-YWhX0?t=31m42s
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
        #results.append(win32gui.GetWindowRect(hwnd))
        (flags, showCmd, ptMin, ptMax, rect) = win32gui.GetWindowPlacement(hwnd)
        results.append(rect)
        return
    # results.append({
    #     #"hwnd":hwnd,
    #     #"hwnd_above":win32gui.GetWindow(hwnd, win32con.GW_HWNDPREV), # Window handle to above window
    #     "title":win32gui.GetWindowText(hwnd),
    #     #"visible":win32gui.IsWindowVisible(hwnd) == 1,
    #     #"minimized":window_placement[1] == win32con.SW_SHOWMINIMIZED,
    #     #"maximized":window_placement[1] == win32con.SW_SHOWMAXIMIZED,
    #     "rectangle":win32gui.GetWindowRect(hwnd) #(left, top, right, bottom)
    #     })


if __name__ == "__main__":
    
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
        
        mouse = win32gui.GetCursorPos()

        enumerated_windows = [[0,0,0,0]]
        win32gui.EnumWindows(enum_handler, enumerated_windows)
        windowCorners = enumerated_windows[-1]
        #print(f"windowCorners: {windowCorners}")
        #print(f"Mouse: {mouse}")
        x_window = windowCorners[0] + 92
        y_window = windowCorners[1]
        
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

        print(f'New x: \n {x}')
