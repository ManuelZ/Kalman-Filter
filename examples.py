import cv2
import matplotlib.pyplot as plt
import win32gui 
import numpy as np
from kalman import Kalman

# Only works in Windows
# Usage: python examples.py
#

WINDOW_NAME = "Mouse"
GREEN = (0,255,0)
RED = (0, 0, 255)


def get_window_coordinates_handler(hwnd, results):
    """
    Get the top left coordinates of the window.
    """
    if win32gui.GetWindowText(hwnd) == WINDOW_NAME:
        # Screen coordinates of the upper-left and lower-right corners of the window.
        win_rect = win32gui.GetWindowRect(hwnd)

        # The client coordinates specify the upper-left and lower-right corners of the client area.
        # This basically gives (0, 0, 512, 512) for a 512x512 imshow call
        client_rect = win32gui.GetClientRect(hwnd)

        extra_h = (win_rect[3] - win_rect[1]) - (client_rect[3] - client_rect[1])
        extra_w = (win_rect[2] - win_rect[0]) - (client_rect[2] - client_rect[0])

        x_top_left_corner = win_rect[0] + extra_w // 2
        # I'm not sure why but the final extra_w // 2 is needed to get the center of the mouse aligned with the arrow
        y_top_left_corner = win_rect[1] + extra_h - (extra_w // 2)

        results[0,0] = x_top_left_corner
        results[0,1] = y_top_left_corner


def example_2D_constant_velocity_mouse():
    """   
    2D Constant velocity model
    
    state = [x; vx; y; vy]
    measurement vector = [x; y]
    """
    
    # State transition matrix
    A = np.array([[1, 1, 0, 0], # NOTE: this row means 1*x + 1*vx, so dt = 1
                  [0, 1, 0, 0],
                  [0, 0, 1, 1],
                  [0, 0, 0, 1]])

    n = A.shape[0]
    
    # No Control model
    B = None

    # Measurement matrix, aka Observation matrix
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

    window_top_left_coordinates = np.zeros((1, 2), dtype=np.int32)
    mouse_position_prev = np.zeros(2, dtype=np.int32)
    x_kalman_prev = np.zeros((4,1), dtype=np.int32)

    while True:
        kalman.predict()

        # Sense
        win32gui.EnumWindows(get_window_coordinates_handler, window_top_left_coordinates)
        mouse_position = np.array(win32gui.GetCursorPos(), dtype=np.int32).reshape(-1)
        z = mouse_position.copy().reshape(2, 1)

        # Calculate the posterior and its covariance
        x, P = kalman.update(z)  # x shape: (4, 1), P shape: (4, 4)

        # Plot mouse_position
        mouse_pt1 = (mouse_position_prev - window_top_left_coordinates).reshape(-1)
        mouse_pt2 = (mouse_position - window_top_left_coordinates).reshape(-1)
        cv2.line(img, pt1=mouse_pt1, pt2=mouse_pt2, color=GREEN, thickness=1)
        
        # Plot Kalman
        position_state_indices = [0, 2]
        predicted_pt1 = (x_kalman_prev[position_state_indices,0] - window_top_left_coordinates).reshape(-1).astype(np.int32)
        predicted_pt2 = (x[position_state_indices,0] - window_top_left_coordinates).reshape(-1).astype(np.int32)
        cv2.line(img, pt1=predicted_pt1, pt2=predicted_pt2, color=RED, thickness=1)
        
        mouse_position_prev = mouse_position.copy()
        x_kalman_prev = x.copy()

        cv2.imshow(WINDOW_NAME, img)
        k = cv2.waitKey(1) & 0xFF
        if k==27:
            break

def example_2D_constant_velocity():
    """
    """
    print("2D constant velocity example.")

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
    example_2D_constant_velocity()