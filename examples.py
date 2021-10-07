import cv2
import matplotlib.pyplot as plt
import win32gui 
import numpy as np
from kalman import Kalman

# Only works in Windows
# Usage: python examples.py
#

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

    A = np.array([[1, 1, 0, 0], # NOTE: this would mean 1 * x + 1 * vx, so dt = 1!!!!   
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
        
        print((x_kalman_prev[0] - x_window, x_kalman_prev[2] - y_window))
        print((x[0], x[2]))
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
    # example_2D_constant_velocity()