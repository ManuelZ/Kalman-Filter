import numpy as np

class Kalman:
    """ 
    Kalman filters assume linear system dinamics.

    Trick: run the Kalman filter for the x, y and z values independently, so you have smaller matrices.
    """
    def __init__(self, A, B, C, Q, R, x_prev=None, P_prev=None):
        # State at time t-1. Column vector.
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
        # THIS IS GOING TO BE UPDATED! HOW? WHO KNOWS!
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
        """
        #print("\nPrediction step...")

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
        #print(f"Kalman Gain:\n{K:.2f}")
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
        #print("\nUpdate step...")

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
            # TODO: read https://www.kalmanfilter.net/simpCovUpdate.html
            #P = (I - K @ self.C) @ self.P_predicted
            P = (I - K @ self.C) @ self.P_predicted @ (I - K @ self.C).T + K @ self.R @ K.T

            self.P_prev = P.copy()
            self.x_prev = x.copy()
        return (x, P)




