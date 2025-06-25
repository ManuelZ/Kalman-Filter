import numpy as np
from math import sqrt
from numpy import dot

class Kalman:
    """ 
    Kalman filters assume linear system dynamics.

    "You have to design the state (x; P), the process (F; Q), the measurement (z; R),and the measurement function H. 
    If the system has control inputs, such as a robot, you will also design B and u." From [4].

    From [4]:
    Initialization
        1. Initialize the state of the filter
        2. Initialize our belief in the state
    Predict
        1. Use process model to predict state at the next time step
        2. Adjust belief to account for the uncertainty in prediction
    Update
        1. Get a measurement and associated belief about its accuracy
        2. Compute residual between estimated state and measurement
        3. Compute scaling factor based on whether the measurement or prediction is more accurate
        4. Set state between the prediction and measurement based on scaling factor
        5. update belief in the state based on how certain we are in the measurement

    Resources:
    - [1] https://www.kalmanfilter.net/stateUpdate.html
    - [2] Kalman Filter tutorial: https://youtu.be/18TKA-YWhX0
    - [3] filterpy package
    - [4] Kalman and Bayesian filters in Python
    """
    def __init__(self, F, B, H, Q, R, x_prev=None, P_prev=None):
        self.history = []

        # State at time t-1. Column vector.
        # Belief at previous time step
        self.x_posterior = x_prev
        
        # Error covariance of the state estimate at previous time step
        self.P_posterior = P_prev

        # State transition model or motion model (equations of motion) that predict the new state
        # matrix of size n x n where n is the dimension of x (the state vector)
        self.F = F

        # Control matrix. It illustrates the mechanism by which uk influences state xk. 
        # Control input model. Model that predicts what changes based on the commands to the vehicle E.g. differential model, ackermann, etc 
        # Matrix of size n x m, where m is the dimension of u
        # If you don't have B and u, you can simply not use them:
        # https://www.youtube.com/watch?v=18TKA-YWhX0?t=31m42s
        self.B = B

        # Measurement matrix, aka Observation matrix.
        # How to map from the sensor reading to the state vector.
        # Matrix of size k x n, where k is the size of the measurement vector z.
        self.H = H

        # Process noise
        # Covariance of the multivariate gaussian noise that models the randomness in the state transition.
        # As a starting point, put the standard deviation of the noise of the sensor given from the manufacturer.
        # Matrix size is n x n
        self.Q = Q

        # Measurement noise covariance
        # Error from the sensor readings
        # Covariance matrix of the multivariate gaussian noise of the sensor measurement
        # As a starting point, put the standard deviation of the noise of the sensor given from the manufacturer.
        # Matrix size is k x k where k is the size of the measurement vector
        self.R = R

        # system uncertainty
        #self.S

        self.set_mode()

    def set_mode(self):
        parameters = [self.F, self.H, self.Q, self.R, self.x_posterior, self.P_posterior]
        
        if all(map(lambda x: np.isscalar(x), parameters)):
            self.__mode = '1D'
        else:
            self.__mode = '2D'
    
    def predict(self, u=None) -> tuple[np.ndarray, np.ndarray]:
        """ 
        Propagate the current state estimate and uncertainty to form the prior belief for the next time step.

        Propagates the current state estimate and its error covariance forward in time using the system dynamics model. 
        Computes the prior (predicted) state and covariance before incorporating any new measurement.
        This step is sometimes referred to as the "control update" because it can incorporate control inputs into the 
        prediction.

        This step increases the uncertainty of the robot belief.
        
        INPUT:
        u: Control at time t. Column vector.

        OUTPUT:
        x_predicted: State at time t. Column vector.
        P_predicted: Error covariance of the state estimate. Quadratic matrix.
        """

        if self.__mode == '1D':
            self.x_prior = self.F * self.x_posterior 
            if self.B:
                self.x_prior += self.B * u
            self.P_prior = self.F * self.P_posterior + self.Q
        
        else:
            # Compute the prior. Estimate the state at time t based on state at t-1.
            self.x_prior = self.F @ self.x_posterior
            
            if self.B:
                self.x_prior += self.B @ u

            # Update the covariance matrix. Predict how much noise will be in the measurements.
            self.P_prior = self.F @ self.P_posterior @ self.F.transpose() + self.Q

        return (self.x_prior, self.P_prior)
    
    def __calculate_kalman_gain(self):
        """
        INPUT
        P: Error covariance of the state estimate. Quadratic matrix.

        OUTPUT: 
        K: Kalman gain. It specifies the degree to which the measurement is incorporated into the new state estimate.
        Or how much to trust this sensor
        """
        if self.__mode == '1D':
            self.S = self.H * self.P_prior + self.R
            K = self.P_prior / self.S
        else:
            # Innovation covariance.
            # This is the covariance of `self.H @ self.x_prior`
            self.S = self.H @ self.P_prior @ self.H.transpose() + self.R
            self.SI = np.linalg.inv(self.S)
            # Kalman gain
            K = self.P_prior @ self.H.transpose() @ self.SI
        
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

        K = self.__calculate_kalman_gain()
        
        if self.__mode == '1D':
            self.y = z - self.H * self.x_prior
            self.x_posterior = self.x_prior + K * self.y
            self.P_posterior = (1 - K * self.H) * self.P_prior

        else:
            # Innovation.
            # `self.H @ self.x_prior` maps the current state to the observation space.
            # This is the error between the prediction and measurement, in measurement space.
            self.y = z - self.H @ self.x_prior
            
            # Update the estimate with measurements from sensor.
            # This is the posterior.
            self.x_posterior = self.x_prior + K @ self.y
            
            I = np.identity(K.shape[0])
            # Covariance update equation. 
            # There is a simplified version, but is numerically unstable:
            # https://www.kalmanfilter.net/simpCovUpdate.html
            self.P_posterior = (I - K @ self.H) @ self.P_prior @ (I - K @ self.H).T + K @ self.R @ K.T

        self.history.append(self.x_posterior)
        return (self.x_posterior, self.P_posterior)




