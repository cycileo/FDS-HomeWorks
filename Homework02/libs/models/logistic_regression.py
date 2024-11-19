import numpy as np
from libs.math import sigmoid

class LogisticRegression:
    def __init__(self, num_features : int):
        # self.parameters = np.random.normal(0, 0.01, num_features)
        self.parameters = np.array([-0.01062133, 0.09915262, 0.09556893])
        
    def predict(self, x:np.array) -> np.array:
        """
        Method to compute the predictions for the input features.

        Args:
            x: it's the input data matrix.

        Returns:
            preds: the predictions of the input features.
        """
        ##############################
        # Compute the linear combination \theta^T * x
        linear_combination = np.dot(x, self.parameters)
        preds = sigmoid(linear_combination)
        ##############################
        return preds
    
    @staticmethod
    def likelihood(preds, y : np.array) -> float:    # it was np.array
        """
        Function to compute the log likelihood of the model parameters according to data x and label y.

        Args:
            preds: the predicted labels.
            y: the label array.

        Returns:
            log_l: the log likelihood of the model parameters according to data x and label y.
        """
        ##############################        
        # Clip the predictions to prevent log(0)
        clipped_preds = np.clip(preds, 1e-10, 1 - 1e-10)  # Prevent log(0)
        # Compute the log of likelihood
        log_l = np.mean( y * np.log(clipped_preds) + (1 - y) * np.log(1 - clipped_preds) )
        ##############################
        return log_l
    
    def update_theta(self, gradient: np.array, lr : float = 0.5):
        """
        Function to update the weights in-place.

        Args:
            gradient: the gradient of the log likelihood.
            lr: the learning rate.

        Returns:
            None
        """
        ##############################
        self.parameters += lr * gradient
        ##############################
        pass
        
    @staticmethod
    def compute_gradient(x : np.array, y: np.array, preds: np.array) -> np.array:
        """
        Function to compute the gradient of the log likelihood.

        Args:
            x: it's the input data matrix.
            y: the label array.
            preds: the predictions of the input features.

        Returns:
            gradient: the gradient of the log likelihood.
        """
        ##############################
        gradient = np.dot(x.T, (y - preds)) / len(y)
        ##############################
        return gradient
    
    # Print model parameters, added for debug and report 1
    @staticmethod
    def print_params(self):
        print("params", self.parameters)

