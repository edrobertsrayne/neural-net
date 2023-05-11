import numpy as np
from scipy.special import softmax


class NeuralNetwork:
    """_summary_"""

    def __init__(self, inputs: int, hidden: int, outputs: int) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.hidden = hidden

        self.W1 = np.random.randn(inputs, hidden)
        self.W2 = np.random.randn(hidden, outputs)
        self.B1 = np.random.randn(hidden)
        self.B2 = np.random.randn(outputs)

    def relu(self, x):
        """return a rectified linear unit"""
        return np.maximum(x, np.zeros(x.shape))

    def mean_squared_error(
        self, expected: np.ndarray, predicted: np.ndarray
    ) -> np.ndarray:
        """return the mean squared error of a prediction"""
        return np.square(expected - predicted)

    def cross_entropy_error(self, expected, predicted):
        """return the corss entropy error of a prediction"""
        return np.sum(expected * np.log(predicted))

    def predict(self, X: np.ndarray) -> np.ndarray:
        # hidden layer
        Z1 = np.dot(X.T, self.W1) + self.B1
        A1 = self.relu(Z1)

        # output layer
        Z2 = np.dot(A1, self.W2) + self.B2
        if self.hidden == 1:
            return self.relu(Z2)
        else:
            return softmax(Z2)
