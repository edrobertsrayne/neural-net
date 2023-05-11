import numpy as np
from scipy.special import softmax


class NeuralNetwork:
    """a simple neural network"""

    def __init__(
        self, inputs: int, hidden: int, outputs: int, learning_rate: float = 0.001
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.hidden = hidden
        self.alpha = learning_rate

        # initialise random weights and biases
        self.W1 = np.random.randn(hidden, inputs)
        self.W2 = np.random.randn(outputs, hidden)
        self.b1 = np.random.randn(hidden, 1)
        self.b2 = np.random.randn(outputs, 1)

    def ReLU(self, x: np.ndarray) -> np.ndarray:
        """return a rectified linear unit"""
        return np.maximum(0, x)

    def ReLU_gradient(self, x: np.ndarray) -> np.ndarray:
        """return the gradient of the ReLU function"""
        return x > 0

    def mean_squared_error(
        self, predicted: np.ndarray, expected: np.ndarray
    ) -> np.ndarray:
        """return the mean squared error of a prediction"""
        return np.mean(np.square(expected - predicted))

    def cross_entropy(self, predicted: np.ndarray, expected: np.ndarray) -> np.ndarray:
        """return the cross entropy error of a prediction"""
        return np.sum(expected * np.log(predicted))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """generate predictions from an input matrix"""
        # hidden layer
        self.Z1 = self.W1.dot(X) + self.b1
        self.A1 = self.ReLU(self.Z1)

        # output layer
        self.Z2 = self.W2.dot(self.A1) + self.b2
        if self.outputs == 1:
            return self.ReLU(self.Z2)
        else:
            return softmax(self.Z2)

    def train(self, X: np.ndarray, Y: np.ndarray, iterations: int = 1000) -> None:
        """train the neural network"""
        n, m = Y.shape
        for i in range(iterations):
            # generate some predictions
            Y_hat = self.predict(X)

            # calculate the differentials
            dZ2 = Y_hat - Y
            dW2 = 1 / m * dZ2.dot(self.A1.T)
            db2 = 1 / m * np.sum(dZ2)
            dZ1 = self.W2.T.dot(dZ2) * self.ReLU_gradient(self.Z1)
            dW1 = 1 / m * dZ1.dot(X.T)
            db1 = 1 / m * np.sum(dZ1)

            # adjust the weights and biases
            self.W1 = self.W1 - self.alpha * dW1
            self.b1 = self.b1 - self.alpha * db1
            self.W2 = self.W2 - self.alpha * dW2
            self.b2 = self.b2 - self.alpha * db2

            # output progress every 10 iterations
            if i % 10 == 0:
                print(f"Iteration: {i}\tError: {self.mean_squared_error(Y, Y_hat)}")
