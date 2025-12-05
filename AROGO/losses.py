import numpy as np

class MSELoss:
    def forward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        return np.mean((y_true - y_pred)**2)

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.size
