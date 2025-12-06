class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train(self, X, Y, loss_fn, optimizer, epochs=5000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = loss_fn.forward(Y, y_pred)

            grad = loss_fn.backward()
            self.backward(grad)
            optimizer.step(self.layers)

            if epoch % 2000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
