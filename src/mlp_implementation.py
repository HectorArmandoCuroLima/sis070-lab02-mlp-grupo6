import numpy as np

# Funciones de activación
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, hidden_size2=None, activation='relu'):
        np.random.seed(42)
        self.activation_name = activation
        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            raise ValueError("activation must be 'relu' or 'sigmoid'")

        # Capa 1: entrada -> oculta 1
        self.W1 = np.random.randn(input_size, hidden_size) * 1.0
        self.b1 = np.zeros((1, hidden_size))

        if hidden_size2 is not None:
            # Capa 2: oculta 1 -> oculta 2
            self.W2 = np.random.randn(hidden_size, hidden_size2) * 1.0
            self.b2 = np.zeros((1, hidden_size2))
            # Capa 3: oculta 2 -> salida
            self.W3 = np.random.randn(hidden_size2, output_size) * 1.0
            self.b3 = np.zeros((1, output_size))
            self.has_second_hidden = True
        else:
            # Capa 2: oculta 1 -> salida
            self.W2 = np.random.randn(hidden_size, output_size) * 1.0
            self.b2 = np.zeros((1, output_size))
            self.has_second_hidden = False

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.activation(self.Z1)

        if self.has_second_hidden:
            self.Z2 = np.dot(self.A1, self.W2) + self.b2
            self.A2 = self.activation(self.Z2)
            self.Z3 = np.dot(self.A2, self.W3) + self.b3
            self.A3 = sigmoid(self.Z3)
            return self.A3
        else:
            self.Z2 = np.dot(self.A1, self.W2) + self.b2
            self.A2 = sigmoid(self.Z2)
            return self.A2

    def train(self, X, y, learning_rate=0.1, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y) ** 2)

            if self.has_second_hidden:
                dZ3 = (output - y) * sigmoid_derivative(self.Z3)
                dW3 = np.dot(self.A2.T, dZ3)
                db3 = np.sum(dZ3, axis=0, keepdims=True)

                dA2 = np.dot(dZ3, self.W3.T)
                dZ2 = dA2 * self.activation_derivative(self.Z2)
                dW2 = np.dot(self.A1.T, dZ2)
                db2 = np.sum(dZ2, axis=0, keepdims=True)

                dA1 = np.dot(dZ2, self.W2.T)
                dZ1 = dA1 * self.activation_derivative(self.Z1)
                dW1 = np.dot(X.T, dZ1)
                db1 = np.sum(dZ1, axis=0, keepdims=True)

                self.W3 -= learning_rate * dW3
                self.b3 -= learning_rate * db3
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1
            else:
                dZ2 = (output - y) * sigmoid_derivative(self.Z2)
                dW2 = np.dot(self.A1.T, dZ2)
                db2 = np.sum(dZ2, axis=0, keepdims=True)

                dA1 = np.dot(dZ2, self.W2.T)
                dZ1 = dA1 * self.activation_derivative(self.Z1)
                dW1 = np.dot(X.T, dZ1)
                db1 = np.sum(dZ1, axis=0, keepdims=True)

                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1

            if epoch % 200 == 0:
                print(f"Época {epoch} - Pérdida: {loss:.6f}")

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    print("=== ACTIVIDAD 3: MLP con 2 capas ocultas (4 capas en total) ===")
    mlp_2h = MLP(input_size=2, hidden_size=4, hidden_size2=3, output_size=1, activation='relu')
    mlp_2h.train(X, y, learning_rate=0.5, epochs=5000)
    print("Predicciones finales:\n", mlp_2h.forward(X))

    print("\n=== ACTIVIDAD 3: MLP con 2 capas ocultas y Sigmoide ===")
    mlp_2h_sig = MLP(input_size=2, hidden_size=4, hidden_size2=3, output_size=1, activation='sigmoid')
    mlp_2h_sig.train(X, y, learning_rate=1.0, epochs=5000)
    print("Predicciones finales:\n", mlp_2h_sig.forward(X))