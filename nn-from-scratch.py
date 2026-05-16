import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self._initialize_parameters()   
    
    def _initialize_parameters(self):
        np.random.seed(42)
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.01
            b = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.relu(z)
            self.activations.append(a)
        
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        a = self.softmax(z)
        self.activations.append(a)
        
        return a
    
    def backward(self, X, y, m):
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        dz = self.activations[-1] - y
        
        for i in range(len(self.weights) - 1, -1, -1):
            dW[i] = np.dot(self.activations[i].T, dz) / m
            db[i] = np.sum(dz, axis=0, keepdims=True) / m
            
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * self.relu_derivative(self.z_values[i - 1])
        
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def train(self, X, y, epochs=100, batch_size=32):
        m = X.shape[0]
        for epoch in range(epochs):
            for i in range(0, m, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                
                self.forward(X_batch)
                self.backward(X_batch, y_batch, X_batch.shape[0])
            
            if (epoch + 1) % 20 == 0:
                predictions = self.forward(X)
                loss = -np.mean(np.log(predictions[np.arange(m), np.argmax(y, axis=1)] + 1e-8))
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# One-hot encode labels
y_onehot = np.eye(3)[y]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train neural network
nn = NeuralNetwork(layer_sizes=[4, 16, 8, 3], learning_rate=0.01)
nn.train(X_train, y_train, epochs=100)

# Evaluate
y_pred = nn.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_labels, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred, target_names=iris.target_names))
