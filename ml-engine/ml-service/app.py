from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

app = Flask(__name__)

# === PyTorch Logistic Regression Model ===
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# === PyTorch Linear Regression Model ===
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# === TRAINING ROUTES ===

@app.route('/train_logistic', methods=['POST'])
def train_logistic():
    data = request.get_json()
    X = np.array(data['features'], dtype=np.float32)
    y = np.array(data['labels'], dtype=np.float32).reshape(-1, 1)

    model = LogisticRegressionModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    images = []
    for epoch in range(20):
        inputs = torch.tensor(X)
        labels = torch.tensor(y)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        img = plot_decision_boundary(X, y.ravel(), model, epoch)
        images.append(img)

    return jsonify({'message': 'Logistic model trained', 'images': images})


@app.route('/train_linear', methods=['POST'])
def train_linear():
    data = request.get_json()
    print("data", data)
    X = np.array(data['features'], dtype=np.float32).reshape(-1, 1)
    y = np.array(data['labels'], dtype=np.float32).reshape(-1, 1)

    model = LinearRegressionModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    images = []
    for epoch in range(20):
        inputs = torch.tensor(X)
        labels = torch.tensor(y)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        img = plot_linear(X, y, model, epoch)
        images.append(img)

    return jsonify({'message': 'Linear model trained', 'images': images})


# === PLOTTING ===

def plot_decision_boundary(X, y, model, iteration):
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    with torch.no_grad():
        logits = model(torch.tensor(grid, dtype=torch.float32))
    Z = logits.reshape(xx.shape).numpy()

    plt.contourf(xx, yy, Z, levels=[0,0.5,1], alpha=0.2)
    plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr')
    plt.title(f"Logistic Iteration {iteration}")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_linear(X, y, model, iteration):
    with torch.no_grad():
        x_tensor = torch.tensor(X)
        y_pred = model(x_tensor).numpy()

    plt.scatter(X, y, color='blue')
    plt.plot(X, y_pred, color='red')
    plt.title(f"Linear Iteration {iteration}")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


if __name__ == '__main__':
    app.run(port=5000)
