import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import base64
from flask import Flask, request, jsonify  # Added request and jsonify
from io import BytesIO

# Create Flask app
app = Flask(__name__)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

def plot_model(X, y, model, epoch=None):
    plt.figure()
    plt.scatter(X, y, color='blue', label='Data points')

    with torch.no_grad():
        x_vals = torch.linspace(X.min(), X.max(), 100).unsqueeze(1)
        y_vals = model(x_vals).numpy()
        plt.plot(x_vals.numpy(), y_vals, color='red', label='Model prediction')

    if epoch is not None:
        plt.title(f'Epoch {epoch+1}')
    else:
        plt.title('Final Trained Model')

    plt.xlabel('Feature')
    plt.ylabel('Label')
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

@app.route('/train_linear', methods=['POST'])
def train_linear():
    data = request.get_json()
    print("training started")

    features = data.get('features')
    labels = data.get('labels')

    if not features or not labels:
        return jsonify({"error": "Missing features or labels"}), 400

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32).reshape(-1, 1)

    input_dim = X.shape[1] if X.ndim > 1 else 1
    X = X.reshape(-1, input_dim)  # Ensure 2D
    model = LinearRegressionModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.15)

    inputs = torch.tensor(X, dtype=torch.float32)
    targets = torch.tensor(y, dtype=torch.float32)

    training_images = []
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        img = plot_model(X, y, model, epoch)
        training_images.append(img)

    completed_image = plot_model(X, y, model)  # Final result

    print("training complete")

    return jsonify({
        "message": "Model trained successfully",
        "completed_image": completed_image,
        "training_images": training_images
    }), 200

if __name__ == '__main__':
    app.run(debug=True)