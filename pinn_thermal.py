import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from heat_sink_model import compute_temperature  # Your physics model

# ---------------------------
# Materials (can add more)
# ---------------------------
MATERIALS = ["aluminum", "copper"]  # Add new materials here

def material_one_hot(material):
    """Convert material name to one-hot vector"""
    vec = np.zeros(len(MATERIALS), dtype=np.float32)
    idx = MATERIALS.index(material.lower())
    vec[idx] = 1.0
    return vec

# ---------------------------
# PINN Model
# ---------------------------
class PINN(nn.Module):
    def __init__(self, num_materials=len(MATERIALS)):
        super(PINN, self).__init__()
        input_size = 2 + num_materials  # power + ambient_temp + one-hot material
        self.net = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# Training Data
# ---------------------------
def generate_training_data(num_samples=200):
    powers = np.random.uniform(10, 100, num_samples)
    ambient_temps = np.random.uniform(20, 40, num_samples)

    X_list = []
    y_list = []

    for p, t in zip(powers, ambient_temps):
        for m in MATERIALS:
            mat_vec = material_one_hot(m)
            # Normalize power and ambient temperature
            X_list.append(np.concatenate(([p/100, t/100], mat_vec)))
            y_list.append(compute_temperature(p, t, m)/100)  # scale output

    # Convert lists to arrays to avoid slow tensor creation
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return torch.tensor(X), torch.tensor(y).unsqueeze(1)

# ---------------------------
# Loss & Training
# ---------------------------
def pinn_loss(model, x, y_true):
    return nn.MSELoss()(model(x), y_true)

def train_pinn(model, x_train, y_train, epochs=500, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = pinn_loss(model, x_train, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")
    return model

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("Generating training data...")
    X_train, y_train = generate_training_data()
    model = PINN()
    print("Training PINN...")
    model = train_pinn(model, X_train, y_train)

    # Save model
    torch.save(model.state_dict(), "trained_pinn.pth")
    print("Model saved as 'trained_pinn.pth'")

    # Test prediction for all materials
    for mat in MATERIALS:
        test_input = torch.tensor([[50/100, 25/100] + material_one_hot(mat).tolist()], dtype=torch.float32)
        pred_temp = model(test_input).item() * 100
        print(f"PINN predicted temperature for {mat.capitalize()}: {pred_temp:.2f}°C")
