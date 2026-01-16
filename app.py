from flask import Flask, request, jsonify
from heat_sink_model import compute_temperature
from pinn_thermal import PINN, MATERIALS, material_one_hot
import torch

app = Flask(__name__)

# Load pre-trained model
model = PINN(num_materials=len(MATERIALS))
model.load_state_dict(torch.load("trained_pinn.pth"))
model.eval()

POWER_SCALE = 100.0
TEMP_SCALE = 100.0

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    power = float(data.get("power", 50))
    ambient_temp = float(data.get("ambient_temp", 25))
    material = data.get("material", "Aluminum").lower()

    if material not in MATERIALS:
        return jsonify({"error": f"Unknown material '{material}'. Available: {MATERIALS}"}), 400

    # Physics model prediction
    physics_temp = compute_temperature(power, ambient_temp, material)

    # PINN prediction
    pinn_input = torch.tensor([[power/POWER_SCALE, ambient_temp/TEMP_SCALE] + material_one_hot(material).tolist()], dtype=torch.float32)
    pinn_temp = model(pinn_input).item() * TEMP_SCALE

    return jsonify({
        "physics_model_temp": round(physics_temp, 2),
        "pinn_predicted_temp": round(pinn_temp, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
