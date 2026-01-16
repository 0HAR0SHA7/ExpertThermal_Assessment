"""
heat_sink_model.py

Thermal model for heat sink temperature prediction.
Includes material properties and a simple physics-based formula.
"""

# Thermal resistance (C/W) for different materials
THERMAL_RESISTANCE = {
    "aluminum": 0.5,
    "copper": 0.3
}

def compute_temperature(power: float, ambient_temp: float, material: str) -> float:
    """
    Compute the heat sink temperature.
    :param power: Power in watts
    :param ambient_temp: Ambient temperature in Celsius
    :param material: Material name ('Aluminum' or 'Copper')
    :return: Temperature in Celsius
    """
    material_lower = material.lower()
    if material_lower not in THERMAL_RESISTANCE:
        raise ValueError(f"Unknown material '{material}'. Use 'Aluminum' or 'Copper'.")
    
    thermal_resistance = THERMAL_RESISTANCE[material_lower]
    temp_rise = power * thermal_resistance
    return ambient_temp + temp_rise

# For testing
if __name__ == "__main__":
    temp = compute_temperature(50, 25, "Aluminum")
    print(f"Predicted temperature: {temp:.2f}°C")
