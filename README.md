# Expert Thermal Assessment 

## Overview

This project implements a physics-informed neural network (PINN) to predict heat sink temperatures. 
It includes a Flask API and a test script to validate predictions for multiple materials.

---

## Folder Structure

- `heat_sink_model.py`: Physics-based heat sink model
- `pinn_thermal.py`: PINN training script
- `app.py`: Flask API
- `test_api.py`: API testing script
- `requirements.txt`: Python dependencies

---

## Setup

1. Clone/download the folder to your local machine.
2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Linux/Mac

