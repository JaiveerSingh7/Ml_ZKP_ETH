# generate_witness_input.py

import json

# Load model
with open("model.json") as f:
    model = json.load(f)

# Load user input (secret x)
with open("input.json") as f:
    user_input = json.load(f)

# Build circuit input
circuit_input = {
    "x": user_input["x"],
    "weights": model["weights"],
    "bias": model["bias"]
}

# Save to circuit_input.json
with open("circuit_input.json", "w") as f:
    json.dump(circuit_input, f)

print("✅ circuit_input.json generated!")
