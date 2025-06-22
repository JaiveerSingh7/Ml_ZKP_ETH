ZKP + ML: Verifiable Linear Regression
Project Overview
This project demonstrates how to:

Train an ML model (Linear Regression)

Generate Zero-Knowledge Proofs (ZKP) of model inference

Verify the proof on-chain in a smart contract

Problem Statement
How can we prove that an ML model prediction is correct without revealing:

the model weights and bias

the input data

Solution
Using ZK-SNARKs, we can generate a cryptographic proof that:

ini
Copy
Edit
y = W * x + b
without revealing W, b, or x.
Anyone can verify the proof on-chain.

Architecture
css
Copy
Edit
[Off-chain ML training] ---> [Generate ZKP Proof] ---> [Smart Contract Verifier]
ML Pipeline
Train Linear Regression using scikit-learn

Export model weights and bias to model.json

Export input data to circuit_input.json

ZKP Pipeline
Compile Circom circuit (linear_regression.circom)

Perform trusted setup (.zkey)

Generate witness

Generate proof (proof.json, public.json)

Verify proof on-chain

Usage
1. Train model and export weights
bash
Copy
Edit
python train_export_model.py
2. Prepare circuit input
bash
Copy
Edit
python generate_witness_input.py
3. Compile circuit
bash
Copy
Edit
circom linear_regression.circom --r1cs --wasm --sym --c
4. Trusted setup
bash
Copy
Edit
snarkjs powersoftau new bn128 12 pot12_0000.ptau -v
snarkjs powersoftau contribute pot12_0000.ptau pot12_final.ptau -v
snarkjs powersoftau prepare phase2 pot12_final.ptau pot12_final_phase2.ptau
snarkjs groth16 setup linear_regression.r1cs pot12_final_phase2.ptau linear_regression_0000.zkey
snarkjs zkey contribute linear_regression_0000.zkey linear_regression_final.zkey -v
5. Generate proof
bash
Copy
Edit
node linear_regression_js/generate_witness.js linear_regression_js/linear_regression.wasm circuit_input.json witness.wtns
snarkjs groth16 prove linear_regression_final.zkey witness.wtns proof.json public.json
6. Verify proof
bash
Copy
Edit
snarkjs groth16 verify verification_key.json public.json proof.json
On-chain Verifier
1. Export Solidity verifier
bash
Copy
Edit
snarkjs zkey export solidityverifier linear_regression_final.zkey verifier.sol
2. Deploy with Foundry
bash
Copy
Edit
forge create src/LinearRegressionZKP.sol --rpc-url <your_rpc_url>
3. Call verifyProof on-chain
Pass proof.json and public.json as calldata to the smart contract verifier.

Experiments
Measure model accuracy (RMSE)

Measure ZKP generation time

Measure on-chain verification gas cost

Paper Ideas
On-chain verification of ML inference

Verifiable AI for DeFi

Privacy-preserving AI for sensitive data (e.g., medical, financial)

