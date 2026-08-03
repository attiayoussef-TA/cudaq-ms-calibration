#  ⚛️ CUDA-Q Effective Quantum Mølmer–Sørensen (MS) Gate Calibration

Optimizes the coupling strength χ and interaction time t of an effective two-qubit Mølmer–Sørensen interaction so that the geometric phase reaches the target value χ·t ≈ π/8, and the computational-basis populations approach the ideal MS Bell-state values P₀₀ = P₁₁ = 0.5.

The evolution is performed with the **CUDA-Q dynamics** backend (Runge–Kutta integrator) on GPU.

## Features

- Effective MS Hamiltonian: H = χ (Sₓ)² with Sₓ = σₓ⁰ + σₓ¹  
- Classical optimizer (L-BFGS-B) that jointly tunes χ and t  
- Fidelity proxy based on populations of |00⟩ and |11⟩  
- High-resolution time traces of fidelity, populations and accumulated phase  
- Ready-to-run on Google Colab with a Tesla T4

## Installation (Google Colab + Tesla T4)

1. Select a **GPU runtime** (Tesla T4).
2. Install CUDA-Q 0.14.0 following the official NVIDIA instructions.
3. Install the remaining Python dependencies:
   
```bash
## pip install -r requirements.txt
```
## Output

- CUDA-Q MS Bell-state effective calibration
- chi_opt / 2π = 2.500e+04 Hz
- t_opt = 2.500 µs
- χ·t = 0.392699 rad (target π/8 = 0.392699)
- fidelity = 1.000000
- P00 = 0.5000
- P11 = 0.5000
- optimizer reached the target phase and Bell-state populations, nfev = 57
