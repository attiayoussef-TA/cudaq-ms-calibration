#!/usr/bin/env python3
"""
CUDA-Q effective Mølmer–Sørensen (MS) gate calibration / geometric-phase optimizer.

Optimizes the coupling strength χ and interaction time t so that
χ·t ≈ π/8 while driving the computational-basis populations to the
ideal MS Bell-state values (P00 = P11 = 0.5).
"""

import cudaq
import numpy as np
from cudaq import spin, Schedule, RungeKuttaIntegrator
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
import cupy as cp

# Target 

cudaq.set_target("dynamics")

# Two-qubit Hilbert space (computational basis)
dimensions = {0: 2, 1: 2}

# Initial state |00>
psi0_vec = cp.zeros(4, dtype=cp.complex128)
psi0_vec[0] = 1.0
rho0 = cudaq.State.from_data(psi0_vec)

# Reference coupling strength (typical trapped-ion MS scale) and target phase

chi_ref = 2.0 * np.pi * 25e3          # rad/s
phi_target = np.pi / 8                # geometric phase we want to accumulate

# Hamiltonian & observables

def build_hamiltonian(chi):
    """Effective MS interaction: H = χ (S_x)^2  with S_x = σ_x⁰ + σ_x¹."""
    Sx = spin.x(0) + spin.x(1)
    return chi * (Sx * Sx)

# Projectors onto |00> and |11>.
# For the ideal MS Bell state both populations equal 0.5.
P00 = 0.25 * (spin.i(0) + spin.z(0)) * (spin.i(1) + spin.z(1))
P11 = 0.25 * (spin.i(0) - spin.z(0)) * (spin.i(1) - spin.z(1))
observables = [P00, P11]

# Time evolution helper

def simulate(chi, t_final, n_steps=200, return_full=False):
    """
    Evolve the density matrix under the MS Hamiltonian and return
    a fidelity proxy that equals 1 when P00 = P11 = 0.5.
    """
    t_final = max(float(t_final), 1e-9)          # avoid zero-time edge case
    H = build_hamiltonian(chi)
    steps = np.linspace(0.0, t_final, n_steps)
    schedule = Schedule(steps, ["t"])

    result = cudaq.evolve(
        H, dimensions, schedule, rho0,
        observables=observables,
        collapse_operators=[],                   # pure unitary evolution
        store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
        integrator=RungeKuttaIntegrator(),
    )

    exp = result.expectation_values()
    p00 = np.array([e[0].expectation() for e in exp])
    p11 = np.array([e[1].expectation() for e in exp])

    # Simple fidelity-like figure of merit (1 = ideal MS populations)
    fid = 1.0 - np.abs(p00 - 0.5) - np.abs(p11 - 0.5)
    fid = np.clip(fid, 0.0, 1.0)

    if return_full:
        return steps, fid, p00, p11
    return fid[-1]

# Cost function for the classical optimizer

def cost_function(x):
    """
    Minimize:
      (1 - fidelity) + 3·(χt − φ_target)² + small time penalty
    Parameters are log-scaled so the optimizer works in a more linear space.
    """
    chi = np.exp(x[0])
    t_final = np.exp(x[1])

    # Soft box constraints (return large cost outside sensible physical range)
    if not (1e3 < chi < 1e7) or not (5e-7 < t_final < 2e-5):
        return 10.0

    fid_final = simulate(chi, t_final, n_steps=100)
    phase_err = (chi * t_final - phi_target)**2
    time_pen = 0.02 * (t_final * 1e6 - 3.0)**2   # prefer ~3 µs

    return (1.0 - fid_final) + 3.0 * phase_err + time_pen

# Optimization

bounds = Bounds(
    lb=[np.log(5e3), np.log(1e-6)],
    ub=[np.log(1e6), np.log(1e-5)]
)
x0 = [np.log(chi_ref), np.log(phi_target / chi_ref)]

res = minimize(
    cost_function, x0,
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter": 40, "ftol": 1e-12},
)

chi_opt = np.exp(res.x[0])
t_opt = np.exp(res.x[1])

# High-resolution trajectory at the optimized point
steps, fid, p00, p11 = simulate(chi_opt, t_opt, n_steps=300, return_full=True)

# Console

print("CUDA-Q MS Bell-state effective calibration")
print(f"chi_opt / 2π = {chi_opt/(2*np.pi):.3e} Hz")
print(f"t_opt = {t_opt*1e6:.3f} µs")
print(f"χ·t = {chi_opt * t_opt:.6f} rad (target π/8 = {np.pi/8:.6f})")
print(f"fidelity = {fid[-1]:.6f}")
print(f"P00 = {p00[-1]:.4f}")
print(f"P11 = {p11[-1]:.4f}")
print(f"optimizer success = {res.success}, nfev = {res.nfev}")

# Plot 1: fidelity proxy + populations vs time

plt.figure(figsize=(8, 5))
plt.plot(steps * 1e6, fid, label="MS fidelity proxy")
plt.plot(steps * 1e6, p00, "--", label=r"$P(|00\rangle)$")
plt.plot(steps * 1e6, p11, "--", label=r"$P(|11\rangle)$")
plt.axhline(0.95, ls=":", color="gray", label="0.95 threshold")
plt.axvline(t_opt * 1e6, ls=":", color="C1", label="optimal time")
plt.xlabel("Time (µs)")
plt.ylabel("Value")
plt.title("MS fidelity proxy and computational-basis populations")
plt.legend()
plt.tight_layout()

# Plot 2: accumulated geometric phase χt vs time

phase = chi_opt * steps
plt.figure(figsize=(8, 4))
plt.plot(steps * 1e6, phase, label=r"$\chi t$")
plt.axhline(phi_target, ls="--", color="C1", label=r"target $\pi/8$")
plt.axvline(t_opt * 1e6, ls=":", color="gray")
plt.xlabel("Time (µs)")
plt.ylabel("Accumulated phase (rad)")
plt.title("Geometric phase accumulation under the optimized MS drive")
plt.legend()
plt.tight_layout()

plt.show()
