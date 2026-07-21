"""Thermal Camera Calibration and Temperature Conversion (Equation 1 from paper)."""

import numpy as np
from scipy.optimize import minimize


# Factory default parameters for FLIR One Pro
FACTORY_PARAMS = {
    "R1": 18333.4,
    "B": 1435.0,
    "F": 1.0,
    "O": -2284.0,
    "R2": 0.0125
}

# Calibrated parameters optimized via Nelder-Mead (Table 2 from paper)
CALIBRATED_PARAMS = {
    "R1": 12755.4,
    "B": 1435.0,
    "F": 1.0,
    "O": -6707.0,
    "R2": 0.0125
}


def dn_to_temperature(dn: np.ndarray, params: dict = CALIBRATED_PARAMS) -> np.ndarray:
    """Convert Digital Number (DN) from FLIR radiometric JPEG to Temperature (°C).

    Equation (1) from paper:
    Temp (°C) = B / ln( R1 / (R2 * (DN + O)) + F ) - 273.15
    """
    R1 = params["R1"]
    B = params["B"]
    F_val = params["F"]
    O = params["O"]
    R2 = params["R2"]

    denominator = np.log((R1 / (R2 * (dn + O))) + F_val)
    temp_celsius = (B / denominator) - 273.15
    return temp_celsius


def calibrate_flir_one_pro(dn_values: np.ndarray, reference_temperatures: np.ndarray):
    """Calibrate R1 and O parameters using Nelder-Mead optimization on reference thermocouple data."""

    def loss_func(p):
        R1, O = p
        params = {"R1": R1, "B": 1435.0, "F": 1.0, "O": O, "R2": 0.0125}
        pred_temp = dn_to_temperature(dn_values, params)
        return np.mean((pred_temp - reference_temperatures) ** 2)

    initial_guess = [FACTORY_PARAMS["R1"], FACTORY_PARAMS["O"]]
    res = minimize(loss_func, initial_guess, method="Nelder-Mead", tol=1e-6)
    
    return {
        "R1": res.x[0],
        "B": 1435.0,
        "F": 1.0,
        "O": res.x[1],
        "R2": 0.0125
    }
