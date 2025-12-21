"""Parameter sets from Kadoch et al. (PoP 29, 102301, 2022).

These configs are intended for validation scripts and are not used by pytest.
"""

CONFIGS = {
    "cHW_0.7": {
        "alpha": 0.7,
        "kappa": 1.0,
        "nu": 5e-3,   # Laplacian viscosity
        "D": 5e-3,    # Density diffusion
        "Lx": 64.0,
        "Ly": 64.0,
        "nx": 256,
        "ny": 256,
        "dt": 5e-4,
        "modified_hw": False,
    },
    "cHW_4.0": {
        "alpha": 4.0,
        "kappa": 1.0,
        "nu": 5e-3,
        "D": 5e-3,
        "Lx": 64.0,
        "Ly": 64.0,
        "nx": 256,
        "ny": 256,
        "dt": 5e-4,
        "modified_hw": False,
    },
    "mHW_4.0": {
        "alpha": 4.0,
        "kappa": 1.0,
        "nu": 5e-3,
        "D": 5e-3,
        "Lx": 64.0,
        "Ly": 64.0,
        "nx": 256,
        "ny": 256,
        "dt": 5e-4,
        "modified_hw": True,
    },
}
