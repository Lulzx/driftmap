"""Arakawa Jacobian for enstrophy-conserving advection.

The Arakawa scheme computes the Poisson bracket {φ, ζ} in a way that
conserves both energy AND enstrophy to machine precision. This is
critical for correctly capturing zonal flow generation in drift-wave
turbulence.

The standard scheme averages three forms of the Jacobian:
    J++ = ∂φ/∂x · ∂ζ/∂y - ∂φ/∂y · ∂ζ/∂x
    J+× = ∂(φ·∂ζ/∂y)/∂x - ∂(φ·∂ζ/∂x)/∂y
    J×+ = ∂(ζ·∂φ/∂x)/∂y - ∂(ζ·∂φ/∂y)/∂x

    J_Arakawa = (J++ + J+× + J×+) / 3

Reference: Arakawa, A. (1966). "Computational design for long-term
numerical integration of the equations of fluid motion."
J. Comp. Phys. 1, 119-143.
"""

import numpy as np


def arakawa_jacobian(phi: np.ndarray, zeta: np.ndarray,
                     dx: float, dy: float) -> np.ndarray:
    """Compute Arakawa Jacobian {φ, ζ}.

    This conserves both energy and enstrophy, which is essential for
    correct zonal flow dynamics in drift-wave turbulence.

    Args:
        phi: Electrostatic potential (nx, ny)
        zeta: Vorticity (nx, ny)
        dx: Grid spacing in x
        dy: Grid spacing in y

    Returns:
        Jacobian {φ, ζ} (nx, ny)
    """
    # Shift indices (periodic)
    def ip(f): return np.roll(f, -1, axis=0)  # i+1
    def im(f): return np.roll(f, 1, axis=0)   # i-1
    def jp(f): return np.roll(f, -1, axis=1)  # j+1
    def jm(f): return np.roll(f, 1, axis=1)   # j-1

    # J++ form
    Jpp = ((ip(phi) - im(phi)) * (jp(zeta) - jm(zeta)) -
           (jp(phi) - jm(phi)) * (ip(zeta) - im(zeta))) / (4 * dx * dy)

    # J+× form
    Jpx = (ip(phi) * (ip(jp(zeta)) - ip(jm(zeta))) -
           im(phi) * (im(jp(zeta)) - im(jm(zeta))) -
           jp(phi) * (ip(jp(zeta)) - im(jp(zeta))) +
           jm(phi) * (ip(jm(zeta)) - im(jm(zeta)))) / (4 * dx * dy)

    # J×+ form
    Jxp = (ip(jp(phi)) * (ip(zeta) - jp(zeta)) -
           im(jm(phi)) * (jm(zeta) - im(zeta)) -
           im(jp(phi)) * (jp(zeta) - im(zeta)) +
           ip(jm(phi)) * (jm(zeta) - ip(zeta))) / (4 * dx * dy)

    # Average of three forms
    return (Jpp + Jpx + Jxp) / 3


def compute_poisson_bracket(phi: np.ndarray, f: np.ndarray,
                            dx: float, dy: float,
                            method: str = 'arakawa') -> np.ndarray:
    """Compute Poisson bracket {φ, f}.

    Args:
        phi: Stream function / potential
        f: Field to advect
        dx, dy: Grid spacing
        method: 'arakawa' (conserving) or 'central' (simple)

    Returns:
        {φ, f}
    """
    if method == 'arakawa':
        return arakawa_jacobian(phi, f, dx, dy)
    elif method == 'central':
        # Simple central difference (does NOT conserve enstrophy)
        dphi_dx = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * dx)
        dphi_dy = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dy)
        df_dx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)
        df_dy = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dy)

        return dphi_dx * df_dy - dphi_dy * df_dx
    else:
        raise ValueError(f"Unknown method: {method}")


def test_conservation(phi: np.ndarray, zeta: np.ndarray,
                      dx: float, dy: float) -> dict:
    """Test conservation properties of Arakawa Jacobian.

    For a single timestep with only advection (no sources), the Arakawa
    scheme should conserve:
    - Total vorticity: ∫ζ dx = const
    - Energy: ∫|∇φ|² dx = const
    - Enstrophy: ∫ζ² dx = const

    Args:
        phi: Potential field
        zeta: Vorticity field
        dx, dy: Grid spacing

    Returns:
        Dictionary with conservation metrics
    """
    dA = dx * dy

    J = arakawa_jacobian(phi, zeta, dx, dy)

    # Total vorticity change: should be zero
    total_vorticity_change = np.sum(J) * dA

    # Enstrophy change: ∫ζ·J dx should be zero
    enstrophy_change = np.sum(zeta * J) * dA

    # Energy change: need to check ∫φ·J dx (related to energy)
    # Actually energy conservation involves ∫φ·∂ζ/∂t which = -∫φ·J
    energy_related_change = np.sum(phi * J) * dA

    return {
        'total_vorticity_change': total_vorticity_change,
        'enstrophy_change': enstrophy_change,
        'energy_related_change': energy_related_change,
    }
