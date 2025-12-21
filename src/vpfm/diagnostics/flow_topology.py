"""Flow topology diagnostics (Okubo-Weiss partitioning)."""

import numpy as np


def compute_weiss_field(vx: np.ndarray, vy: np.ndarray, dx: float, dy: float) -> tuple:
    """Compute Okubo-Weiss field Q and vorticity omega.

    Args:
        vx, vy: Velocity components on grid
        dx, dy: Grid spacing

    Returns:
        (Q, omega) where Q = sigma^2 - omega^2
    """
    dudx = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2.0 * dx)
    dudy = (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)) / (2.0 * dy)
    dvdx = (np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)) / (2.0 * dx)
    dvdy = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2.0 * dy)

    omega = dvdx - dudy
    s1 = dudx - dvdy
    s2 = dudy + dvdx
    sigma_sq = s1**2 + s2**2
    Q = sigma_sq - omega**2

    return Q, omega


def partition_flow(Q: np.ndarray, threshold_factor: float = 1.0) -> tuple:
    """Partition flow into elliptic, intermediate, hyperbolic regions.

    Args:
        Q: Okubo-Weiss field
        threshold_factor: Multiplier for Q0 = std(Q)

    Returns:
        (labels, Q0) where labels are -1 (elliptic), 0 (intermediate), +1 (hyperbolic)
    """
    Q0 = threshold_factor * np.std(Q)
    labels = np.zeros_like(Q, dtype=int)
    labels[Q <= -Q0] = -1
    labels[Q >= Q0] = 1
    return labels, Q0


def compute_residence_times(label_history: np.ndarray, dt: float) -> dict:
    """Compute residence time samples for each region.

    Args:
        label_history: Array of shape (n_particles, n_steps)
        dt: Timestep size

    Returns:
        Dict mapping region label to list of residence durations.
    """
    residence_times = {-1: [], 0: [], 1: []}
    n_particles, n_steps = label_history.shape

    for p in range(n_particles):
        labels = label_history[p]
        current_label = labels[0]
        current_duration = 1

        for t in range(1, n_steps):
            if labels[t] == current_label:
                current_duration += 1
            else:
                residence_times[current_label].append(current_duration * dt)
                current_label = labels[t]
                current_duration = 1

        residence_times[current_label].append(current_duration * dt)

    return residence_times
