"""Flux diagnostics for SOL turbulence validation.

Implements virtual probe measurements to compare with experimental
data from MAST-U, ASDEX-Upgrade, and other machines.

Key metrics:
- Particle flux: Γ = <ñṽ_r>
- Heat flux: Q = <T̃ṽ_r> (if temperature is tracked)
- Flux statistics: skewness, kurtosis, intermittency
- Blob detection and tracking
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class FluxStatistics:
    """Statistics of turbulent flux at a probe location."""
    mean: float
    std: float
    skewness: float
    kurtosis: float
    intermittency: float  # Fraction of flux in bursts
    max_event: float      # Largest flux event


class VirtualProbe:
    """Virtual probe for measuring turbulent flux.

    Mimics experimental Langmuir probes or gas-puff imaging diagnostics.
    """

    def __init__(self, x_pos: float, y_range: Tuple[float, float],
                 sample_rate: int = 1):
        """Initialize probe.

        Args:
            x_pos: Radial position of probe line
            y_range: (y_min, y_max) poloidal range
            sample_rate: Sample every n timesteps
        """
        self.x_pos = x_pos
        self.y_min, self.y_max = y_range
        self.sample_rate = sample_rate

        # Time series storage
        self.times: List[float] = []
        self.flux_series: List[np.ndarray] = []
        self.density_series: List[np.ndarray] = []
        self.velocity_series: List[np.ndarray] = []

    def measure(self, time: float, n_grid: np.ndarray, vx_grid: np.ndarray,
                grid_x: np.ndarray, grid_y: np.ndarray):
        """Take a measurement at the probe location.

        Args:
            time: Current simulation time
            n_grid: Density field (nx, ny)
            vx_grid: Radial velocity field (nx, ny)
            grid_x: x coordinates (1D)
            grid_y: y coordinates (1D)
        """
        # Find grid index closest to probe x position
        ix = np.argmin(np.abs(grid_x - self.x_pos))

        # Find y indices in range
        y_mask = (grid_y >= self.y_min) & (grid_y <= self.y_max)

        # Extract profiles
        n_profile = n_grid[ix, y_mask]
        vx_profile = vx_grid[ix, y_mask]
        flux_profile = n_profile * vx_profile

        self.times.append(time)
        self.density_series.append(n_profile.copy())
        self.velocity_series.append(vx_profile.copy())
        self.flux_series.append(flux_profile.copy())

    def compute_statistics(self) -> FluxStatistics:
        """Compute flux statistics from time series.

        Returns:
            FluxStatistics with mean, std, skewness, kurtosis
        """
        if len(self.flux_series) < 10:
            return FluxStatistics(0, 0, 0, 0, 0, 0)

        # Concatenate all flux measurements
        all_flux = np.concatenate(self.flux_series)

        mean = np.mean(all_flux)
        std = np.std(all_flux)

        if std < 1e-10:
            return FluxStatistics(mean, std, 0, 0, 0, 0)

        # Normalized moments
        normalized = (all_flux - mean) / std
        skewness = np.mean(normalized**3)
        kurtosis = np.mean(normalized**4) - 3  # Excess kurtosis

        # Intermittency: fraction of flux from events > 2σ
        large_events = np.abs(all_flux - mean) > 2 * std
        if np.sum(large_events) > 0:
            intermittency = np.sum(np.abs(all_flux[large_events])) / np.sum(np.abs(all_flux))
        else:
            intermittency = 0.0

        max_event = np.max(np.abs(all_flux))

        return FluxStatistics(
            mean=mean,
            std=std,
            skewness=skewness,
            kurtosis=kurtosis,
            intermittency=intermittency,
            max_event=max_event,
        )

    def get_time_series(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get spatially-averaged flux vs time.

        Returns:
            (times, flux) arrays
        """
        times = np.array(self.times)
        flux = np.array([np.mean(f) for f in self.flux_series])
        return times, flux


class BlobDetector:
    """Detect and track coherent structures (blobs).

    Uses threshold-based detection to identify blob-like structures
    in the density field.
    """

    def __init__(self, threshold_sigma: float = 2.0, min_size: int = 4):
        """Initialize detector.

        Args:
            threshold_sigma: Detection threshold in σ above mean
            min_size: Minimum blob size in grid cells
        """
        self.threshold_sigma = threshold_sigma
        self.min_size = min_size

    def detect(self, n_grid: np.ndarray, grid_x: np.ndarray,
               grid_y: np.ndarray) -> List[dict]:
        """Detect blobs in density field.

        Args:
            n_grid: Density field
            grid_x, grid_y: Grid coordinates

        Returns:
            List of blob dictionaries with position, size, amplitude
        """
        mean_n = np.mean(n_grid)
        std_n = np.std(n_grid)

        if std_n < 1e-10:
            return []

        # Threshold
        threshold = mean_n + self.threshold_sigma * std_n
        blob_mask = n_grid > threshold

        # Simple connected component labeling
        blobs = self._find_connected_components(blob_mask)

        # Extract blob properties
        blob_list = []
        X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')

        for blob_indices in blobs:
            if len(blob_indices[0]) < self.min_size:
                continue

            # Blob properties
            n_blob = n_grid[blob_indices]
            x_blob = X[blob_indices]
            y_blob = Y[blob_indices]

            # Centroid (weighted by density)
            total_n = np.sum(n_blob)
            x_centroid = np.sum(x_blob * n_blob) / total_n
            y_centroid = np.sum(y_blob * n_blob) / total_n

            # Size (effective radius)
            size = np.sqrt(len(blob_indices[0]) * (grid_x[1]-grid_x[0]) * (grid_y[1]-grid_y[0]) / np.pi)

            # Amplitude
            amplitude = np.max(n_blob) - mean_n

            blob_list.append({
                'x': x_centroid,
                'y': y_centroid,
                'size': size,
                'amplitude': amplitude,
                'total_density': total_n,
            })

        return blob_list

    def _find_connected_components(self, mask: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Simple flood-fill connected component detection.

        Args:
            mask: Boolean mask of above-threshold regions

        Returns:
            List of (row_indices, col_indices) for each component
        """
        visited = np.zeros_like(mask, dtype=bool)
        components = []

        nx, ny = mask.shape

        for i in range(nx):
            for j in range(ny):
                if mask[i, j] and not visited[i, j]:
                    # BFS to find connected region
                    component_i = []
                    component_j = []
                    queue = [(i, j)]
                    visited[i, j] = True

                    while queue:
                        ci, cj = queue.pop(0)
                        component_i.append(ci)
                        component_j.append(cj)

                        # Check 4 neighbors (with periodic BC)
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni = (ci + di) % nx
                            nj = (cj + dj) % ny

                            if mask[ni, nj] and not visited[ni, nj]:
                                visited[ni, nj] = True
                                queue.append((ni, nj))

                    components.append((np.array(component_i), np.array(component_j)))

        return components


def compute_radial_flux_profile(n_grid: np.ndarray, vx_grid: np.ndarray,
                                 axis: int = 0) -> np.ndarray:
    """Compute radial flux profile averaged over poloidal direction.

    Args:
        n_grid: Density field
        vx_grid: Radial velocity field
        axis: Axis to average over (0=x, 1=y)

    Returns:
        Flux profile Γ(r)
    """
    flux = n_grid * vx_grid
    return np.mean(flux, axis=1-axis)


def compute_power_spectrum_1d(signal: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 1D power spectrum of a time series.

    Args:
        signal: 1D time series
        dt: Time step

    Returns:
        (frequencies, power) arrays
    """
    n = len(signal)
    fft = np.fft.fft(signal)
    power = np.abs(fft[:n//2])**2 / n
    freq = np.fft.fftfreq(n, dt)[:n//2]

    return freq, power
