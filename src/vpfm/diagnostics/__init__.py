"""Diagnostics and analysis utilities."""

from .diagnostics import (
    compute_diagnostics,
    compute_energy_spectral,
    find_vortex_centroid,
    find_vortex_peak,
    compute_spectrum,
    compare_structure_preservation,
)
from .spectral import energy_spectrum, density_flux_spectrum, fit_power_law
from .flow_topology import compute_weiss_field, partition_flow, compute_residence_times
from .flux_diagnostics import VirtualProbe, BlobDetector, FluxStatistics

__all__ = [
    "compute_diagnostics",
    "compute_energy_spectral",
    "find_vortex_centroid",
    "find_vortex_peak",
    "compute_spectrum",
    "compare_structure_preservation",
    "energy_spectrum",
    "density_flux_spectrum",
    "fit_power_law",
    "compute_weiss_field",
    "partition_flow",
    "compute_residence_times",
    "VirtualProbe",
    "BlobDetector",
    "FluxStatistics",
]
