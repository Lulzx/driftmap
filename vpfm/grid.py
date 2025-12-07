"""Grid data structure for VPFM-Plasma."""

import numpy as np


class Grid:
    """Eulerian grid for velocity/potential reconstruction.

    Attributes:
        nx, ny: Number of grid points in x and y directions
        Lx, Ly: Domain size in x and y directions
        dx, dy: Grid spacing
        q: Potential vorticity field (nx, ny)
        phi: Electrostatic potential field (nx, ny)
        vx, vy: E×B velocity components (nx, ny)
    """

    def __init__(self, nx: int, ny: int, Lx: float, Ly: float):
        """Initialize grid.

        Args:
            nx: Number of grid points in x direction
            ny: Number of grid points in y direction
            Lx: Domain length in x direction
            Ly: Domain length in y direction
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / nx
        self.dy = Ly / ny

        # Field arrays (cell-centered)
        self.q = np.zeros((nx, ny))      # Potential vorticity
        self.phi = np.zeros((nx, ny))    # Electrostatic potential
        self.vx = np.zeros((nx, ny))     # E×B velocity x component
        self.vy = np.zeros((nx, ny))     # E×B velocity y component

        # Velocity gradients (for Jacobian evolution)
        self.dvx_dx = np.zeros((nx, ny))
        self.dvx_dy = np.zeros((nx, ny))
        self.dvy_dx = np.zeros((nx, ny))
        self.dvy_dy = np.zeros((nx, ny))

        # Grid coordinates (cell centers)
        self.x = np.linspace(self.dx/2, Lx - self.dx/2, nx)
        self.y = np.linspace(self.dy/2, Ly - self.dy/2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

    def reset_fields(self):
        """Reset all field arrays to zero."""
        self.q.fill(0.0)
        self.phi.fill(0.0)
        self.vx.fill(0.0)
        self.vy.fill(0.0)
        self.dvx_dx.fill(0.0)
        self.dvx_dy.fill(0.0)
        self.dvy_dx.fill(0.0)
        self.dvy_dy.fill(0.0)

    def wrap_coordinates(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """Apply periodic boundary conditions to coordinates.

        Args:
            x: x-coordinates (may be outside domain)
            y: y-coordinates (may be outside domain)

        Returns:
            Wrapped (x, y) coordinates within [0, Lx) × [0, Ly)
        """
        x_wrapped = x % self.Lx
        y_wrapped = y % self.Ly
        return x_wrapped, y_wrapped

    def get_cell_indices(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """Get cell indices containing given coordinates.

        Args:
            x: x-coordinates
            y: y-coordinates

        Returns:
            (i, j) integer cell indices
        """
        i = np.floor(x / self.dx).astype(int) % self.nx
        j = np.floor(y / self.dy).astype(int) % self.ny
        return i, j

    def get_bilinear_weights(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """Compute bilinear interpolation weights.

        Args:
            x: x-coordinates
            y: y-coordinates

        Returns:
            (i, j, fx, fy) where i,j are cell indices and fx,fy are fractional positions
        """
        # Normalize to cell coordinates
        x_cell = x / self.dx
        y_cell = y / self.dy

        # Get integer cell indices
        i = np.floor(x_cell).astype(int) % self.nx
        j = np.floor(y_cell).astype(int) % self.ny

        # Fractional position within cell [0, 1)
        fx = x_cell - np.floor(x_cell)
        fy = y_cell - np.floor(y_cell)

        return i, j, fx, fy
