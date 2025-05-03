"""
Electron class for simulating leptons in the QMD simulation.
"""
import numpy as np
from physics import electron_mass

class Electron:
    """
    Represents an electron in the simulation.
    
    Attributes:
        x, y (float): Position in 2D space
        vx, vy (float): Velocity in 2D space
        mass (float): Mass in MeV/c²
        charge (float): Electric charge (-1)
    """
    
    def __init__(self, x, y, vx, vy):
        """Initialize an electron with position and velocity."""
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = electron_mass  # Much lighter than quarks
        self.charge = -1.0  # Negative charge
    
    def update(self, dt, force_x, force_y):
        """Update position and velocity based on forces."""
        # Calculate acceleration (F = ma -> a = F/m)
        ax = force_x / self.mass
        ay = force_y / self.mass
        
        # Update velocity (v = v + at)
        self.vx += ax * dt
        self.vy += ay * dt
        
        # Update position (x = x + vt)
        self.x += self.vx * dt
        self.y += self.vy * dt
    
    def __repr__(self):
        """String representation of the electron."""
        return f"Electron(pos=({self.x:.2f}, {self.y:.2f}), v=({self.vx:.2f}, {self.vy:.2f}))"

def electrons_to_array(electrons):
    """Convert a list of Electron objects to a structured numpy array."""
    dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('vx', np.float32),
        ('vy', np.float32),
        ('mass', np.float32),
        ('charge', np.float32)
    ])
    
    arr = np.zeros(len(electrons), dtype=dtype)
    
    for i, electron in enumerate(electrons):
        arr[i]['x'] = electron.x
        arr[i]['y'] = electron.y
        arr[i]['vx'] = electron.vx
        arr[i]['vy'] = electron.vy
        arr[i]['mass'] = electron.mass
        arr[i]['charge'] = electron.charge
    
    return arr
