"""
Quark class and associated helper functions.
"""
import numpy as np
import enum

class QuarkFlavor(enum.IntEnum):
    """Quark flavor types."""
    UP = 0    # u quark
    DOWN = 1  # d quark

class QuarkColor(enum.IntEnum):
    """Quark color charge types."""
    RED = 0
    GREEN = 1
    BLUE = 2

class Quark:
    """
    Represents a single quark in the simulation.
    
    Attributes:
        x, y (float): Position in 2D space
        vx, vy (float): Velocity in 2D space
        flavor (QuarkFlavor): UP or DOWN
        color (QuarkColor): RED, GREEN, or BLUE
        mass (float): Mass in MeV/c²
        charge (float): Electric charge (+2/3 for up, -1/3 for down)
        is_bound (bool): Whether this quark is bound in a baryon
    """
    
    # Constants for quark properties in MeV/c²
    UP_MASS = 1.5    # Reduced from 2.2
    DOWN_MASS = 3.0  # Reduced from 4.7
    UP_CHARGE = 2/3
    DOWN_CHARGE = -1/3
    
    def __init__(self, x, y, vx, vy, flavor, color):
        """Initialize a quark with position, velocity, flavor, and color."""
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.flavor = flavor
        self.color = color
        self.is_bound = False  # Track if quark is bound in a baryon
        
        # Set mass and charge based on flavor
        if flavor == QuarkFlavor.UP:
            self.mass = self.UP_MASS
            self.charge = self.UP_CHARGE
        else:
            self.mass = self.DOWN_MASS
            self.charge = self.DOWN_CHARGE

    def __repr__(self):
        """String representation of the quark."""
        flavor_str = 'u' if self.flavor == QuarkFlavor.UP else 'd'
        color_str = ['R', 'G', 'B'][self.color]
        return f"Quark({flavor_str}{color_str}, pos=({self.x:.2f}, {self.y:.2f}))"

# Functions to create numpy structured arrays for GPU use
def create_quark_dtype():
    """Create numpy dtype for quarks to be used in CUDA."""
    return np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('vx', np.float32),
        ('vy', np.float32),
        ('flavor', np.int32),
        ('color', np.int32),
        ('mass', np.float32),
        ('charge', np.float32),
        ('is_bound', np.int32)  # Add a flag to track bound quarks
    ])

def quarks_to_array(quarks):
    """Convert a list of Quark objects to a structured numpy array."""
    dtype = create_quark_dtype()
    arr = np.zeros(len(quarks), dtype=dtype)
    
    for i, quark in enumerate(quarks):
        arr[i]['x'] = quark.x
        arr[i]['y'] = quark.y
        arr[i]['vx'] = quark.vx
        arr[i]['vy'] = quark.vy
        arr[i]['flavor'] = int(quark.flavor)
        arr[i]['color'] = int(quark.color)
        arr[i]['mass'] = quark.mass
        arr[i]['charge'] = quark.charge
        arr[i]['is_bound'] = int(quark.is_bound)
    
    return arr
