"""
Baryon class and grouping logic.
"""
import numpy as np
import enum
from quark import QuarkFlavor

class BaryonType(enum.IntEnum):
    """Types of baryons."""
    PROTON = 0
    NEUTRON = 1
    DELTA_PLUS_PLUS = 2  # Delta++ (uuu)
    DELTA_MINUS = 3      # Delta- (ddd)

class Baryon:
    """
    Represents a baryon composed of three quarks.
    
    Attributes:
        quark_indices (list): Indices of the three quarks in the quark array
        x, y (float): Center of mass position
        vx, vy (float): Center of mass velocity
        type (BaryonType): PROTON, NEUTRON, DELTA_PLUS_PLUS, or DELTA_MINUS
        mass (float): Total mass including binding energy
    """
    
    # Binding energy in MeV
    BINDING_ENERGY = -10.0
    
    def __init__(self, quark_indices, quark_array=None):
        """
        Initialize a baryon from three quark indices.
        
        Args:
            quark_indices: List of 3 indices pointing to quarks in the simulation
            quark_array: Optional numpy array of quarks to compute derived properties
        """
        self.quark_indices = quark_indices
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.type = None
        self.mass = 0.0
        
        # For stability: Store the initial baryon type and lock it
        self.type_locked = False
        self._type_history = []
        self.formation_time = 0
        
        if quark_array is not None:
            self.update_from_quarks(quark_array)
    
    def update_from_quarks(self, quark_array):
        """Update baryon properties based on its constituent quarks."""
        if len(self.quark_indices) != 3:
            raise ValueError("A baryon must have exactly 3 quarks")
            
        quarks = quark_array[self.quark_indices]
        
        # Count up quarks
        up_count = np.sum(quarks['flavor'] == QuarkFlavor.UP)
        
        # Calculate center of mass and velocity
        total_mass = np.sum(quarks['mass'])
        self.x = np.sum(quarks['x'] * quarks['mass']) / total_mass
        self.y = np.sum(quarks['y'] * quarks['mass']) / total_mass
        self.vx = np.sum(quarks['vx'] * quarks['mass']) / total_mass
        self.vy = np.sum(quarks['vy'] * quarks['mass']) / total_mass
        
        # Only update the baryon type if not locked yet
        if not self.type_locked:
            # Determine baryon type based on quark composition
            current_type = None
            if up_count == 3:
                current_type = BaryonType.DELTA_PLUS_PLUS
            elif up_count == 2:
                current_type = BaryonType.PROTON
            elif up_count == 1:
                current_type = BaryonType.NEUTRON
            elif up_count == 0:
                current_type = BaryonType.DELTA_MINUS
            else:
                raise ValueError(f"Invalid quark composition: {up_count} up quarks")
            
            # Add current type to history (keep last 10)
            self._type_history.append(current_type)
            if len(self._type_history) > 10:
                self._type_history.pop(0)
            
            # Use most frequent type in history
            if len(self._type_history) >= 5:
                # Lock type after several consistent determinations
                from collections import Counter
                type_counts = Counter(self._type_history)
                most_common_type = type_counts.most_common(1)[0][0]
                
                if type_counts[most_common_type] >= 4:  # At least 4 out of 10 are the same
                    self.type = most_common_type
                    self.type_locked = True  # Lock type permanently
            else:
                # Use current determination until we have history
                self.type = current_type
        
        # Total mass is sum of quark masses plus binding energy
        self.mass = total_mass + self.BINDING_ENERGY
    
    def is_color_neutral(self, quark_array):
        """Check if the baryon has all three colors (R,G,B)."""
        colors = quark_array[self.quark_indices]['color']
        return (0 in colors) and (1 in colors) and (2 in colors)
    
    def __repr__(self):
        """String representation of the baryon."""
        type_strs = {
            BaryonType.PROTON: "Proton",
            BaryonType.NEUTRON: "Neutron",
            BaryonType.DELTA_PLUS_PLUS: "Delta++",
            BaryonType.DELTA_MINUS: "Delta-"
        }
        type_str = type_strs.get(self.type, "Unknown")
        return f"{type_str}(pos=({self.x:.2f}, {self.y:.2f}), mass={self.mass:.2f})"

# Functions to create numpy structured arrays for baryons
def create_baryon_dtype():
    """Create numpy dtype for baryons to be used in CUDA."""
    return np.dtype([
        ('quark_indices', np.int32, 3),
        ('x', np.float32),
        ('y', np.float32),
        ('vx', np.float32),
        ('vy', np.float32),
        ('type', np.int32),
        ('mass', np.float32)
    ])

def baryons_to_array(baryons):
    """Convert a list of Baryon objects to a structured numpy array."""
    dtype = create_baryon_dtype()
    arr = np.zeros(len(baryons), dtype=dtype)
    
    for i, baryon in enumerate(baryons):
        arr[i]['quark_indices'] = baryon.quark_indices
        arr[i]['x'] = baryon.x
        arr[i]['y'] = baryon.y
        arr[i]['vx'] = baryon.vx
        arr[i]['vy'] = baryon.vy
        arr[i]['type'] = baryon.type
        arr[i]['mass'] = baryon.mass
    
    return arr
