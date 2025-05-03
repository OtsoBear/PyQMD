"""
Nucleus class and related functions.
"""
import numpy as np
from baryon import BaryonType

class Nucleus:
    """
    Represents an atomic nucleus composed of baryons (protons and neutrons).
    
    Attributes:
        baryon_indices (list): Indices of baryons in the baryon array
        binding_energy (float): Nuclear binding energy in MeV
        total_mass (float): Total mass including binding energy
    """
    
    # Semi-empirical mass formula parameters (in MeV)
    VOLUME_COEFF = 15.8  # Volume term
    SURFACE_COEFF = 18.3  # Surface term
    COULOMB_COEFF = 0.714  # Coulomb term
    ASYMMETRY_COEFF = 23.2  # Asymmetry term
    PAIRING_COEFF = 33.5  # Pairing term
    
    def __init__(self, baryon_indices, baryon_array=None):
        """
        Initialize a nucleus from baryon indices.
        
        Args:
            baryon_indices: List of indices pointing to baryons in the simulation
            baryon_array: Optional numpy array of baryons to compute derived properties
        """
        self.baryon_indices = baryon_indices
        self.binding_energy = 0.0
        self.total_mass = 0.0
        
        if baryon_array is not None:
            self.update_from_baryons(baryon_array)
    
    def update_from_baryons(self, baryon_array):
        """Update nucleus properties based on its constituent baryons."""
        baryons = baryon_array[self.baryon_indices]
        
        # Count protons and neutrons
        proton_count = np.sum(baryons['type'] == BaryonType.PROTON)
        neutron_count = np.sum(baryons['type'] == BaryonType.NEUTRON)
        
        # Calculate binding energy using semi-empirical mass formula
        self.binding_energy = self._calculate_binding_energy(proton_count, neutron_count)
        
        # Calculate total mass (sum of baryon masses plus binding energy)
        baryon_mass_sum = np.sum(baryons['mass'])
        self.total_mass = baryon_mass_sum + self.binding_energy
        
    def _calculate_binding_energy(self, Z, N):
        """
        Calculate nuclear binding energy using the semi-empirical mass formula.
        
        Args:
            Z: Number of protons
            N: Number of neutrons
            
        Returns:
            Binding energy in MeV
        """
        A = Z + N  # Mass number
        
        if A == 0:
            return 0.0
            
        # Calculate each term of the semi-empirical mass formula
        volume_term = -self.VOLUME_COEFF * A
        surface_term = self.SURFACE_COEFF * A**(2/3)
        coulomb_term = self.COULOMB_COEFF * Z * (Z - 1) / A**(1/3)
        asymmetry_term = self.ASYMMETRY_COEFF * (N - Z)**2 / A
        
        # Pairing term
        if Z % 2 == 0 and N % 2 == 0:  # even-even
            pairing_term = -self.PAIRING_COEFF / A**(1/2)
        elif Z % 2 == 1 and N % 2 == 1:  # odd-odd
            pairing_term = self.PAIRING_COEFF / A**(1/2)
        else:  # even-odd or odd-even
            pairing_term = 0
        
        # Total binding energy
        return volume_term + surface_term + coulomb_term + asymmetry_term + pairing_term
    
    def is_stable(self):
        """Check if the nucleus is stable based on binding energy."""
        return self.binding_energy < 0  # Negative binding energy means bound state
    
    @property
    def proton_count(self):
        """Get the number of protons in the nucleus."""
        if not hasattr(self, '_proton_count'):
            return None
        return self._proton_count
    
    @property
    def neutron_count(self):
        """Get the number of neutrons in the nucleus."""
        if not hasattr(self, '_neutron_count'):
            return None
        return self._neutron_count
    
    def __repr__(self):
        """String representation of the nucleus."""
        if hasattr(self, '_proton_count') and hasattr(self, '_neutron_count'):
            return f"Nucleus(Z={self._proton_count}, N={self._neutron_count}, BE={self.binding_energy:.2f} MeV)"
        else:
            return f"Nucleus({len(self.baryon_indices)} baryons, BE={self.binding_energy:.2f} MeV)"

# Functions to create numpy structured arrays for nuclei
def create_nucleus_dtype(max_baryons=256):
    """Create numpy dtype for nuclei to be used in CUDA."""
    return np.dtype([
        ('baryon_indices', np.int32, max_baryons),
        ('baryon_count', np.int32),
        ('binding_energy', np.float32),
        ('total_mass', np.float32)
    ])

def nuclei_to_array(nuclei, max_baryons=256):
    """Convert a list of Nucleus objects to a structured numpy array."""
    dtype = create_nucleus_dtype(max_baryons)
    arr = np.zeros(len(nuclei), dtype=dtype)
    
    for i, nucleus in enumerate(nuclei):
        # Fill baryon indices, pad with -1
        baryon_count = len(nucleus.baryon_indices)
        arr[i]['baryon_count'] = baryon_count
        arr[i]['baryon_indices'][:baryon_count] = nucleus.baryon_indices
        if baryon_count < max_baryons:
            arr[i]['baryon_indices'][baryon_count:] = -1
        
        arr[i]['binding_energy'] = nucleus.binding_energy
        arr[i]['total_mass'] = nucleus.total_mass
    
    return arr
