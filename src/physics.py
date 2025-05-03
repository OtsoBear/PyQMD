"""
Physical constants, conversions, and CPU utilities for the simulation.
"""
import numpy as np

# Conversion factors
fm_to_unit = 1.0  # 1 simulation unit = 1 femtometer

# Physical constants in simulation units
c = 299792458  # speed of light in m/s
c2 = c * c     # c² (speed of light squared)
hbar = 197.3   # MeV·fm (ħc in natural units)
dt = 1e-20     # time step in seconds - 100x larger than default

# Force constants - MASSIVELY increased for guaranteed visible movement
strong_const = 5000.0   # Strong force constant (10x stronger)
coulomb_const = 500.0   # Electromagnetic coupling (50x stronger)
gravity_const = 5.0     # Gravitational constant (much stronger)

# Interaction parameters
binding_thresh = 4.0    # Increased binding threshold
weak_prob = 0.005      # Decay probability per time step

# Quark masses - EXTREMELY reduced for dramatic movement
u_mass = 0.01  # Ultra-light up quark (200x lighter)
d_mass = 0.02  # Ultra-light down quark (200x lighter)

# Electron properties
electron_mass = 0.001  # Ultra-light electron (500x lighter)

# Particle masses in MeV/c²
proton_mass = 0.1  # Extremely reduced for testing (10000x lighter)
neutron_mass = 0.1  # Extremely reduced for testing

# Debug flags
DEBUG_FORCES = True    # Print force information
DEBUG_MOVEMENT = True  # Track and print movement statistics

def distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def kinetic_energy(mass, vx, vy):
    """Calculate kinetic energy in MeV."""
    v_squared = vx**2 + vy**2
    # Non-relativistic KE = 0.5 * m * v²
    return 0.5 * mass * v_squared

def relativistic_kinetic_energy(mass, vx, vy):
    """Calculate relativistic kinetic energy in MeV."""
    v_squared = vx**2 + vy**2
    gamma = 1.0 / np.sqrt(1.0 - v_squared / c**2)
    # Relativistic KE = (γ-1) * m * c²
    return (gamma - 1.0) * mass * c2

def color_is_complementary(c1, c2, c3):
    """Check if three colors form a complementary set (R,G,B)."""
    return set([c1, c2, c3]) == {0, 1, 2}

def decay_probability(half_life):
    """
    Convert half-life to decay probability per time step.
    
    Args:
        half_life: Half-life in seconds
    
    Returns:
        Probability of decay in one simulation time step
    """
    decay_constant = np.log(2) / half_life
    return 1.0 - np.exp(-decay_constant * dt)

def get_reaction_q_value(reactant_masses, product_masses):
    """
    Calculate Q-value for a nuclear reaction.
    
    Args:
        reactant_masses: List of masses of reactants in MeV/c²
        product_masses: List of masses of products in MeV/c²
    
    Returns:
        Q-value in MeV (positive for exothermic reactions)
    """
    return sum(reactant_masses) - sum(product_masses)

def are_color_neutral(quarks):
    """
    Check if a set of quarks forms a color-neutral combination.
    
    For baryons: must have one of each color (R,G,B)
    For mesons: must have a color and its anti-color
    """
    if len(quarks) == 3:  # Baryon
        colors = [q.color for q in quarks]
        return set(colors) == {0, 1, 2}
    elif len(quarks) == 2:  # Meson
        return False  # Mesons not implemented in this version
    return False
