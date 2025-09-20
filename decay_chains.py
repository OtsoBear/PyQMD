import random
import math
from particles import ParticleType, DecayType, Particle

# Conversion constants
YEAR = 31557600.0   # seconds in a year
DAY = 86400.0       # seconds in a day
HOUR = 3600.0       # seconds in an hour
MINUTE = 60.0       # seconds in a minute

# Comprehensive half-life database for isotopes
# Format: (Z, N): half_life_in_seconds
HALF_LIVES = {
    # Hydrogen isotopes
    (1, 0): float('inf'),        # H-1: stable
    (1, 1): float('inf'),        # H-2 (deuterium): stable
    (1, 2): 12.32 * YEAR,        # H-3 (tritium): 12.32 years
    (1, 3): 0.000000000139,      # H-4: 139 zeptoseconds
    
    # Helium isotopes
    (2, 1): float('inf'),        # He-3: stable
    (2, 2): float('inf'),        # He-4: stable
    (2, 3): 0.806,               # He-5: 806 ms
    (2, 4): 0.000000000119,      # He-6: 119 zeptoseconds
    (2, 6): 0.807,               # He-8: 807 ms
    
    # Lithium isotopes
    (3, 3): float('inf'),        # Li-6: stable
    (3, 4): float('inf'),        # Li-7: stable
    (3, 5): 0.839,               # Li-8: 839 ms
    (3, 6): 0.1783,              # Li-9: 178.3 ms
    
    # Beryllium isotopes
    (4, 3): 53.22 * DAY,         # Be-7: 53.22 days
    (4, 5): float('inf'),        # Be-9: stable
    (4, 6): 1.51e6 * YEAR,       # Be-10: 1.51 million years
    (4, 7): 13.81,               # Be-11: 13.81 seconds
    
    # Carbon isotopes
    (6, 6): float('inf'),        # C-12: stable
    (6, 7): float('inf'),        # C-13: stable
    (6, 8): 5730 * YEAR,         # C-14: 5,730 years
    
    # Nitrogen isotopes
    (7, 7): float('inf'),        # N-14: stable
    (7, 8): float('inf'),        # N-15: stable
    
    # Oxygen isotopes
    (8, 8): float('inf'),        # O-16: stable
    (8, 9): float('inf'),        # O-17: stable
    (8, 10): float('inf'),       # O-18: stable
    
    # Iron isotopes
    (26, 28): float('inf'),      # Fe-54: stable
    (26, 30): float('inf'),      # Fe-56: stable
    (26, 31): float('inf'),      # Fe-57: stable
    (26, 32): float('inf'),      # Fe-58: stable
    (26, 33): 44.5 * DAY,        # Fe-59: 44.5 days
    
    # Common medium-weight isotopes
    (27, 32): float('inf'),      # Co-59: stable
    (28, 30): float('inf'),      # Ni-58: stable
    (29, 34): float('inf'),      # Cu-63: stable
    (30, 34): float('inf'),      # Zn-64: stable
    (36, 48): float('inf'),      # Kr-84: stable
    (38, 50): float('inf'),      # Sr-88: stable
    (42, 56): float('inf'),      # Mo-98: stable
    
    # Silver isotopes
    (47, 60): float('inf'),      # Ag-107: stable
    (47, 62): float('inf'),      # Ag-109: stable
    (47, 58): 8.3 * 60,          # Ag-105: 41.29 days
    (47, 56): 5.1 * 60,          # Ag-103: 65.7 minutes
    (47, 63): 2.38 * 60,         # Ag-110m: 249.79 days
    (47, 64): 7.45 * DAY,        # Ag-111: 7.45 days
    (47, 59): 2.37 * MINUTE,     # Ag-106m: 8.28 days
    
    # Heavy stable isotopes
    (78, 117): float('inf'),     # Pt-195: stable
    (79, 118): float('inf'),     # Au-197: stable
    (80, 120): float('inf'),     # Hg-200: stable
    (81, 122): float('inf'),     # Tl-203: stable
    (82, 124): float('inf'),     # Pb-206: stable
    (82, 125): float('inf'),     # Pb-207: stable
    (82, 126): float('inf'),     # Pb-208: stable
    
    # Uranium series
    (92, 142): 2.455e5 * YEAR,   # U-234: 245,500 years
    (92, 143): 7.04e8 * YEAR,    # U-235: 704 million years
    (92, 146): 4.468e9 * YEAR,   # U-238: 4.468 billion years
    
    # Thorium series
    (90, 140): 7.54e4 * YEAR,    # Th-230: 75,400 years
    (90, 142): 1.405e10 * YEAR,  # Th-232: 14.05 billion years
    (90, 144): 24.10 * DAY,      # Th-234: 24.10 days
    
    # Neptunium and Plutonium
    (93, 144): 2.14e6 * YEAR,    # Np-237: 2.14 million years
    (94, 145): 6.56e3 * YEAR,    # Pu-239: 24,100 years
    (94, 146): 6.56e3 * YEAR,    # Pu-240: 6,560 years
    (94, 150): 8.00e7 * YEAR,    # Pu-244: 80 million years
    
    # Radium, Radon, etc.
    (88, 138): 1600 * YEAR,      # Ra-226: 1600 years
    (86, 136): 3.8235 * DAY,     # Rn-222: 3.8235 days
    (84, 124): 138.376 * DAY,    # Po-208: 2.898 years
    (84, 126): 138.376 * DAY,    # Po-210: 138.376 days
    
    # More decay chain elements
    (84, 130): 164.3e-6,         # Po-214: 164.3 microseconds
    (84, 134): 3.1 * MINUTE,     # Po-218: 3.1 minutes
    (83, 127): 5.015 * DAY,      # Bi-210: 5.015 days
    (83, 131): 19.9 * MINUTE,    # Bi-214: 19.9 minutes
    (82, 128): 22.3 * YEAR,      # Pb-210: 22.3 years
    (82, 132): 26.8 * MINUTE,    # Pb-214: 26.8 minutes
    
    # Add many more common medical and industrial isotopes
    (27, 33): 5.27 * YEAR,       # Co-60: 5.27 years
    (43, 56): 6.01 * HOUR,       # Tc-99m: 6.01 hours
    (53, 74): 8.02 * DAY,        # I-131: 8.02 days
    (55, 82): 30.17 * YEAR,      # Cs-137: 30.17 years
    (38, 52): 28.79 * YEAR,      # Sr-90: 28.79 years
}

# Enhanced list of decay chains
DECAY_CHAINS = {
    # Uranium-238 decay chain
    (92, 146): [(90, 144, DecayType.ALPHA, 1.0)],           # U-238 → Th-234
    (90, 144): [(91, 143, DecayType.BETA_MINUS, 1.0)],      # Th-234 → Pa-234
    (91, 143): [(92, 142, DecayType.BETA_MINUS, 1.0)],      # Pa-234 → U-234
    (92, 142): [(90, 140, DecayType.ALPHA, 1.0)],           # U-234 → Th-230
    (90, 140): [(88, 138, DecayType.ALPHA, 1.0)],           # Th-230 → Ra-226
    (88, 138): [(86, 136, DecayType.ALPHA, 1.0)],           # Ra-226 → Rn-222
    (86, 136): [(84, 134, DecayType.ALPHA, 1.0)],           # Rn-222 → Po-218
    (84, 134): [(82, 132, DecayType.ALPHA, 0.9998),         # Po-218 → Pb-214
                (83, 133, DecayType.BETA_PLUS, 0.0002)],    # Po-218 → At-218
    (82, 132): [(83, 131, DecayType.BETA_MINUS, 1.0)],      # Pb-214 → Bi-214
    (83, 131): [(84, 130, DecayType.BETA_MINUS, 0.9998),    # Bi-214 → Po-214
                (81, 133, DecayType.ALPHA, 0.0002)],        # Bi-214 → Tl-210
    (84, 130): [(82, 128, DecayType.ALPHA, 1.0)],           # Po-214 → Pb-210
    (82, 128): [(83, 127, DecayType.BETA_MINUS, 1.0)],      # Pb-210 → Bi-210
    (83, 127): [(84, 126, DecayType.BETA_MINUS, 1.0)],      # Bi-210 → Po-210
    (84, 126): [(82, 124, DecayType.ALPHA, 1.0)],           # Po-210 → Pb-206 (stable)
    
    # Uranium-235 decay chain
    (92, 143): [(90, 141, DecayType.ALPHA, 1.0)],           # U-235 → Th-231
    (90, 141): [(91, 140, DecayType.BETA_MINUS, 1.0)],      # Th-231 → Pa-231
    (91, 140): [(89, 138, DecayType.ALPHA, 1.0)],           # Pa-231 → Ac-227
    
    # Thorium-232 decay chain 
    (90, 142): [(88, 140, DecayType.ALPHA, 1.0)],           # Th-232 → Ra-228
    (88, 140): [(89, 139, DecayType.BETA_MINUS, 1.0)],      # Ra-228 → Ac-228
    (89, 139): [(90, 138, DecayType.BETA_MINUS, 1.0)],      # Ac-228 → Th-228
    
    # Common medical isotopes
    (43, 56): [(43, 56, DecayType.GAMMA, 0.99),            # Tc-99m → Tc-99 (gamma)
               (43, 56, DecayType.BETA_MINUS, 0.01)],      # Tc-99m → Ru-99 (beta)
    (53, 74): [(54, 73, DecayType.BETA_MINUS, 1.0)],       # I-131 → Xe-131
    
    # Fission products
    (55, 82): [(56, 81, DecayType.BETA_MINUS, 1.0)],       # Cs-137 → Ba-137m
    (38, 52): [(39, 51, DecayType.BETA_MINUS, 1.0)],       # Sr-90 → Y-90
    
    # Light elements
    (1, 2): [(2, 1, DecayType.BETA_MINUS, 1.0)],           # H-3 (tritium) → He-3
    (6, 8): [(7, 7, DecayType.BETA_MINUS, 1.0)],           # C-14 → N-14
}

def expand_decay_chain(z, n):
    """Add predicted decay types for isotopes not specifically in our decay chains"""
    key = (z, n)
    
    if key in DECAY_CHAINS:
        return
        
    # Determine natural decay mode based on nuclear physics principles
    # N/Z ratio determines predominant decay mode
    n_to_z = n / max(1, z)
    
    # Determine stability band approximation (very simplified)
    # More accurate would use binding energy calculations from semi-empirical mass formula
    if z < 20:
        # Light elements stability band is roughly N ≈ Z
        stable_ratio = 1.0
    else:
        # For heavier elements, stability band follows N ≈ 1.5Z - 0.4Z^(2/3)
        stable_ratio = 1.0 + 0.015 * z**1.3
        
    # Add decay modes based on distance from stability band
    if z > 83:  # Very heavy elements
        DECAY_CHAINS[key] = [(z-2, n-2, DecayType.ALPHA, 0.9)]
    elif n_to_z > stable_ratio + 0.15:  # Too neutron-rich
        DECAY_CHAINS[key] = [(z+1, n-1, DecayType.BETA_MINUS, 0.9)]
    elif n_to_z < stable_ratio - 0.15:  # Too proton-rich
        if z > 30:  # Heavier elements can undergo electron capture
            DECAY_CHAINS[key] = [(z-1, n+1, DecayType.BETA_PLUS, 0.9)]
        else:
            DECAY_CHAINS[key] = [(z-1, n, DecayType.PROTON_EMISSION, 0.9)]
    else:
        # Within stability band - likely stable or very long-lived
        DECAY_CHAINS[key] = [(z, n, DecayType.NONE, 1.0)]

def get_decay_product(z, n):
    """Get the decay product for a nucleus with Z protons and N neutrons."""
    key = (z, n)
    
    # If not in our decay chains, predict the behavior
    if key not in DECAY_CHAINS:
        expand_decay_chain(z, n)
    
    # Choose decay mode based on probabilities
    options = DECAY_CHAINS[key]
    
    if not options:  # Empty list or None
        return z, n, None, lambda x, y: []
    
    # Select a decay option based on probabilities
    if len(options) == 1:
        new_z, new_n, decay_type, prob = options[0]
    else:
        r = random.random()
        cumulative_prob = 0
        new_z, new_n, decay_type, prob = options[0]  # Default in case we don't match
        for option in options:
            nz, nn, dt, p = option
            cumulative_prob += p
            if r <= cumulative_prob:
                new_z, new_n, decay_type, prob = option
                break
    
    if decay_type == DecayType.NONE:
        return z, n, None, lambda x, y: []
    
    # Return appropriate particle creation function
    particle_creators = {
        DecayType.ALPHA: create_alpha,
        DecayType.BETA_MINUS: create_beta_minus,
        DecayType.BETA_PLUS: create_beta_plus,
        DecayType.GAMMA: create_gamma,
        DecayType.NEUTRON_EMISSION: create_neutron,
        DecayType.PROTON_EMISSION: create_proton,
        DecayType.SPONTANEOUS_FISSION: create_fission,
    }
    
    return new_z, new_n, decay_type, particle_creators.get(decay_type, lambda x, y: [])

def get_half_life(z, n):
    """Get the half-life for a nucleus with Z protons and N neutrons."""
    key = (z, n)
    
    # Debug print for silver isotopes
    if z == 47:
        import logging
        logger = logging.getLogger("NuclearSim")
        logger.info(f"Looking up half-life for Silver isotope: Z={z}, N={n}, A={z+n}")
    
    if key in HALF_LIVES:
        if z == 47:  # More debugging for silver
            import logging
            logger = logging.getLogger("NuclearSim")
            logger.info(f"Found Silver isotope in database: half-life = {HALF_LIVES[key]}")
        return HALF_LIVES[key]
    
    # If not in database, estimate half-life using systematic trends
    # Using the semi-empirical mass formula concepts
    
    # Get total nucleon count
    a = z + n
    
    # Stability region approximation
    if z <= 20:
        optimal_n = z  # For light elements, N=Z is often stable
    else:
        # For heavier elements, stability follows a curve
        optimal_n = z * (1.0 + 0.0075 * z**(2/3))
        
    # Calculate deviation from stability
    n_to_z = n / max(1, z)
    if z < 20:
        stable_ratio = 1.0
    else:
        stable_ratio = 1.0 + 0.015 * z**1.3
        
    deviation = abs(n_to_z - stable_ratio)
    
    # Magic numbers in nuclear physics increase stability
    magic_numbers = [2, 8, 20, 28, 50, 82, 126]
    magic_bonus = 0
    if z in magic_numbers:
        magic_bonus += 0.2
    if n in magic_numbers:
        magic_bonus += 0.2
    
    # Even-even nuclei are more stable than odd-odd
    parity_factor = 1.0
    if z % 2 == 0 and n % 2 == 0:  # Even-even
        parity_factor = 0.5  # More stable
    elif z % 2 == 1 and n % 2 == 1:  # Odd-odd
        parity_factor = 2.0  # Less stable
    
    # Calculate stability factor (0-1, higher is more stable)
    stability = max(0, 1.0 - deviation * 2.0 - parity_factor * 0.1 + magic_bonus)
    
    # Very heavy elements have alpha decay regardless of stability
    if z > 83:
        stability *= 0.5
    
    # Convert stability to half-life (logarithmic scale)
    if stability >= 0.95:
        return float('inf')  # Essentially stable
    elif stability >= 0.85:  # Very stable
        return 10 ** (random.uniform(15, 17)) * YEAR  # ~10^15-10^17 years
    elif stability >= 0.75:
        return 10 ** (random.uniform(9, 14)) * YEAR   # ~10^9-10^14 years
    elif stability >= 0.65:
        return 10 ** (random.uniform(6, 9)) * YEAR    # ~10^6-10^9 years
    elif stability >= 0.50:
        return 10 ** (random.uniform(3, 6)) * YEAR    # ~10^3-10^6 years
    elif stability >= 0.40:
        return 10 ** (random.uniform(0, 3)) * YEAR    # ~1-1000 years
    elif stability >= 0.30:
        return 10 ** (random.uniform(0, 2)) * DAY     # ~1-100 days
    elif stability >= 0.20:
        return 10 ** (random.uniform(0, 4)) * HOUR    # ~1-10000 hours
    elif stability >= 0.10:
        return 10 ** (random.uniform(-1, 3)) * MINUTE # ~0.1-1000 minutes
    else:
        return 10 ** (random.uniform(-6, 1))          # ~1μs-10s

# Particle creation functions
def create_alpha(x, y):
    angle = random.uniform(0, 2 * math.pi)
    speed = 100
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)
    return [Particle(x, y, ParticleType.ALPHA, vx, vy)]

def create_beta_minus(x, y):
    angle = random.uniform(0, 2 * math.pi)
    speed = 150
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)
    return [Particle(x, y, ParticleType.ELECTRON, vx, vy)]

def create_beta_plus(x, y):
    angle = random.uniform(0, 2 * math.pi)
    speed = 150
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)
    return [Particle(x, y, ParticleType.POSITRON, vx, vy)]

def create_gamma(x, y):
    angle = random.uniform(0, 2 * math.pi)
    speed = 200
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)
    return [Particle(x, y, ParticleType.GAMMA, vx, vy)]

def create_neutron(x, y):
    angle = random.uniform(0, 2 * math.pi)
    speed = 60
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)
    return [Particle(x, y, ParticleType.NEUTRON, vx, vy)]

def create_proton(x, y):
    angle = random.uniform(0, 2 * math.pi)
    speed = 50
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)
    return [Particle(x, y, ParticleType.PROTON, vx, vy)]

def create_fission(x, y):
    # Simplified fission - just creates multiple particles
    particles = []
    # Create 2-3 "fragments"
    for _ in range(random.randint(2, 3)):
        angle = random.uniform(0, 2 * math.pi)
        speed = 80 + random.random() * 40
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        
        # Randomly create either alpha particle or neutron
        if random.random() < 0.7:
            particles.append(Particle(x, y, ParticleType.ALPHA, vx, vy))
        else:
            particles.append(Particle(x, y, ParticleType.NEUTRON, vx, vy))
    return particles

class Nucleus:
    def __init__(self, protons, neutrons, x, y):
        self.protons = protons
        self.neutrons = neutrons
        self.x = x
        self.y = y
        self.stability = get_half_life(protons, neutrons)
        
        # Ensure any nucleus methods defined here match with the particle.py implementation
        
    def should_decay(self, dt):
        """Determine if nucleus should decay based on half-life."""
        # Stable nuclei don't decay
        if self.stability == float('inf'):
            return False
            
        # Modified calculation that correctly handles small dt values and extremely large timescales
        # For very large time scales, this improves precision
        if dt > self.stability * 0.01:
            # For large time steps relative to half-life, use direct probability
            probability = 1.0 - 0.5 ** (dt / self.stability)
        else:
            # For small time steps, use approximation to avoid precision issues
            # P ≈ λ·dt where λ = ln(2)/T½
            decay_constant = 0.693 / self.stability  # ln(2)/T½
            probability = decay_constant * dt
            
        # Ensure probability is in valid range [0,1]
        probability = max(0.0, min(1.0, probability))
        
        # Return True if we should decay
        return random.random() < probability
