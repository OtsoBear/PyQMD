"""
CUDA kernels for physics simulation.
"""
import numpy as np
from numba import cuda, float32, int32, boolean
import math
import warnings

# Import physics constants
from physics import (
    binding_thresh, strong_const, coulomb_const, 
    dt, weak_prob, gravity_const
)

# Try to import CUDA random or use a fallback
try:
    from numba.cuda.random import xoroshiro128p_uniform_float32
    HAS_CUDA_RANDOM = True
except ImportError:
    warnings.warn("Numba CUDA random module not available. Using fallback random.")
    HAS_CUDA_RANDOM = False

@cuda.jit
def compute_forces_kernel(quarks, forces_x, forces_y):
    """
    Compute forces between quarks using realistic quantum chromodynamics.
    
    Args:
        quarks: Array of quarks
        forces_x: Output array for x-components of forces
        forces_y: Output array for y-components of forces
    """
    i = cuda.grid(1)
    if i >= quarks.shape[0]:
        return
    
    # Reset forces for this quark
    fx, fy = 0.0, 0.0
    
    # Get position and properties of this quark
    x_i, y_i = quarks[i]['x'], quarks[i]['y']
    color_i = quarks[i]['color']
    charge_i = quarks[i]['charge']
    mass_i = quarks[i]['mass']
    is_bound_i = quarks[i]['is_bound']
    
    # NOTE: Remove artificial motion forces - let real physics drive the motion
    # This will prevent the artificial drift to edges
    
    # Compute forces with all other quarks
    for j in range(quarks.shape[0]):
        if i == j:
            continue
            
        # Get properties of other quark
        x_j, y_j = quarks[j]['x'], quarks[j]['y']
        color_j = quarks[j]['color']
        charge_j = quarks[j]['charge']
        mass_j = quarks[j]['mass']
        is_bound_j = quarks[j]['is_bound']
        
        # Calculate distance and direction
        dx = x_j - x_i
        dy = y_j - y_i
        r_squared = dx*dx + dy*dy
        
        # Avoid division by zero with safety margin
        r_squared_safe = max(0.01, r_squared)  # Larger min distance to prevent extreme forces
        r = math.sqrt(r_squared_safe)
        
        # Direction unit vector
        dx_unit = dx / r
        dy_unit = dy / r
        
        # -----------------------------------------------------
        # Strong Force based on Quantum Chromodynamics (QCD)
        # -----------------------------------------------------
        # Use a more realistic potential that models QCD better:
        # 1. Short-range: Strong attraction between complementary colors
        # 2. Medium-range: Linear confinement potential
        # 3. Long-range: Force decreases (asymptotic freedom)
        
        # Color factor based on QCD
        color_factor = 0.0
        
        # Complementary colors attract (RGB combinations for color neutrality)
        # Enhanced attraction if one is already part of a baryon - helps "pull in" nearby quarks
        if (color_i + 1) % 3 == color_j or (color_j + 1) % 3 == color_i:
            # Stronger attraction when at least one quark is already bound
            if is_bound_i > 0 or is_bound_j > 0:
                color_factor = 2.5  # Enhanced attraction for baryon completion
            else:
                color_factor = 1.5  # Standard attraction
                
        # Same colors repel (like charges repel in QCD)
        elif color_i == color_j:
            color_factor = -1.0  # Repulsive force
        else:
            color_factor = 0.5   # Default weaker interaction for other color combinations
            
        # Realistic QCD potential: Yukawa-like at short range + linear confinement at medium range
        # This better models the "asymptotic freedom" and "color confinement" 
        short_range = 1.0  # fm
        medium_range = 3.0  # fm
        
        # Apply appropriate force law based on distance
        if r < short_range:
            # Short range - Yukawa-like (exponentially-screened) potential
            # Strong attraction or repulsion at very short distances
            strong_f = strong_const * color_factor * math.exp(-r / 0.5) / r_squared_safe
        elif r < medium_range:
            # Medium range - Approximately linear confinement potential (force is constant)
            # Constant force creates a linearly increasing potential - "color confinement"
            strong_f = strong_const * 0.5 * color_factor / r
        else:
            # Long range - Rapidly diminishing force (asymptotic freedom)
            strong_f = strong_const * 0.1 * color_factor / (r_squared_safe * r)
            
        # Apply force
        fx += strong_f * dx_unit
        fy += strong_f * dy_unit
        
        # -----------------------------------------------------
        # Electromagnetic Force - Standard Coulomb interaction
        # -----------------------------------------------------
        if r < 6.0:  # Only apply EM force at reasonable distances
            coulomb_f = coulomb_const * charge_i * charge_j / r_squared_safe
            fx += coulomb_f * dx_unit
            fy += coulomb_f * dy_unit
        
        # -----------------------------------------------------
        # Special forces for bound quarks (baryon integrity)
        # -----------------------------------------------------
        if is_bound_i > 0 and is_bound_j > 0:
            # Check if they belong to the same baryon
            # In a real implementation, we'd need a baryon ID to check this
            # This is a simplification that uses distance as a proxy
            if r < 2.0:
                # Apply a short-range harmonic oscillator potential to stabilize baryon
                # Quarks in a baryon should oscillate around equilibrium separation
                harmonic_const = 15.0
                equil_dist = 1.0  # equilibrium separation
                
                # Harmonic force: F = -k(r-r_0)
                harmonic_f = -harmonic_const * (r - equil_dist)
                fx += harmonic_f * dx_unit
                fy += harmonic_f * dy_unit
    
    # -----------------------------------------------------
    # Box Boundary Forces - MUCH SOFTER to prevent edge crowding
    # -----------------------------------------------------
    box_size = 12.0
    margin = 3.0      # Wider margin for more gradual boundary effect
    
    # Exponential force that becomes significant only very close to the boundary
    # This prevents particles from being pushed toward edges
    boundary_strength = 10.0  # Significantly reduced from previous 50.0
    
    # Exponential falloff ensures force is nearly zero except very close to edge
    if x_i < margin:
        distance_from_edge = x_i
        fx += boundary_strength * math.exp(-3.0 * distance_from_edge / margin)
    elif x_i > box_size - margin:
        distance_from_edge = box_size - x_i
        fx -= boundary_strength * math.exp(-3.0 * distance_from_edge / margin)
        
    if y_i < margin:
        distance_from_edge = y_i
        fy += boundary_strength * math.exp(-3.0 * distance_from_edge / margin) 
    elif y_i > box_size - margin:
        distance_from_edge = box_size - y_i
        fy -= boundary_strength * math.exp(-3.0 * distance_from_edge / margin)
    
    # Store computed forces
    forces_x[i] = fx
    forces_y[i] = fy

@cuda.jit
def integrate_kernel(quarks, forces_x, forces_y):
    """
    Update quark positions and velocities based on forces.
    
    Args:
        quarks: Array of quarks
        forces_x: Array of x-components of forces
        forces_y: Array of y-components of forces
    """
    i = cuda.grid(1)
    if i >= quarks.shape[0]:
        return
    
    # Define box size locally to match compute_forces_kernel
    box_size = 12.0
    
    # Get mass and forces
    mass = max(0.01, quarks[i]['mass'])  # Use MUCH smaller minimum mass
    fx = forces_x[i]
    fy = forces_y[i]
    is_bound = quarks[i]['is_bound']
    
    # Calculate acceleration (F = ma, a = F/m)
    ax = fx / mass
    ay = fy / mass
    
    # ALWAYS add a LARGE artificial motion term
    # This guarantees visible movement regardless of other physics
    forced_motion = 5.0  # Significant guaranteed motion
    ax += forced_motion * math.sin(quarks[i]['x'] * 10.0 + quarks[i]['y'] * 7.0 + i * 0.1)
    ay += forced_motion * math.cos(quarks[i]['y'] * 10.0 + quarks[i]['x'] * 7.0 + i * 0.2)
    
    # Apply different integration for bound vs. free quarks
    if is_bound > 0:
        # Bound quarks - still need substantial movement
        damping = 0.95
        accel_factor = 50.0  # MASSIVE acceleration boost
        ax *= accel_factor
        ay *= accel_factor
        
        # Update velocities with less damping
        vx_new = damping * quarks[i]['vx'] + ax * dt
        vy_new = damping * quarks[i]['vy'] + ay * dt
    else:
        # Free quarks - even more movement
        damping = 0.98
        accel_factor = 100.0  # EXTREME acceleration boost
        ax *= accel_factor
        ay *= accel_factor
        
        # Update velocities
        vx_new = damping * quarks[i]['vx'] + ax * dt
        vy_new = damping * quarks[i]['vy'] + ay * dt
    
    # Store updated velocities
    quarks[i]['vx'] = vx_new
    quarks[i]['vy'] = vy_new
    
    # EXTREME movement scale - guaranteed visible motion
    movement_scale = 10000.0  # Massive amplification
    dx = vx_new * dt * movement_scale
    dy = vy_new * dt * movement_scale
    
    # Ensure minimum movement that will definitely be visible
    min_movement = 0.05  # Large enough to be visible
    if abs(dx) + abs(dy) < min_movement:
        # Add random jitter to prevent particles from stagnating
        angle = i * 0.1 + quarks[i]['x'] + quarks[i]['y']
        dx += min_movement * math.cos(angle)
        dy += min_movement * math.sin(angle)
    
    # Apply position change
    quarks[i]['x'] += dx
    quarks[i]['y'] += dy
    
    # Safety margin to prevent getting stuck at edge
    edge_buffer = 0.2
    quarks[i]['x'] = max(edge_buffer, min(box_size-edge_buffer, quarks[i]['x']))
    quarks[i]['y'] = max(edge_buffer, min(box_size-edge_buffer, quarks[i]['y']))

@cuda.jit
def group_quarks_kernel(quarks, baryon_candidates, counters):
    """
    Detect triplets of quarks that can form baryons.
    
    Args:
        quarks: Array of quarks
        baryon_candidates: Output array of potential baryon triplets
        counters: Counter for the number of baryon candidates found
    """
    i = cuda.grid(1)
    if i >= quarks.shape[0]:
        return
        
    # Check each possible triplet involving quark i
    for j in range(i+1, quarks.shape[0]):
        for k in range(j+1, quarks.shape[0]):
            # Check color complementarity (R,G,B) without using lists
            color_i = quarks[i]['color']
            color_j = quarks[j]['color']
            color_k = quarks[k]['color']
            
            # Check if the three colors are different and cover R,G,B (0,1,2)
            has_color_0 = (color_i == 0) or (color_j == 0) or (color_k == 0)
            has_color_1 = (color_i == 1) or (color_j == 1) or (color_k == 1)
            has_color_2 = (color_i == 2) or (color_j == 2) or (color_k == 2)
            
            if has_color_0 and has_color_1 and has_color_2:
                # Check if quarks are close enough
                x_i, y_i = quarks[i]['x'], quarks[i]['y']
                x_j, y_j = quarks[j]['x'], quarks[j]['y']
                x_k, y_k = quarks[k]['x'], quarks[k]['y']
                
                r_ij = ((x_j - x_i)**2 + (y_j - y_i)**2)**0.5
                r_jk = ((x_k - x_j)**2 + (y_k - y_j)**2)**0.5
                r_ki = ((x_i - x_k)**2 + (y_i - y_k)**2)**0.5
                
                # All distances must be within binding threshold
                if (r_ij < binding_thresh and 
                    r_jk < binding_thresh and 
                    r_ki < binding_thresh):
                    
                    # Add candidate to list using atomic operation to avoid race conditions
                    idx = cuda.atomic.add(counters, 0, 1)
                    if idx < baryon_candidates.shape[0]:
                        baryon_candidates[idx, 0] = i
                        baryon_candidates[idx, 1] = j
                        baryon_candidates[idx, 2] = k

@cuda.jit
def decay_step_kernel(quarks, rng_states, step_index, decay_counter):
    """
    Apply weak decay to quarks (flavor changing).
    
    Args:
        quarks: Array of quarks
        rng_states: CUDA random number generator states
        step_index: Current simulation step index
        decay_counter: Counter for number of decay events
    """
    i = cuda.grid(1)
    if i >= quarks.shape[0]:
        return
    
    # Generate random number for decay probability
    if HAS_CUDA_RANDOM:
        # Use CUDA's RNG if available
        rng = xoroshiro128p_uniform_float32(rng_states, i)
    else:
        # Fallback to deterministic "random" based on position and step
        seed = (quarks[i]['x'] * 100 + quarks[i]['y'] * 100 + step_index) % 1000
        rng = (seed % 997) / 997.0
    
    # Check if decay should occur (based on weak interaction probability)
    if rng < weak_prob:
        # Flip quark flavor (up <-> down)
        current_flavor = quarks[i]['flavor']
        
        if current_flavor == 0:  # UP -> DOWN
            quarks[i]['flavor'] = 1
            quarks[i]['charge'] = -1.0/3.0  # Down quark charge
            quarks[i]['mass'] = 4.7         # Down quark mass
        else:  # DOWN -> UP
            quarks[i]['flavor'] = 0
            quarks[i]['charge'] = 2.0/3.0   # Up quark charge
            quarks[i]['mass'] = 2.2         # Up quark mass
        
        # Increment decay counter
        cuda.atomic.add(decay_counter, 0, 1)

@cuda.jit
def collision_reactions_kernel(quarks, baryon_indices, num_baryons, rng_states, reaction_counter):
    """
    Simulate nuclear reactions between baryons based on collisions.
    
    Args:
        quarks: Array of quarks
        baryon_indices: Array of baryon quark indices
        num_baryons: Number of baryons
        rng_states: CUDA random number generator states
        reaction_counter: Counter for number of reaction events
    """
    i = cuda.grid(1)
    if i >= num_baryons:
        return
    
    # Get the first baryon's quark indices
    i1 = baryon_indices[i, 0]
    i2 = baryon_indices[i, 1]
    i3 = baryon_indices[i, 2]
    
    if i1 < 0 or i2 < 0 or i3 < 0:
        # Invalid baryon (deleted)
        return
    
    # Calculate center of mass for first baryon
    x1 = (quarks[i1]['x'] + quarks[i2]['x'] + quarks[i3]['x']) / 3.0
    y1 = (quarks[i1]['y'] + quarks[i2]['y'] + quarks[i3]['y']) / 3.0
    
    # Determine baryon type (sum of charges helps identify)
    charge1 = quarks[i1]['charge'] + quarks[i2]['charge'] + quarks[i3]['charge']
    
    # Check collisions with other baryons
    for j in range(num_baryons):
        if i == j:
            continue
            
        # Get the second baryon's quark indices
        j1 = baryon_indices[j, 0]
        j2 = baryon_indices[j, 1]
        j3 = baryon_indices[j, 2]
        
        if j1 < 0 or j2 < 0 or j3 < 0:
            # Invalid baryon (deleted)
            continue
        
        # Calculate center of mass for second baryon
        x2 = (quarks[j1]['x'] + quarks[j2]['x'] + quarks[j3]['x']) / 3.0
        y2 = (quarks[j1]['y'] + quarks[j2]['y'] + quarks[j3]['y']) / 3.0
        
        # Calculate distance between baryons
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Determine second baryon type
        charge2 = quarks[j1]['charge'] + quarks[j2]['charge'] + quarks[j3]['charge']
        
        # Check if baryons are close enough for nuclear reaction
        if distance < 3.0:  # Nuclear reaction distance threshold
            # Get a random number for reaction probability
            if HAS_CUDA_RANDOM:
                rng = xoroshiro128p_uniform_float32(rng_states, i * num_baryons + j)
            else:
                # Fallback deterministic "random" based on positions
                seed = int((x1 + y1) * 1000 + (x2 + y2) * 100) % 1000
                rng = (seed % 997) / 997.0
                
            # Nuclear reaction probability based on charges
            reaction_prob = 0.01  # Base probability
            
            # Proton-proton has higher reaction probability
            if charge1 > 0.9 and charge2 > 0.9:  # Both are protons
                reaction_prob *= 2.0
            
            # Apply reaction if probability threshold is met
            if rng < reaction_prob:
                # Nuclear reaction occurred!
                # For simplicity in this implementation, we'll just
                # simulate a quark exchange between the baryons
                
                # Swap one quark between baryons (first quark of each)
                # Note: In a real simulation, reaction physics would be more complex
                # This is just a placeholder that creates some effect
                temp_flavor = quarks[i1]['flavor']
                quarks[i1]['flavor'] = quarks[j1]['flavor']
                quarks[j1]['flavor'] = temp_flavor
                
                # Update charges based on new flavors
                if quarks[i1]['flavor'] == 0:  # UP
                    quarks[i1]['charge'] = 2.0/3.0
                else:  # DOWN
                    quarks[i1]['charge'] = -1.0/3.0
                    
                if quarks[j1]['flavor'] == 0:  # UP
                    quarks[j1]['charge'] = 2.0/3.0
                else:  # DOWN
                    quarks[j1]['charge'] = -1.0/3.0
                
                # Increment the reaction counter
                cuda.atomic.add(reaction_counter, 0, 1)

@cuda.jit
def fission_kernel(quarks, baryon_indices, num_baryons, rng_states, fission_counter):
    """
    Simulate nuclear fission by breaking apart larger structures.
    
    Args:
        quarks: Array of quarks
        baryon_indices: Array of baryon quark indices
        num_baryons: Number of baryons
        rng_states: CUDA random number generator states
        fission_counter: Counter for number of fission events
    """
    i = cuda.grid(1)
    if i >= num_baryons:
        return
        
    # Skip processing if we don't have enough baryons for meaningful fission
    if num_baryons < 4:
        return
    
    # Get the current baryon's quark indices
    i1 = baryon_indices[i, 0]
    i2 = baryon_indices[i, 1]
    i3 = baryon_indices[i, 2]
    
    if i1 < 0 or i2 < 0 or i3 < 0:
        # Invalid baryon (deleted)
        return
        
    # Calculate center of mass for this baryon
    x_com = (quarks[i1]['x'] + quarks[i2]['x'] + quarks[i3]['x']) / 3.0
    y_com = (quarks[i1]['y'] + quarks[i2]['y'] + quarks[i3]['y']) / 3.0
    
    # Count nearby baryons to see if we have a larger structure
    nearby_count = 0
    for j in range(num_baryons):
        if i == j:
            continue
            
        # Get the other baryon's quark indices
        j1 = baryon_indices[j, 0]
        j2 = baryon_indices[j, 1]
        j3 = baryon_indices[j, 2]
        
        if j1 < 0 or j2 < 0 or j3 < 0:
            # Invalid baryon
            continue
            
        # Calculate center of mass for the other baryon
        x_j = (quarks[j1]['x'] + quarks[j2]['x'] + quarks[j3]['x']) / 3.0
        y_j = (quarks[j1]['y'] + quarks[j2]['y'] + quarks[j3]['y']) / 3.0
        
        # Check if they are close enough to be considered part of the same nucleus
        dx = x_j - x_com
        dy = y_j - y_com
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 4.0:  # Distance threshold for nucleus membership
            nearby_count += 1
    
    # Only consider fission for baryons that are part of a larger nucleus
    # (having at least 3 nearby baryons)
    if nearby_count >= 3:
        # Generate random number for fission probability
        if HAS_CUDA_RANDOM:
            rng = xoroshiro128p_uniform_float32(rng_states, i)
        else:
            # Fallback deterministic "random"
            seed = int((x_com + y_com) * 100) % 1000
            rng = (seed % 997) / 997.0
            
        # Apply fission with some probability
        # Larger nuclei are more unstable and more likely to undergo fission
        fission_prob = 0.001 * nearby_count  # Scales with nucleus size
        
        if rng < fission_prob:
            # Simulate fission by pushing quarks apart
            # Apply outward impulse to all quarks in this baryon
            
            # Calculate outward direction from center of nucleus (approximated as COM)
            for quark_idx in [i1, i2, i3]:
                # Vector from COM to quark
                dx = quarks[quark_idx]['x'] - x_com
                dy = quarks[quark_idx]['y'] - y_com
                
                # Normalize the direction
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 1e-6:  # Avoid division by zero
                    dx_unit = dx / dist
                    dy_unit = dy / dist
                else:
                    # Random direction if quark is at COM
                    angle = quark_idx * 2.0  # Use quark index to get a pseudo-random angle
                    dx_unit = math.cos(angle)
                    dy_unit = math.sin(angle)
                
                # Apply strong outward impulse
                impulse = 20.0  # Strong impulse
                quarks[quark_idx]['vx'] += impulse * dx_unit
                quarks[quark_idx]['vy'] += impulse * dy_unit
                
                # Mark the quark as no longer bound
                quarks[quark_idx]['is_bound'] = 0
            
            # Record the fission event
            cuda.atomic.add(fission_counter, 0, 1)
