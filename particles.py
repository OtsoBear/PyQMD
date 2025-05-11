from enum import Enum
import random
import math

class ParticleType(Enum):
    PROTON = 0
    NEUTRON = 1
    ALPHA = 2
    ELECTRON = 3
    GAMMA = 4
    POSITRON = 5

class DecayType(Enum):
    NONE = 0
    ALPHA = 1
    BETA_MINUS = 2
    BETA_PLUS = 3
    GAMMA = 4
    NEUTRON_EMISSION = 5
    PROTON_EMISSION = 6
    SPONTANEOUS_FISSION = 7

class Particle:
    def __init__(self, x, y, particle_type, vx=0, vy=0):
        self.x = x
        self.y = y
        self.type = particle_type
        self.vx = vx
        self.vy = vy
        self.radius = 2.5 if particle_type in [ParticleType.PROTON, ParticleType.NEUTRON] else 1.0
        self.lifetime = {
            ParticleType.PROTON: float('inf'),
            ParticleType.NEUTRON: float('inf'),
            ParticleType.ALPHA: 2.0,
            ParticleType.ELECTRON: 3.0,
            ParticleType.GAMMA: 1.0,
            ParticleType.POSITRON: 3.0,
        }.get(particle_type, float('inf'))
        self.age = 0
        
    def get_color(self):
        colors = {
            ParticleType.PROTON: (255, 100, 100),
            ParticleType.NEUTRON: (100, 100, 255),
            ParticleType.ALPHA: (255, 200, 0),
            ParticleType.ELECTRON: (0, 255, 255),
            ParticleType.GAMMA: (0, 255, 0),
            ParticleType.POSITRON: (255, 0, 255),
        }
        return colors.get(self.type, (255, 255, 255))

class Nucleus:
    def __init__(self, protons, neutrons, x, y):
        self.protons = protons
        self.neutrons = neutrons
        self.x = x
        self.y = y
        self.particles = []
        self.stability = 0.0
        self.initialize_particles()
        
    def initialize_particles(self):
        total = self.protons + self.neutrons
        r0 = 1.2
        nuclear_radius = r0 * (total ** (1/3))
        initial_radius = nuclear_radius * 0.7
        shell_capacity = [2, 8, 20, 28, 50, 82, 126]
        shell_radii = [initial_radius * (i+1)/len(shell_capacity) for i in range(len(shell_capacity))]
        
        placed_protons, placed_neutrons = 0, 0
        
        def place_in_shell(shell_index, is_proton):
            nonlocal placed_protons, placed_neutrons
            shell_radius = shell_radii[min(shell_index, len(shell_radii)-1)]
            radius = shell_radius * (0.8 + 0.2 * random.random())
            
            best_angle, max_min_dist = 0, 0
            for _ in range(20):  # Try multiple angles
                angle = random.uniform(0, 2 * math.pi)
                x = self.x + radius * math.cos(angle)
                y = self.y + radius * math.sin(angle)
                
                # Check distance to other particles of same type
                min_dist = float('inf')
                for p in self.particles:
                    if (p.type == ParticleType.PROTON and is_proton) or \
                       (p.type == ParticleType.NEUTRON and not is_proton):
                        dist = math.sqrt((p.x - x)**2 + (p.y - y)**2)
                        min_dist = min(min_dist, dist)
                
                if min_dist == float('inf') or min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_angle = angle
            
            x = self.x + radius * math.cos(best_angle)
            y = self.y + radius * math.sin(best_angle)
            
            if is_proton:
                self.particles.append(Particle(x, y, ParticleType.PROTON))
                placed_protons += 1
            else:
                self.particles.append(Particle(x, y, ParticleType.NEUTRON))
                placed_neutrons += 1
        
        # Place particles in shells
        shell_index = 0
        while placed_protons < self.protons and placed_neutrons < self.neutrons:
            shell_size = shell_capacity[min(shell_index, len(shell_capacity)-1)]
            pairs_to_place = min(shell_size // 2, 
                                min(self.protons - placed_protons, self.neutrons - placed_neutrons))
            
            for _ in range(pairs_to_place):
                place_in_shell(shell_index, True)   # Proton
                place_in_shell(shell_index, False)  # Neutron
            
            shell_index += 1
            if shell_index >= len(shell_capacity):
                shell_index = len(shell_capacity) - 1
        
        while placed_protons < self.protons:
            place_in_shell(shell_index, True)
        
        while placed_neutrons < self.neutrons:
            place_in_shell(shell_index, False)
    
    def should_decay(self, dt):
        """Determine if nucleus should decay based on half-life with improved calculation."""
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
    
    def adjust_particles(self, decay_type):
        nucleons = {ParticleType.PROTON: 0, ParticleType.NEUTRON: 0}
        for p in self.particles:
            if p.type in nucleons:
                nucleons[p.type] += 1
        
        if decay_type == DecayType.ALPHA:
            protons_to_remove = 2
            neutrons_to_remove = 2
        elif decay_type == DecayType.BETA_MINUS:
            # Convert neutron to proton
            for p in self.particles:
                if p.type == ParticleType.NEUTRON:
                    p.type = ParticleType.PROTON
                    break
            return
        elif decay_type == DecayType.BETA_PLUS:
            # Convert proton to neutron
            for p in self.particles:
                if p.type == ParticleType.PROTON:
                    p.type = ParticleType.NEUTRON
                    break
            return
        elif decay_type == DecayType.NEUTRON_EMISSION:
            protons_to_remove = 0
            neutrons_to_remove = 1
        elif decay_type == DecayType.PROTON_EMISSION:
            protons_to_remove = 1
            neutrons_to_remove = 0
        else:
            return
        
        # For alpha decay and nucleon emission
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            if protons_to_remove > 0 and p.type == ParticleType.PROTON:
                particles_to_remove.append(i)
                protons_to_remove -= 1
            elif neutrons_to_remove > 0 and p.type == ParticleType.NEUTRON:
                particles_to_remove.append(i)
                neutrons_to_remove -= 1
            
            if protons_to_remove == 0 and neutrons_to_remove == 0:
                break
                
        # Remove particles in reverse order to maintain indices
        particles_to_remove.sort(reverse=True)
        for i in particles_to_remove:
            if i < len(self.particles):
                self.particles.pop(i)
        
        # Apply velocity damping for stability
        for p in self.particles:
            p.vx *= 0.8
            p.vy *= 0.8
    
    def update_center_of_mass(self):
        if self.particles:
            self.x = sum(p.x for p in self.particles) / len(self.particles)
            self.y = sum(p.y for p in self.particles) / len(self.particles)
