import numpy as np
import pyopencl as cl
import pyopencl.array
import math
from particles import ParticleType
import logging

logger = logging.getLogger("NuclearSim")

class NuclearForces:
    def __init__(self):
        self.setup_opencl()
        self.strong_strength = 150.0
        self.coulomb_strength = 30.0
        self.pauli_strength = 35.0
        self.gravity_strength = 0.01
        self.weak_strength = 1.0
        
    def setup_opencl(self):
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")
            
        ctx = None
        
        # Try GPU first
        for platform in platforms:
            try:
                gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
                if gpu_devices:
                    ctx = cl.Context([gpu_devices[0]])
                    logger.info(f"Using GPU: {gpu_devices[0].name}")
                    break
            except:
                continue
        
        # Fall back to CPU
        if ctx is None:
            for platform in platforms:
                try:
                    cpu_devices = platform.get_devices(device_type=cl.device_type.CPU)
                    if cpu_devices:
                        ctx = cl.Context([cpu_devices[0]])
                        logger.info(f"Using CPU: {cpu_devices[0].name}")
                        break
                except:
                    continue
        
        if ctx is None:
            raise RuntimeError("Failed to create OpenCL context")
        
        self.ctx = ctx
        self.queue = cl.CommandQueue(self.ctx)
        self.compile_kernel()
        
    def compile_kernel(self):
        kernel_src = """
        #define EPSILON 0.15f

        __kernel void update_forces_and_positions(
                                   __global float4* particles,
                                   __global int* types,
                                   int num_particles,
                                   float2 center,
                                   float strong_strength,
                                   float coulomb_strength,
                                   float pauli_strength,
                                   float dt) {
            int i = get_global_id(0);
            if (i >= num_particles) return;
            
            float4 particle = particles[i];
            float px = particle.x;
            float py = particle.y;
            float pvx = particle.z;
            float pvy = particle.w;
            int type_i = types[i];
            
            float totalFx = 0.0f;
            float totalFy = 0.0f;
            
            float nucleon_radius = 2.5f;
            float max_force = 12.0f;
            
            for (int j = 0; j < num_particles; j++) {
                if (i == j) continue;
                
                float4 other = particles[j];
                float qx = other.x;
                float qy = other.y;
                int type_j = types[j];
                
                float dx = qx - px;
                float dy = qy - py;
                float dist2 = dx*dx + dy*dy;
                if (dist2 < 0.01f) continue;
                
                float dist = sqrt(dist2);
                float net_force = 0.0f;
                
                // Hard-core repulsion
                float min_allowed_dist = nucleon_radius * 1.7f;
                if (dist < min_allowed_dist) {
                    float overlap = min_allowed_dist - dist;
                    net_force -= 60.0f * pow(overlap / min_allowed_dist, 1.5f);
                }
                
                // Nuclear force
                float strong_range = 7.0f;
                float r_ratio = dist / strong_range;
                
                if (dist < 2.8f) {
                    // Repulsive core
                    net_force -= 0.7f * strong_strength / (dist2 + EPSILON);
                } else if (dist < 9.0f) {
                    // Attractive region
                    net_force += 1.25f * strong_strength * exp(-r_ratio) / (dist + EPSILON);
                } else {
                    // Tail
                    net_force += 0.15f * strong_strength * exp(-r_ratio * 1.8f) / (dist + EPSILON);
                }
                
                // Coulomb force - only between protons
                if (type_i == 0 && type_j == 0) {
                    net_force -= coulomb_strength / (dist2 + EPSILON);
                }
                
                // Pauli exclusion for same particle types
                if (type_i == type_j) {
                    float pauli_range = 8.0f;
                    if (dist < pauli_range) {
                        net_force -= pauli_strength * exp(-dist / pauli_range * 2.0f);
                    }
                }
                
                // Apply force along direction vector
                net_force = clamp(net_force, -max_force, max_force);
                if (dist > 0.0f) {
                    totalFx += dx * net_force / dist;
                    totalFy += dy * net_force / dist;
                }
            }
            
            // Center-of-mass force
            float center_dx = center.x - px;
            float center_dy = center.y - py;
            float center_dist = sqrt(center_dx*center_dx + center_dy*center_dy);
            float nuclear_radius = 1.2f * pow(num_particles, 1.0f/3.0f) * 2.0f;
            
            if (center_dist > nuclear_radius * 1.5f && center_dist > 0.01f) {
                float center_force = 0.03f * (center_dist - nuclear_radius);
                totalFx += center_force * center_dx / center_dist;
                totalFy += center_force * center_dy / center_dist;
            }
            
            // Update velocity
            pvx += totalFx * dt;
            pvy += totalFy * dt;
            
            // Apply damping
            pvx *= 0.85f;
            pvy *= 0.85f;
            
            // Update position
            px += pvx * dt;
            py += pvy * dt;
            
            particles[i].x = px;
            particles[i].y = py;
            particles[i].z = pvx;
            particles[i].w = pvy;
        }
        """
        try:
            self.program = cl.Program(self.ctx, kernel_src).build(options="-cl-fast-relaxed-math")
        except Exception as e:
            logger.error(f"Failed to build OpenCL program: {e}")
            try:
                self.program = cl.Program(self.ctx, kernel_src).build()
                logger.info("OpenCL program built with default options")
            except Exception as e2:
                logger.error(f"Failed to build with fallback options: {e2}")
                raise
    
    def update_particles_gpu(self, particles, dt):
        num_particles = len(particles)
        if num_particles == 0:
            return
            
        h_particles = np.zeros((num_particles, 4), dtype=np.float32)
        h_types = np.zeros(num_particles, dtype=np.int32)
        
        # Copy data to arrays
        for i, p in enumerate(particles):
            h_particles[i, 0] = p.x
            h_particles[i, 1] = p.y
            h_particles[i, 2] = p.vx
            h_particles[i, 3] = p.vy
            h_types[i] = 0 if p.type == ParticleType.PROTON else 1
        
        # Transfer to GPU
        particles_buffer = cl.array.to_device(self.queue, h_particles)
        types_buffer = cl.array.to_device(self.queue, h_types)
        
        # Calculate center of mass
        center_x = sum(p.x for p in particles) / len(particles)
        center_y = sum(p.y for p in particles) / len(particles)
        center = np.array([center_x, center_y], dtype=np.float32)
        
        # Run OpenCL kernel
        try:
            event = self.program.update_forces_and_positions(
                self.queue, (num_particles,), None,
                particles_buffer.data, types_buffer.data,
                np.int32(num_particles), center,
                np.float32(self.strong_strength),
                np.float32(self.coulomb_strength),
                np.float32(self.pauli_strength),
                np.float32(dt)
            )
            event.wait()
        except Exception as e:
            logger.error(f"OpenCL kernel execution failed: {e}")
            return
        
        # Get results back from GPU
        result_particles = particles_buffer.get()
        
        # Update the particles
        for i, p in enumerate(particles):
            p.x = result_particles[i, 0]
            p.y = result_particles[i, 1]
            p.vx = result_particles[i, 2]
            p.vy = result_particles[i, 3]
    
    def update_particles_cpu(self, particles, dt):
        """CPU fallback implementation for forces calculation"""
        if not particles:
            return
            
        # Calculate center of mass
        center_x = sum(p.x for p in particles) / len(particles)
        center_y = sum(p.y for p in particles) / len(particles)
        
        # Calculate forces for all particles
        forces = [[0.0, 0.0] for _ in range(len(particles))]
        
        for i, p1 in enumerate(particles):
            for j, p2 in enumerate(particles):
                if i == j:
                    continue
                    
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                dist2 = dx*dx + dy*dy
                
                if dist2 < 0.01:
                    continue
                    
                dist = math.sqrt(dist2)
                net_force = 0.0
                
                # Hard-core repulsion
                min_allowed_dist = 4.25  # 1.7 * radius
                if dist < min_allowed_dist:
                    overlap = min_allowed_dist - dist
                    net_force -= 60.0 * pow(overlap / min_allowed_dist, 1.5)
                
                # Strong nuclear force
                strong_range = 7.0
                r_ratio = dist / strong_range
                
                if dist < 2.8:
                    # Repulsive core
                    net_force -= 0.7 * self.strong_strength / (dist2 + 0.15)
                elif dist < 9.0:
                    # Attractive region
                    net_force += 1.25 * self.strong_strength * math.exp(-r_ratio) / (dist + 0.15)
                else:
                    # Tail
                    net_force += 0.15 * self.strong_strength * math.exp(-r_ratio * 1.8) / (dist + 0.15)
                
                # Coulomb force between protons
                if p1.type == ParticleType.PROTON and p2.type == ParticleType.PROTON:
                    net_force -= self.coulomb_strength / (dist2 + 0.15)
                
                # Pauli exclusion for same types
                if p1.type == p2.type:
                    pauli_range = 8.0
                    if dist < pauli_range:
                        net_force -= self.pauli_strength * math.exp(-dist / pauli_range * 2.0)
                
                # Cap force and apply
                net_force = max(-12.0, min(12.0, net_force))
                
                if dist > 0:
                    forces[i][0] += dx * net_force / dist
                    forces[i][1] += dy * net_force / dist
            
            # Center-of-mass containment force
            center_dx = center_x - p1.x
            center_dy = center_y - p1.y
            center_dist = math.sqrt(center_dx**2 + center_dy**2)
            nuclear_radius = 1.2 * pow(len(particles), 1.0/3.0) * 2.0
            
            if center_dist > nuclear_radius * 1.5 and center_dist > 0.01:
                center_force = 0.03 * (center_dist - nuclear_radius)
                forces[i][0] += center_force * center_dx / center_dist
                forces[i][1] += center_force * center_dy / center_dist
        
        # Update velocities and positions
        for i, p in enumerate(particles):
            # Update velocity
            p.vx += forces[i][0] * dt
            p.vy += forces[i][1] * dt
            
            # Apply damping
            p.vx *= 0.85
            p.vy *= 0.85
            
            # Update position
            p.x += p.vx * dt
            p.y += p.vy * dt
