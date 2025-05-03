// OpenCL kernel functions for nuclear physics simulation

// Calculate forces between nucleons
__kernel void compute_nuclear_forces(__global float4* particles,
                                   __global float4* forces,
                                   __global int* types,
                                   int num_particles,
                                   float strength) {
    int i = get_global_id(0);
    if (i >= num_particles) return;
    
    float totalFx = 0.0f;
    float totalFy = 0.0f;
    
    // Compute force from all other particles
    for (int j = 0; j < num_particles; j++) {
        if (i == j) continue;
        
        float dx = particles[j].x - particles[i].x;
        float dy = particles[j].y - particles[i].y;
        float dist2 = dx*dx + dy*dy;
        
        // Avoid division by zero
        if (dist2 < 1e-10f) continue;
        
        float dist = sqrt(dist2);
        
        // Nuclear force (attractive at medium range, repulsive very close)
        float force = 0.0f;
        
        // Coulomb repulsion between protons
        if (types[i] == 0 && types[j] == 0) {  // Both protons
            force -= 2.0f / dist2;  // Repulsive
        }
        
        // Nuclear strong force (simplified)
        float r = dist;
        float r0 = 10.0f;  // Characteristic radius
        float nuclear_force = strength * exp(-r/r0) * (1.0f - exp(-r/(r0/3.0f))) / r;
        force += nuclear_force;
        
        // Apply force in the correct direction
        totalFx += force * dx / dist;
        totalFy += force * dy / dist;
    }
    
    forces[i].x = totalFx;
    forces[i].y = totalFy;
}

// Update particle positions based on forces
__kernel void update_particles(__global float4* particles,
                              __global float4* forces,
                              int num_particles,
                              float dt) {
    int gid = get_global_id(0);
    if (gid < num_particles) {
        // Update position based on velocity
        particles[gid].x += particles[gid].z * dt;
        particles[gid].y += particles[gid].w * dt;
        
        // Apply forces to update velocity
        particles[gid].z += forces[gid].x * dt;
        particles[gid].w += forces[gid].y * dt;
    }
}

// Calculate decay probability
__kernel void calculate_decay_probability(__global float* half_lives,
                                         __global float* time_elapsed,
                                         __global float* probabilities,
                                         int num_nuclei) {
    int i = get_global_id(0);
    if (i < num_nuclei) {
        // Skip stable nuclei
        if (half_lives[i] > 1e20f) {
            probabilities[i] = 0.0f;
            return;
        }
        
        // Calculate decay probability based on half-life
        float lambda = 0.693147f / half_lives[i];  // ln(2) / half_life
        float dt = time_elapsed[i];
        
        // P = 1 - e^(-λt)
        probabilities[i] = 1.0f - exp(-lambda * dt);
    }
}
