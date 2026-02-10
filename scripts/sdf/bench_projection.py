import numpy as np
import time
import vamp
np.random.seed(0)

PROBLEM_SPHERES = [
    [0.55, 0, 0.25],
    [0.35, 0.35, 0.25],
    [0, 0.55, 0.25],
    [-0.55, 0, 0.25],
    [-0.35, -0.35, 0.25],
    [0, -0.55, 0.25],
    [0.35, -0.35, 0.25],
    [-0.35, 0.35, 0.25],
    
    [0.35, 0.35, 0.8],
    [0, 0.55, 0.8],
    [-0.35, 0.35, 0.8],
    [-0.55, 0, 0.8],
    [-0.35, -0.35, 0.8],
    [0, -0.55, 0.8],
    [0.35, -0.35, 0.8],
    [0.55, 0, 0.8],
]
SPHERE_RADIUS = 0.2

lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

def main():
    # Create environment
    env = vamp.Environment()
    for pos in PROBLEM_SPHERES:
        env.add_sphere(vamp.Sphere(pos, SPHERE_RADIUS))

    n_samples = 10000
    print(f"Starting Benchmark with {n_samples} samples...")

    n_success = 0
    total_time = 0.0
    env_collision_free = 0
    # We sample configs and if they are in collision, we project them
    # and measure the time.
    valid_samples = 0
    while valid_samples < n_samples:
        q_rand = np.random.uniform(lower, upper).astype(np.float32)
        
        # Check initial SDF
        initial_valid = vamp.panda.validate(q_rand, env)
        if initial_valid:
            continue

        valid_samples += 1
            
        start_t = time.perf_counter()
        q_proj = vamp.panda.project_to_valid(q_rand, env)
        end_t = time.perf_counter()
            
        # Verify projection success
        # Convert q_proj to numpy if it's not already
        q_proj = np.array(q_proj).flatten()[:len(lower)]
        sdf_env = vamp.panda.sdf(q_proj, env)
        min_sdf = np.min(sdf_env)
        valid = vamp.panda.validate(q_proj, env)
            
        if valid: # Small epsilon for numerical precision
            n_success += 1
        if min_sdf > 0:
            env_collision_free += 1
        total_time += (end_t - start_t)

    print("-" * 30)
    print(f"Benchmark Results:")
    print(f"Total Samples:          {valid_samples}")
    if valid_samples > 0:
        print(f"Environment Collision Free: {env_collision_free} ({100.0 * env_collision_free / valid_samples:.2f}%)")
        print(f"Env + robot Collision Free: {n_success} ({100.0 * n_success / valid_samples:.2f}%)")
        print(f"Average Projection Time: {(total_time / valid_samples) * 1000.0:.4f} ms")
    else:
        print("No samples were in collision, try increasing n_samples or adjusting the environment.")
    print("-" * 30)

if __name__ == "__main__":
    main()