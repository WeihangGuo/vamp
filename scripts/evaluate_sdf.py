import numpy as np
import vamp
import time
from fire import Fire

def main(robot_name: str = "panda", n_samples: int = 10000):
    # Load robot module
    if not hasattr(vamp, robot_name):
        print(f"Robot {robot_name} not found in vamp.")
        available_robots = [attr for attr in dir(vamp) if hasattr(getattr(vamp, attr), "sdf")]
        print(f"Available robots: {available_robots}")
        return
    
    robot = getattr(vamp, robot_name)
    
    # Create environment with some obstacles
    env = vamp.Environment()
    
    # Add some spheres to create a non-trivial environment
    obstacles = [
        ([0.5, 0.0, 0.5], 0.2),
        ([0.0, 0.5, 0.5], 0.2),
        ([0.5, 0.5, 0.5], 0.2),
        ([0.3, -0.3, 0.3], 0.15),
        ([-0.3, 0.3, 0.3], 0.15),
        ([0.6, 0.0, 0.2], 0.1),
    ]
    
    for center, radius in obstacles:
        env.add_sphere(vamp.Sphere(center, radius))
    
    print(f"Evaluating SDF for robot: {robot_name}")
    print(f"Environment: {len(obstacles)} spheres")
    
    # Initialize RNG
    rng = robot.halton()
    rng.reset()
    
    sdf_values = []
    
    print(f"Sampling {n_samples} configurations...")
    start_time = time.time()
    
    for i in range(n_samples):
        q = rng.next()
        # Compute SDF
        # Positive means outside (safe), Negative means inside (collision)
        dist = robot.sdf(q, env)
        sdf_values.append(dist)
        
    end_time = time.time()
    duration = end_time - start_time
    
    sdf_values = np.array(sdf_values)
    
    print("-" * 30)
    print(f"Results for {n_samples} queries:")
    print(f"Total Time:       {duration:.4f} s")
    print(f"Average Time:     {duration/n_samples*1e6:.2f} µs/query")
    print(f"Throughput:       {n_samples/duration:.2f} queries/s")
    print("-" * 30)
    print(f"SDF Statistics:")
    print(f"  Min Dist:       {np.min(sdf_values):.4f}")
    print(f"  Max Dist:       {np.max(sdf_values):.4f}")
    print(f"  Mean Dist:      {np.mean(sdf_values):.4f}")
    print(f"  Std Dev:        {np.std(sdf_values):.4f}")
    print("-" * 30)
    
    n_collisions = np.sum(sdf_values < 0)
    print(f"Collisions:       {n_collisions} ({n_collisions/n_samples*100:.2f}%)")
    print(f"Safe Configs:     {n_samples - n_collisions}")

    # Benchmark Solver
    print("\n" + "=" * 30)
    print("Benchmarking Solver (project_to_valid)...")
    
    def benchmark_solver(steps):
        print(f"\nRunning Solver with {steps} steps...")
        valid_count = 0
        total_time = 0
        
        # Use a subset of samples to save time if needed, but let's do all
        # To strictly measure solver time, we measure the call
        
        start_t = time.time()
        for i in range(n_samples // 10): # Run on 10% of samples to be quick, or full? Let's do 1000.
             if i >= 1000: break
             
             # Re-generate to ensure randomness or reuse? Let's reuse configs from a list if we stored them,
             # but we didn't store them all.
             # Let's just generate new ones or use a block. 
             # Ideally we want to see if it fixes collisions.
             
             q_init = rng.next() # New random sample
             
             # project_to_valid returns a list of valid configurations (or candidates)
             # C++ signature: returns std::vector<Type>
             candidates = robot.project_to_valid(q_init, env, steps=steps, learning_rate=0.5, noise_scale=0.1)
             
             if len(candidates) > 0:
                 valid_count += 1
                 
        end_t = time.time()
        elapsed = end_t - start_t
        n_bench = min(n_samples // 10, 1000)
        
        print(f"  Time for {n_bench} calls: {elapsed:.4f} s")
        print(f"  Avg Time: {elapsed/n_bench*1e3:.4f} ms/call")
        print(f"  Success Rate (returned >=1 candidates): {valid_count}/{n_bench} ({valid_count/n_bench*100:.1f}%)")

    benchmark_solver(10)
    benchmark_solver(100)

if __name__ == "__main__":
    Fire(main)
