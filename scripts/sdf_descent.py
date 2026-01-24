import numpy as np
import vamp
import time
from fire import Fire

def project_to_valid(robot, q, env, steps=100, learning_rate=0.5, noise_scale=0.1):
    n_candidates = 8
    dim = len(q)
    
    # Initialize candidates with noise
    # Use fixed seed for reproducibility matching C++
    rng = np.random.default_rng(42)
    
    # Broadcast and add noise
    candidates = np.tile(q, (n_candidates, 1))
    noise = rng.uniform(-noise_scale, noise_scale, (n_candidates, dim))
    candidates += noise
    
    for step in range(steps):
        # 1. Compute SDFs
        dists = np.zeros(n_candidates)
        for i in range(n_candidates):
            dists[i] = robot.sdf(candidates[i], env)
            
        # 2. No early exit - we want to maximize SDF
            
        # 3. Compute Gradients (Finite Difference)
        grads = np.zeros_like(candidates)
        h = 1e-4
        
        for i in range(n_candidates):
            # Compute gradient for all, even if safe
            original_q = candidates[i].copy()
            for d in range(dim):
                val = original_q[d]
                
                # f(x+h)
                candidates[i, d] = val + h
                f_plus = robot.sdf(candidates[i], env)
                
                # f(x-h)
                candidates[i, d] = val - h
                f_minus = robot.sdf(candidates[i], env)
                
                # Restore
                candidates[i, d] = val
                
                grads[i, d] = (f_plus - f_minus) / (2 * h)
        
        # 4. Update Rule: Gradient Ascent
        # q_new = q + lr * grad
        for i in range(n_candidates):
             candidates[i] += grads[i] * learning_rate
                
    return candidates

def once(robot_name: str = "panda"):
    # Load robot module
    if not hasattr(vamp, robot_name):
        print(f"Robot {robot_name} not found in vamp.")
        return
    
    robot = getattr(vamp, robot_name)
    
    # Create environment
    env = vamp.Environment()
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
        
    print(f"Benchmarking Python Solver for {robot_name}")
    
    rng = robot.halton()
    rng.reset()
    
    q_init = rng.next()
    
    initial_valid = robot.sdf(q_init, env) >= 0
    initial_dist = robot.sdf(q_init, env)

    # Test 10 steps
    # Lowered learning rate to 0.05 since we are doing direct gradient steps now
    candidates_10 = project_to_valid(robot, q_init, env, steps=10, learning_rate=0.05, noise_scale=0.1)
    dist_10 = np.max([robot.sdf(q, env) for q in candidates_10])
    candidates_100 = project_to_valid(robot, q_init, env, steps=100, learning_rate=0.05, noise_scale=0.1)
    dist_100 = np.max([robot.sdf(q, env) for q in candidates_100])
    candidates_1000 = project_to_valid(robot, q_init, env, steps=1000, learning_rate=0.05, noise_scale=0.1)
    dist_1000 = np.max([robot.sdf(q, env) for q in candidates_1000])
    print(f"Inital Dist: {initial_dist}, Dist 10: {dist_10}, Dist 100: {dist_100}, Dist 1000: {dist_1000}")
    candidates_10000 = project_to_valid(robot, q_init, env, steps=10000, learning_rate=0.05, noise_scale=0.1)
    dist_10000 = np.max([robot.sdf(q, env) for q in candidates_10000])
    print(f"Inital Dist: {initial_dist}, Dist 10: {dist_10}, Dist 100: {dist_100}, Dist 1000: {dist_1000}, Dist 10000: {dist_10000}")
    return initial_valid, dist_10, dist_100, dist_1000, dist_10000

def bench(robot_name: str = "panda", n_samples: int = 100):
    # Load robot module
    if not hasattr(vamp, robot_name):
        print(f"Robot {robot_name} not found in vamp.")
        return
    
    robot = getattr(vamp, robot_name)
    
    # Create environment
    env = vamp.Environment()
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
        
    print(f"Benchmarking Python Solver for {robot_name}")
    
    rng = robot.halton()
    rng.reset()
    
    valid_count_10 = 0
    valid_count_100 = 0
    initial_valid_count = 0

    start_time = time.time()
    
    print(f"Running {n_samples} queries...")
    
    for i in range(n_samples):
        q_init = rng.next()
        
        initial_valid = robot.sdf(q_init, env) >= 0
 
        if initial_valid:
            initial_valid_count += 1
        
        # Test 10 steps
        candidates_10 = project_to_valid(robot, q_init, env, steps=10, learning_rate=0.05, noise_scale=0.1)
        found_valid = False
        for q in candidates_10:
            if robot.sdf(q, env) >= 0:
                found_valid = True
                break
        if found_valid:
            valid_count_10 += 1
            
        # Test 100 steps (independent run for stats, though inefficient)
        candidates_100 = project_to_valid(robot, q_init, env, steps=100, learning_rate=0.05, noise_scale=0.1)
        found_valid = False
        for q in candidates_100:
            if robot.sdf(q, env) >= 0:
                found_valid = True
                break
        if found_valid:
            valid_count_100 += 1
            
    elapsed = time.time() - start_time
    
    print(f"Total Time: {elapsed:.2f}s")
    print(f"Initial Valid: {initial_valid_count}/{n_samples} ({initial_valid_count/n_samples*100:.1f}%)")
    print(f"Success Rate (10 steps):  {valid_count_10}/{n_samples} ({valid_count_10/n_samples*100:.1f}%)")
    print(f"Success Rate (100 steps): {valid_count_100}/{n_samples} ({valid_count_100/n_samples*100:.1f}%)")


if __name__ == "__main__":
    bench()

