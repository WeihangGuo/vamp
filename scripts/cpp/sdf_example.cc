#include <chrono>
#include <vector>
#include <array>
#include <utility>
#include <iostream>


#include <vamp/collision/factory.hh>
#include <vamp/planning/validate.hh>
#include <vamp/planning/simplify.hh>
#include <vamp/robots/panda.hh>
#include <vamp/random/halton.hh>

using Robot = vamp::robots::Panda;
static constexpr const std::size_t rake = vamp::FloatVectorWidth;
using EnvironmentInput = vamp::collision::Environment<float>;
using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;



// Spheres for the cage problem - (x, y, z) center coordinates with fixed, common radius defined below
static const std::vector<std::array<float, 3>> problem = {
    {0.55, 0, 0.25},
    {0.35, 0.35, 0.25},
    {0, 0.55, 0.25},
    {-0.55, 0, 0.25},
    {-0.35, -0.35, 0.25},
    {0, -0.55, 0.25},
    {0.35, -0.35, 0.25},
    {-0.35, 0.35, 0.25},
    {0.35, 0.35, 0.8},
    {0, 0.55, 0.8},
    {-0.35, 0.35, 0.8},
    {-0.55, 0, 0.8},
    {-0.35, -0.35, 0.8},
    {0, -0.55, 0.8},
    {0.35, -0.35, 0.8},
    {0.55, 0, 0.8},
};

// Radius for obstacle spheres
static constexpr float radius = 0.2;

auto main(int, char **) -> int
{
    // Build sphere cage environment
    EnvironmentInput environment;
    for (const auto &sphere : problem)
    {
        environment.spheres.emplace_back(vamp::collision::factory::sphere::array(sphere, radius));
    }

    environment.sort();
    auto env_v = EnvironmentVector(environment);

    // Benchmark
    vamp::rng::Halton<Robot> sampler;
    int n_samples = 1000;
    int n_collisions = 0;
    int n_success = 0;
    double total_time_ms = 0.0;
    
    std::cout << "Starting Benchmark with " << n_samples << " samples..." << std::endl;

    for (int i = 0; i < n_samples; ++i) {
        auto q_random = sampler.next();
        std::array<float, Robot::dimension> q_curr;
        q_random.to_array(q_curr.data());

        // Check initial collision
        std::vector<float> q_vec(q_curr.begin(), q_curr.end());
        auto block = Robot::ConfigurationBlock<rake>(q_vec, true);
        auto dists = Robot::sdf(env_v, block);
        
        float min_dist = 1e9;
        for(const auto& v : dists) {
            auto arr = v.to_array();
            for(auto d : arr) {
                if(d < min_dist) min_dist = d;
            }
        }

        if (min_dist >= 0) continue;
        
        n_collisions++;
        
        // Project
        auto start_t = std::chrono::high_resolution_clock::now();
        
        std::array<float, Robot::dimension> q_new = q_curr;
        float current_min_dist = min_dist;
        int iter = 0;
        const int max_iters = 100;
        float alpha = 0.1f;

        while (current_min_dist < 0 && iter < max_iters) {
            // Re-evaluate
            std::vector<float> q_v(q_new.begin(), q_new.end());
            auto b = Robot::ConfigurationBlock<rake>(q_v, true);
            auto res = Robot::sdf_gradient(env_v, b);

            current_min_dist = 1e9;
            for(const auto& v : res.first) {
                auto arr = v.to_array();
                for(auto d : arr) {
                    if(d < current_min_dist) current_min_dist = d;
                }
            }

            if (current_min_dist >= 0) break;

            // Gradient
            std::array<float, Robot::n_spheres * 3> flat_grads;
            for(size_t k=0; k < Robot::n_spheres * 3; ++k) {
                flat_grads[k] = res.second[k].to_array()[0]; 
            }

            std::array<float, Robot::dimension> dq;
            Robot::d_collision_d_q(q_new, flat_grads, dq);
            
            float dq_norm = 0.0f;
            for(float v : dq) dq_norm += v*v;
            dq_norm = std::sqrt(dq_norm);

            if (dq_norm > 1e-6) {
                for(size_t k=0; k<Robot::dimension; ++k) {
                    q_new[k] += alpha * (dq[k] / dq_norm);
                }
            } else {
                break;
            }
            iter++;
        }

        auto end_t = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end_t - start_t);

        if (current_min_dist >= 0) {
            n_success++;
            total_time_ms += dur.count() / 1e6;
        }
    }

    std::cout << "Benchmark Results:" << std::endl;
    std::cout << "Total Samples: " << n_samples << std::endl;
    std::cout << "Initial Collisions: " << n_collisions << std::endl;
    std::cout << "Successful Projections: " << n_success << " (" << (n_collisions > 0 ? (100.0 * n_success / n_collisions) : 0.0) << "%)" << std::endl;
    if (n_success > 0) {
        std::cout << "Average Projection Time: " << (total_time_ms / n_success) << " ms" << std::endl;
    }

    return 0;
}
