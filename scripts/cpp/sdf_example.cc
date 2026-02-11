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
    int n_samples = 1;
    int n_success = 0;
    double total_time_ms = 0.0;
    int total_iter = 0;
    std::cout << "Starting Benchmark with " << n_samples << " samples..." << std::endl;
    int i = 0;
    while (i < n_samples)
    {
        auto q_random = sampler.next();
        std::cout << "Sample " << i << ": " << q_random << std::endl;
        Robot::ConfigurationBlock<rake> b;
        for (auto k = 0U; k < Robot::dimension; ++k)
        {
            b[k] = q_random.broadcast(k);
        }
        auto valid = Robot::fkcc<rake>(env_v, b);
        if (valid)
        {
            continue;
        }
        i++;

        // Project
        auto start_t = std::chrono::high_resolution_clock::now();

        float current_min_dist = -1e9f;
        int iter = 0;
        const int max_iters = 1;
        float alpha = 0.1f;

        while (current_min_dist < 0 && iter < max_iters)
        {
            // Re-evaluate
            auto res = Robot::sdf_gradient(env_v, b);
            std::cout << "b: " << b << std::endl;
            // std::cout << "res: " << res.first << std::endl;
            // std::cout << "grad: " << res.second << std::endl;
            auto dists_arr = res.first.to_array();
            current_min_dist = 1e9f;
            for (auto d : dists_arr)
            {
                if (d < current_min_dist)
                {
                    current_min_dist = d;
                }
            }

            if (current_min_dist >= 0)
            {
                break;
            }

            // Gradient
            // Original: flatten grads, then call d_collision_d_q
            // New: pass blocks directly
            Robot::ConfigurationBlock<rake> dq_block;
            // std::cout << "res.second: " << res.second << std::endl;
            Robot::d_collision_d_q(b, res.second, dq_block);  // b is already the block for q_new
            std::cout << "dq_block: " << dq_block << std::endl;
            std::vector<float> dq(Robot::dimension);
            for (auto k = 0U; k < Robot::dimension; ++k)
            {
                dq[k] = dq_block[k].element(0);
            }

            std::printf(
                "dq: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
                dq[0],
                dq[1],
                dq[2],
                dq[3],
                dq[4],
                dq[5],
                dq[6]);
            float dq_norm = 0.0f;
            for (float v : dq)
            {
                dq_norm += v * v;
            }
            dq_norm = std::sqrt(dq_norm);

            if (dq_norm > std::numeric_limits<float>::epsilon())
            {
                std::cout << "dq_norm: " << dq_norm << std::endl;
                for (auto k = 0U; k < Robot::dimension; ++k)
                {
                    // b[k] += alpha * (dq[k] / dq_norm);
                    b[k] = b[k] + alpha * (dq[k] / dq_norm);
                }
            }
            else
            {
                break;
            }
            iter++;
        }
        std::cout << "Projection Iterations: " << iter << std::endl;
        auto end_t = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end_t - start_t);
        // std::cout << "Projection Time: " << dur.count() / 1e6 << " ms" << std::endl;
        auto b_final = b;
        std::vector<float> b_final_vec(Robot::dimension);
        for (size_t k = 0; k < Robot::dimension; ++k)
        {
            b_final_vec[k] = b_final[k].element(0);
        }
        std::printf(
            "Final Configuration: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
            b_final_vec[0],
            b_final_vec[1],
            b_final_vec[2],
            b_final_vec[3],
            b_final_vec[4],
            b_final_vec[5],
            b_final_vec[6]);
        valid = Robot::fkcc<rake>(env_v, b_final);
        std::printf("-------\n");
        if (valid)
        {
            n_success++;
            total_time_ms += dur.count() / 1e6;
            total_iter += iter;
        }
    }

    std::cout << "Benchmark Results:" << std::endl;
    std::cout << "Total Samples: " << n_samples << std::endl;
    std::cout << "Successful Projections: " << n_success << " ("
              << (n_samples > 0 ? (100.0 * n_success / n_samples) : 0.0) << "%)" << std::endl;
    if (n_success > 0)
    {
        std::cout << "Average Projection Time: " << (total_time_ms / n_success) << " ms" << std::endl;
        std::cout << "Average Iterations: " << (total_iter / n_success) << std::endl;
    }

    return 0;
}
