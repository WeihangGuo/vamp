#include <vector>
#include <array>
#include <utility>
#include <iostream>
#include <iomanip>
#include <chrono>

#include <vamp/collision/factory.hh>
#include <vamp/robots/panda.hh>
#include <vamp/optimization/sdf.hh>
#include <vamp/random/halton.hh>

// Use Panda as in rrtc_example.cc
using Robot = vamp::robots::Panda;
static constexpr const std::size_t rake = vamp::FloatVectorWidth;
using EnvironmentInput = vamp::collision::Environment<float>;
using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;

// Environment Setup from rrtc_example.cc
static const std::vector<std::array<float, 3>> problem = {
    {0.55, 0, 0.25},
    {0.35, 0.35, 0.25},
    {0, 0.55, 0.25},
    {-0.55, 0, 0.25},
    {-0.35, -0.35, 0.25},
    {0, -0.55, 0.25},
    {0.35, -0.35, 0.25},
    {0.35, 0.35, 0.8},
    {0, 0.55, 0.8},
    {-0.35, 0.35, 0.8},
    {-0.55, 0, 0.8},
    {-0.35, -0.35, 0.8},
    {0, -0.55, 0.8},
    {0.35, -0.35, 0.8},
};

static constexpr float radius = 0.2;

// Benchmark Helper
template <typename Func>
double benchmark(std::string name, int iterations, Func func)
{
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func(i);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    double avg_time = duration.count() / iterations;
    std::cout << name << ": " << avg_time << " ms/iter (Total: " << duration.count() << " ms)" << std::endl;
    return avg_time;
}

auto main(int, char **) -> int
{
    std::cout << "Initializing Benchmark..." << std::endl;

    // 1. Build Environment
    EnvironmentInput environment;
    for (const auto &sphere : problem)
    {
        environment.spheres.emplace_back(vamp::collision::factory::sphere::array(sphere, radius));
    }
    environment.sort();
    auto env_v = EnvironmentVector(environment);

    // 2. Generate Random Configurations
    static constexpr int N_SAMPLES = 1000;
    std::vector<Robot::Configuration> configs;
    auto rng = std::make_shared<vamp::rng::Halton<Robot>>();
    
    // Discard first few samples (Halton)
    for(int i=0; i<100; ++i) rng->next();

    for(int i=0; i<N_SAMPLES; ++i) {
        // rng->next() returns Configuration (wrapped)
        configs.push_back(rng->next()); 
    }
    
    std::cout << "Generated " << N_SAMPLES << " random configurations." << std::endl;
    std::cout << "Running benchmarks with " << N_SAMPLES << " samples..." << std::endl;
    std::cout << "Note: Each sample is broadcasted to " << rake << " lanes for SIMD ops." << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    // 3. Benchmark: SDF Only
    benchmark("SDF Only", N_SAMPLES, [&](int idx) {
        // Broadcast single config to block
        Robot::ConfigurationBlock<rake> block;
        auto& cfg = configs[idx];
        for (size_t d = 0; d < Robot::dimension; ++d) {
             std::array<float, rake> row;
             row.fill(cfg.element(d));
             block[d] = Robot::ConfigurationBlock<rake>::RowT(row.data(), false);
        }
        
        auto dists = Robot::sdf(env_v, block);
        // Ensure not optimized away
        volatile float val = dists.to_array()[0];
        (void)val;
    });

    // 4. Benchmark: Solver (10 steps)
    benchmark("Solver (10 steps)", N_SAMPLES, [&](int idx) {
        auto valid_block = vamp::optimization::project_to_valid<Robot, rake>(
            configs[idx], 
            env_v, 
            10,    // steps
            0.5f,  // learning rate
            0.05f  // noise
        );
        volatile float val = valid_block[0].to_array()[0];
        (void)val;
    });

    // 5. Benchmark: Solver (100 steps)
    benchmark("Solver (100 steps)", N_SAMPLES, [&](int idx) {
        auto valid_block = vamp::optimization::project_to_valid<Robot, rake>(
            configs[idx], 
            env_v, 
            100,    // steps
            0.5f,  // learning rate
            0.05f  // noise
        );
        volatile float val = valid_block[0].to_array()[0];
        (void)val;
    });

    std::cout << "\n--------------------------------------------------" << std::endl;
    std::cout << "Convergence Analysis:" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    auto analyze_convergence = [&](std::string label, int steps) {
        int valid_lanes = 0;
        int total_lanes = 0;
        double total_sdf = 0.0;
        double min_sdf = 1e9;
        double max_sdf = -1e9;

        for(int idx = 0; idx < N_SAMPLES; ++idx) {
            Robot::ConfigurationBlock<rake> block;
            
            if (steps == -1) { // Raw samples (no noise, no solver)
                 auto& cfg = configs[idx];
                 for (size_t d = 0; d < Robot::dimension; ++d) {
                     std::array<float, rake> row;
                     row.fill(cfg.element(d));
                     block[d] = Robot::ConfigurationBlock<rake>::RowT(row.data(), false);
                 }
            } else {
                block = vamp::optimization::project_to_valid<Robot, rake>(
                    configs[idx], 
                    env_v, 
                    steps, 
                    0.5f, 
                    0.05f 
                );
            }

            auto dists = Robot::sdf(env_v, block);
            auto dists_arr = dists.to_array();

            for(float d : dists_arr) {
                if(d > 0) valid_lanes++;
                total_lanes++;
                total_sdf += d;
                if(d < min_sdf) min_sdf = d;
                if(d > max_sdf) max_sdf = d;
            }
        }

        double valid_rate = 100.0 * valid_lanes / total_lanes;
        double avg_sdf = total_sdf / total_lanes;
        
        std::cout << std::left << std::setw(20) << label 
                  << " | Valid Rate: " << std::fixed << std::setprecision(1) << valid_rate << "%"
                  << " | Avg SDF: " << std::setprecision(4) << avg_sdf 
                  << " | Range: [" << min_sdf << ", " << max_sdf << "]" << std::endl;
    };

    analyze_convergence("Initial (Raw)", -1);
    analyze_convergence("Solver (10 steps)", 10);
    analyze_convergence("Solver (100 steps)", 100);

    return 0;
}
