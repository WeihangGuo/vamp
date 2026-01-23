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

    return 0;
}
