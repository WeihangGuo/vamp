#include <vector>
#include <array>
#include <utility>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

#include <vamp/collision/factory.hh>
#include <vamp/robots/panda.hh>

// Use Panda since we added the SDF method to it
using Robot = vamp::robots::Panda;
static constexpr const std::size_t rake = vamp::FloatVectorWidth;
using EnvironmentInput = vamp::collision::Environment<float>;
using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;

// Spheres for the environment - (x, y, z) center coordinates
static const std::vector<std::array<float, 3>> obstacles = {
    {0.55, 0, 0.25},
    {0.35, 0.35, 0.25},
    {0, 0.55, 0.25},
    {-0.55, 0, 0.25},
    {-0.35, -0.35, 0.25},
    {0, -0.55, 0.25},
    {0.35, -0.35, 0.25},
    {-0.55, 0.55, 0.25},
    
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

auto read_configs(const std::string& filename) -> std::vector<Robot::ConfigurationArray>
{
    std::vector<Robot::ConfigurationArray> configs;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return configs;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        Robot::ConfigurationArray config;
        bool valid = true;
        for (size_t i = 0; i < Robot::dimension; ++i) {
            if (!(ss >> config[i])) {
                valid = false; 
                break;
            }
        }
        
        if (valid) {
            configs.push_back(config);
        }
    }
    return configs;
}

auto main(int, char **) -> int
{
    // Build environment
    EnvironmentInput environment;
    for (const auto &sphere : obstacles)
    {
        environment.spheres.emplace_back(vamp::collision::factory::sphere::array(sphere, radius));
    }

    // Sort environment for efficiency
    environment.sort();
    auto env_v = EnvironmentVector(environment);

    // Read configurations
    std::string config_file = "scripts/cpp/configs.txt";
    // Check if path exists relative to build dir or source
    // Assuming run from project root, path is correct. 
    // If run from build, it might be ../scripts/cpp/configs.txt
    // But user command is usually ./build/exe from root.
    
    auto all_configs = read_configs(config_file);
    
    if (all_configs.empty()) {
        std::cerr << "No configurations loaded from " << config_file << std::endl;
        return 1;
    }

    // Open output file
    std::string output_file = "scripts/cpp/sdf_results.txt";
    std::ofstream out_stream(output_file);
    if (!out_stream.is_open()) {
        std::cerr << "Error: Could not open output file " << output_file << std::endl;
        return 1;
    }

    std::cout << "Writing results to " << output_file << "..." << std::endl;

    // Process in batches of 'rake'
    size_t num_configs = all_configs.size();
    for (size_t batch_idx = 0; batch_idx < num_configs; batch_idx += rake)
    {
        Robot::ConfigurationBlock<rake> q;
        
        // Fill the batch (by dimension first, then lane)
        for (size_t j = 0; j < Robot::dimension; ++j)
        {
            std::array<float, rake> joint_vals;
            for (size_t lane = 0; lane < rake; ++lane)
            {
                size_t config_idx = batch_idx + lane;
                if (config_idx < num_configs) {
                    joint_vals[lane] = all_configs[config_idx][j];
                } else {
                    joint_vals[lane] = 0.0f; // Padding
                }
            }
            q[j] = Robot::ConfigurationBlock<rake>::RowT(joint_vals);
        }

        // 1. Compute Forward Kinematics to get sphere positions
        Robot::Spheres<rake> spheres;
        Robot::sphere_fk(q, spheres);

        // 2. Compute Signed Distance Field
        auto dists = Robot::sdf(env_v, spheres);
        std::array<float, rake> dists_arr = dists.to_array();

        // Output results for valid configs in this batch
        for (size_t lane = 0; lane < rake; ++lane)
        {
             size_t config_idx = batch_idx + lane;
             if (config_idx >= num_configs) break;

             // Write to file: ConfigIndex <space> MinSDF <space> Sphere0_X Sphere0_Y Sphere0_Z Sphere0_R ...
             out_stream << config_idx << " " << std::fixed << std::setprecision(6) << dists_arr[lane];

             // Output sphere poses and radii for this config (lane)
             // Spheres struct has FloatVector x, y, z, r which are arrays of vectors
             // We need to access element 'lane' of each sphere 'i'
             
             // Extract sphere data for the current lane
             for (size_t i = 0; i < Robot::n_spheres; ++i) {
                 float sx = spheres.x[i][std::pair(0, lane)]; // Row 0, Col 'lane' logic
                 float sy = spheres.y[i][std::pair(0, lane)];
                 float sz = spheres.z[i][std::pair(0, lane)];
                 float sr = spheres.r[i][std::pair(0, lane)];
                 out_stream << " " << sx << " " << sy << " " << sz << " " << sr;
             }
             
             out_stream << std::endl;
        }
    }
    
    out_stream.close();
    std::cout << "Done." << std::endl;

    return 0;
}
