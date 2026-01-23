#pragma once

#include <vamp/vector.hh>
#include <vamp/collision/environment.hh>
#include <random>
#include <cmath>
#include <iostream>

namespace vamp::optimization
{
    // Computes Gradient of SDF via Central Difference
    template <typename Robot, std::size_t rake>
    inline auto compute_gradient(
        const collision::Environment<FloatVector<rake>> &environment,
        const typename Robot::template ConfigurationBlock<rake> &state,
        float h = 1e-4f) noexcept -> typename Robot::template ConfigurationBlock<rake>
    {
        using ConfigBlock = typename Robot::template ConfigurationBlock<rake>;
        
        ConfigBlock grad;
        auto h_vec = FloatVector<rake>::fill(h);
        auto inv_2h = FloatVector<rake>::fill(1.0f / (2.0f * h));

        // Create a mutable copy of state to perturb
        ConfigBlock perturbed = state;

        for (std::size_t i = 0; i < Robot::dimension; ++i)
        {
            auto original_val = perturbed[i];

            // f(x + h)
            perturbed[i] = original_val + h_vec;
            auto f_plus = Robot::sdf(environment, perturbed);

            // f(x - h)
            perturbed[i] = original_val - h_vec;
            auto f_minus = Robot::sdf(environment, perturbed);

            // Restore original
            perturbed[i] = original_val;

            // g = (f_plus - f_minus) / 2h
            grad[i] = (f_plus - f_minus) * inv_2h;
        }
        return grad;
    }

    // Projects a single state to multiple valid candidates.
    template <typename Robot, std::size_t rake = FloatVectorWidth>
    inline auto project_to_valid(
        const typename Robot::Configuration &start_state,
        const collision::Environment<FloatVector<rake>> &environment,
        int steps = 100,
        float learning_rate = 0.5f,
        float noise_scale = 0.1f) noexcept -> typename Robot::template ConfigurationBlock<rake>
    {
        using ConfigBlock = typename Robot::template ConfigurationBlock<rake>;
        using Vector = FloatVector<rake>;

        ConfigBlock current_state;

        // 1. Initialization: Broadcast and Add Noise
        // seeding with a fixed value for reproducibility, or use std::random_device{}
        static std::random_device gen; 
        std::uniform_real_distribution<float> dist_noise(-noise_scale, noise_scale);

        for (std::size_t i = 0; i < Robot::dimension; ++i)
        {
            // Create an array for the row
            std::array<float, rake> noise_vals;
            for (std::size_t k = 0; k < rake; ++k)
            {
                noise_vals[k] = start_state.element(i) + dist_noise(gen);
            }
            // Load into the SIMD vector for dimension i
            current_state[i] = Vector(noise_vals.data(), false);
        }

        auto zero = Vector::fill(0.0f);
        auto lr = Vector::fill(learning_rate);

        for (int step = 0; step < steps; ++step)
        {
            // Calculate SDF
            auto dist = Robot::sdf(environment, current_state);

            // Calculate Gradient
            auto grad = compute_gradient<Robot, rake>(environment, current_state);

            // Update Rule
            auto neg_dist = -dist;
            auto relu_neg_dist = neg_dist.max(zero); 
            auto magnitude = lr * relu_neg_dist;

            // Apply update per dimension
            for (std::size_t i = 0; i < Robot::dimension; ++i)
            {
                auto delta = grad[i] * magnitude;
                current_state[i] = current_state[i] + delta;
            }
        }

        return current_state;
    }
}
