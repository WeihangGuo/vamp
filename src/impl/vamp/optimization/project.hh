#pragma once

#include <cmath>
#include <array>
#include <vector>

#include <vamp/vector.hh>
#include <vamp/collision/environment.hh>

namespace vamp::optimization
{
    template <typename Robot, std::size_t rake = vamp::FloatVectorWidth>
    inline auto project_to_valid(
        const typename Robot::Configuration &c_in,
        const vamp::collision::Environment<vamp::FloatVector<rake>> &env,
        float alpha = 0.1f,
        int max_iters = 100) -> typename Robot::Configuration
    {
        // Copy to modify
        typename Robot::Configuration q_final = c_in;
        auto q_padded = q_final.to_array();

        std::array<float, Robot::dimension> q_new_arr;
        for (size_t i = 0; i < Robot::dimension; ++i)
        {
            q_new_arr[i] = q_padded[i];
        }

        // Compute bounds
        // Assuming physical values in q_new_arr, based on Robot::s_m (range) and Robot::s_a (lower bound)
        // lower = s_a
        // upper = s_a + s_m
        std::array<float, Robot::dimension> lower_bound;
        std::array<float, Robot::dimension> upper_bound;
        for (size_t k = 0; k < Robot::dimension; ++k)
        {
            lower_bound[k] = Robot::s_a[k];
            upper_bound[k] = Robot::s_a[k] + Robot::s_m[k];
        }

        // Initial check
        std::vector<float> q_vec(q_new_arr.begin(), q_new_arr.end());
        typename Robot::template ConfigurationBlock<rake> block(q_vec, true);

        // Boolean check: valid if both self-collision free AND environment collision free
        // fkcc returns true if VALID.
        if (Robot::template fkcc<rake>(env, block))
        {
            return c_in;
        }

        int iter = 0;
        while (iter < max_iters)
        {
            // Re-evaluate
            std::vector<float> q_v(q_new_arr.begin(), q_new_arr.end());
            typename Robot::template ConfigurationBlock<rake> b(q_v, true);

            // Early exit if valid (self + env)
            if (Robot::template fkcc<rake>(env, b))
            {
                break;
            }

            auto res = Robot::sdf_gradient(env, b);

            // Gradient
            // Original: flatten grads, then call d_collision_d_q
            // New: pass blocks directly
            typename Robot::template ConfigurationBlock<rake> dq_block;
            Robot::template d_collision_d_q<rake>(b, res.second, dq_block);

            // Extract gradient for update (scalar update)
            std::array<float, Robot::dimension> dq;
            for (size_t k = 0; k < Robot::dimension; ++k)
            {
                dq[k] = dq_block[k].element(0);
            }

            float dq_norm = 0.0f;
            for (float v : dq)
            {
                dq_norm += v * v;
            }
            dq_norm = std::sqrt(dq_norm);

            if (dq_norm > 1e-6f)
            {
                for (size_t k = 0; k < Robot::dimension; ++k)
                {
                    q_new_arr[k] += alpha * (dq[k] / dq_norm);

                    // Clamp to bounds
                    if (q_new_arr[k] < lower_bound[k])
                    {
                        q_new_arr[k] = lower_bound[k];
                    }
                    if (q_new_arr[k] > upper_bound[k])
                    {
                        q_new_arr[k] = upper_bound[k];
                    }
                }
            }
            else
            {
                break;
            }
            iter++;
        }

        // Convert back to Configuration
        // We can construct from array or padded array.
        // Configuration constructor takes std::array<ScalarT, num_scalars_rounded> usually?
        // Let's check vector.hh / interface.hh.
        // Helper often uses: Input::from(Configuration(q_final)); where q_final is vector<float>.
        // Robot::Configuration(std::array) exists.

        // Let's pad it back.
        // Actually Robot::Configuration has checking constructor?
        // Robot::Configuration is FloatVector.
        // Vector constructor: explicit Vector(const std::array<ScalarT, num_scalars_rounded> &data)

        return typename Robot::Configuration(q_new_arr);
    }
}  // namespace vamp::optimization
