#include <vamp/robots/panda.hh>

using Robot = vamp::robots::Panda;
static constexpr const std::size_t rake = vamp::FloatVectorWidth;

inline static void d()
{
    std::array<vamp::FloatVector<rake, 1>, 232> x;
    std::array<vamp::FloatVector<rake, 1>, 7> y;
    vamp::FloatVector<rake, 1> a;
    a[0] = 5;
    Robot::ConfigurationBlock<rake> q;
    for (int i = 0; i < 7; ++i)
    {
        q[i] = i;
    }

    y[0] = 2;
    x[0] = y[0] + 1;
    x[1] = q[1];
    x[2] = a[0];
    x[3] = y[0] * x[0];
    std::cout << x[0] << std::endl;
    std::cout << x[1] << std::endl;
    std::cout << x[2] << std::endl;
    std::cout << x[3] << std::endl;
}

int main()
{
    d();

    return 0;
}
