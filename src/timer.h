#include <chrono>
#include <atomic>

template <typename Clock = std::chrono::high_resolution_clock>
class Stopwatch
{
    const typename Clock::time_point start_point;
public:
    Stopwatch() :
        start_point(Clock::now())
    {}

    template <typename Rep = typename Clock::duration::rep, typename Units = typename Clock::duration>
    Rep elapsed_time() const
    {
        std::atomic_thread_fence(std::memory_order_relaxed);
        auto counted_time = std::chrono::duration_cast<Units>(Clock::now() - start_point).count();
        std::atomic_thread_fence(std::memory_order_relaxed);
        return static_cast<Rep>(counted_time);
    }
};

using precise_stopwatch = Stopwatch<>;
using system_stopwatch = Stopwatch<std::chrono::system_clock>;
using monotonic_stopwatch = Stopwatch<std::chrono::steady_clock>;