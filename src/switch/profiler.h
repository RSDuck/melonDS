#pragma once

#include <stdint.h>
#include <assert.h>

namespace profiler
{

class Section
{
    const char* name_;

    uint64_t start_ = 0;

    bool registered_ = false;
    int hit_ = 0;
    uint64_t timeSpend_ = 0;
    int recursive_ = 0;
public:
    Section(const char* name);

    const char* Name()
    { return name_; }
    int Hit()
    { return hit_; }
    uint64_t TimeSpend()
    { return timeSpend_; }

    void Reset()
    {
        assert(recursive_ == 0);
        assert(start_ == 0);
        hit_ = 0;
        timeSpend_ = 0;
    }

    void Enter();
    void Leave();

    float History[32] = {0};
    int LastHit = 0;
};

void Frame();
void Render();

void EndSection();

}

/*#define PROFILER_SECTION(name) \
    static profiler::Section __profiling_##name(#name); \
    __profiling_##name.Enter();
#define PROFILER_END_SECTION \
    profiler::EndSection();*/
#define PROFILER_SECTION(name)
#define PROFILER_END_SECTION