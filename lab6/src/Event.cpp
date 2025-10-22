#include <print>
#include <string>
#include "SystemBase.h"
#include "Event.h"

Event::Event(double time)
    : time_(time)
    , verbosity_(1)
{}

Event::Event(double time, int verbosity)
    : time_(time)
    , verbosity_(verbosity)
{}

Event::~Event() = default;

void Event::process(SystemBase & system)
{
    if (verbosity_ > 0)
    {
        std::print("[t={:07.3f}s] {}", time_, getProcessString());
    }

    // Time update
    system.predict(time_);

    // Event-specific implementation
    update(system);

    if (verbosity_ > 0)
    {
        std::println(" done");
    }
}

std::string Event::getProcessString() const
{
    return "Processing event:";
}