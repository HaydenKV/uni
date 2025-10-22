/**
 * @file Event.h
 * @brief Defines the base Event class for the system.
 */
#ifndef EVENT_H
#define EVENT_H

#include <string>
#include "SystemBase.h"

/**
 * @class Event
 * @brief Base class for all events in the system.
 *
 * This class represents an abstract event that can be processed by the system.
 * Concrete event types should inherit from this class and implement the update method.
 */
class Event
{
public:
    /**
     * @brief Construct a new Event object.
     * @param time The time at which the event occurs.
     */
    Event(double time);

    /**
     * @brief Construct a new Event object.
     * @param time The time at which the event occurs.
     * @param verbosity The verbosity level for the event.
     */
    Event(double time, int verbosity);

    /**
     * @brief Destroy the Event object.
     */
    virtual ~Event();

    /**
     * @brief Process the event in the given system.
     * @param system The system in which to process the event.
     */
    void process(SystemBase & system);

protected:
    /**
     * @brief Update the system based on this event.
     * @param system The system to update.
     * 
     * This pure virtual function must be implemented by derived classes
     * to define the specific behavior of the event.
     */
    virtual void update(SystemBase & system) = 0;

    /**
     * @brief Get a string representation of the event processing.
     * @return A string describing the event processing.
     * 
     * This virtual function can be overridden by derived classes to provide
     * a custom string representation of the event processing.
     */
    virtual std::string getProcessString() const;

    double time_;       ///< The time at which the event occurs.
    int verbosity_;     ///< Verbosity level
};

#endif