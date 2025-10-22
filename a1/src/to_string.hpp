#ifndef TO_STRING_HPP
#define TO_STRING_HPP

#include <string>
#include <ostream>
#include <sstream>
#include <concepts>

// Helper template to convert a stream into a string

template <typename T>
concept StreamInsertable = requires(std::ostream & os, const T & obj)
{
    os << obj;
};

template <StreamInsertable T>
std::string to_string(const T & obj)
{
    std::ostringstream oss;
    oss << obj;
    return oss.str();
}

#endif
