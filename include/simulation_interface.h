#pragma once

#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <any>

class SimulationInterface {
public:
    virtual ~SimulationInterface() = default;
    
    // Abstract method for running a simulation
    // Using std::any to be flexible with return types (similar to Python's dynamic typing)
    virtual std::any simulate(const std::map<std::string, std::any>& params) = 0;
    
protected:
    // Helper method to get a typed parameter with error checking
    template<typename T>
    T getParam(const std::map<std::string, std::any>& params, const std::string& key) const {
        auto it = params.find(key);
        if (it == params.end()) {
            throw std::runtime_error("Required parameter not found: " + key);
        }
        
        try {
            return std::any_cast<T>(it->second);
        } catch (const std::bad_any_cast&) {
            throw std::runtime_error("Parameter type mismatch for: " + key);
        }
    }
};