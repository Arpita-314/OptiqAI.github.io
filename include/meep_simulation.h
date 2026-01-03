#pragma once

#include "simulation_interface.h"
#include <meep.hpp>
#include <vector>

class MEEPSimulation : public SimulationInterface {
public:
    MEEPSimulation();
    ~MEEPSimulation() override;
    
    // Implementation of the simulate method
    std::any simulate(const std::map<std::string, std::any>& params) override;
};