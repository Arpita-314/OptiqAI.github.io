#pragma once

#include "simulation_interface.h"
#include <torch/torch.h>
#include <string>

class FDTDSimulation : public SimulationInterface {
public:
    explicit FDTDSimulation(const std::string& device = "cpu");
    ~FDTDSimulation() override;
    
    std::any simulate(const std::map<std::string, std::any>& params) override;
    
private:
    torch::Device device;
};