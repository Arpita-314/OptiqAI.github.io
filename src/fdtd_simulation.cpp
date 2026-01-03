#include "fdtd_simulation.h"
#include <stdexcept>

FDTDSimulation::FDTDSimulation(const std::string& device_str) 
    : device(device_str == "cuda" && torch::cuda::is_available() ? 
             torch::kCUDA : torch::kCPU) {}

FDTDSimulation::~FDTDSimulation() = default;

std::any FDTDSimulation::simulate(const std::map<std::string, std::any>& params) {
    try {
        // Extract parameters
        double wavelength_val = getParam<double>(params, "wavelength");
        int grid_size_val = getParam<int>(params, "grid_size");
        
        // Convert to tensors
        auto wavelength = torch::tensor(wavelength_val, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        auto grid_size = torch::tensor(grid_size_val, torch::TensorOptions().dtype(torch::kInt32).device(device));
        
        // Dummy calculation (replace with your actual FDTD implementation)
        auto result = wavelength * grid_size;
        
        // Convert result to vector
        std::vector<float> output;
        output.push_back(result.item<float>());
        
        return output;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("FDTD Simulation Failed: ") + e.what());
    }
}