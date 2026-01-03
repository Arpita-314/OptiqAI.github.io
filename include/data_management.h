#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>

class DataManager {
public:
    explicit DataManager(const std::string& data_dir = "data");
    
    // Save simulation data (vector)
    std::string save_simulation_data(const std::vector<double>& data, const std::string& filename = "simulation_data.bin");
    
    // Load simulation data (vector)
    std::vector<double> load_simulation_data(const std::string& filename = "simulation_data.bin");
    
    // Save PyTorch model
    std::string save_model(const torch::nn::Module& model, const std::string& filename = "model.pt");
    
    // Load PyTorch model
    void load_model(torch::nn::Module& model, const std::string& filename = "model.pt");
    
private:
    std::string data_dir;
};