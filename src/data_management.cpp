#include "data_management.h"
#include <fstream>
#include <stdexcept>
#include <filesystem>

namespace fs = std::filesystem;

DataManager::DataManager(const std::string& data_dir) : data_dir(data_dir) {
    // Create directory if it doesn't exist
    if (!fs::exists(data_dir)) {
        fs::create_directories(data_dir);
    }
}

std::string DataManager::save_simulation_data(const std::vector<double>& data, const std::string& filename) {
    std::string filepath = data_dir + "/" + filename;
    
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    
    // Write the size of the vector
    size_t size = data.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    
    // Write the data
    file.write(reinterpret_cast<const char*>(data.data()), size * sizeof(double));
    
    return filepath;
}

std::vector<double> DataManager::load_simulation_data(const std::string& filename) {
    std::string filepath = data_dir + "/" + filename;
    
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }
    
    // Read the size of the vector
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    
    // Read the data
    std::vector<double> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(double));
    
    return data;
}

std::string DataManager::save_model(const torch::nn::Module& model, const std::string& filename) {
    std::string filepath = data_dir + "/" + filename;
    
    torch::save(model, filepath);
    
    return filepath;
}

void DataManager::load_model(torch::nn::Module& model, const std::string& filename) {
    std::string filepath = data_dir + "/" + filename;
    
    if (!fs::exists(filepath)) {
        throw std::runtime_error("Model file not found: " + filepath);
    }
    
    torch::load(model, filepath);
}