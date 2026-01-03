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
};#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <expected>
#include <span>
#include <memory>

namespace foaml {

// Custom error types for type-safe error handling
enum class DataError {
    FileNotFound,
    CorruptedData,
    InvalidFormat,
    WriteFailure,
    ReadFailure
};

class DataManager {
public:
    // Modern constructor with std::filesystem::path
    explicit DataManager(std::filesystem::path data_dir = "data");
    
    // Prevent copying (manage resources carefully)
    DataManager(const DataManager&) = delete;
    DataManager& operator=(const DataManager&) = delete;
    
    // Allow moving
    DataManager(DataManager&&) noexcept = default;
    DataManager& operator=(DataManager&&) noexcept = default;
    
    // Save with versioning and validation
    [[nodiscard]] std::expected<std::filesystem::path, DataError> 
    save_simulation_data(
        std::span<const double> data,
        std::string_view filename,
        bool compress = false
    ) noexcept;
    
    // Load with validation
    [[nodiscard]] std::expected<std::vector<double>, DataError>
    load_simulation_data(std::string_view filename) noexcept;
    
    // Template for generic model saving (decouple from PyTorch)
    template<typename ModelType>
    [[nodiscard]] std::expected<std::filesystem::path, DataError>
    save_model(const ModelType& model, std::string_view filename) noexcept;
    
    template<typename ModelType>
    [[nodiscard]] std::expected<void, DataError>
    load_model(ModelType& model, std::string_view filename) noexcept;
    
    // Async I/O support for large datasets
    [[nodiscard]] std::future<std::expected<void, DataError>>
    save_async(std::span<const double> data, std::string_view filename);
    
private:
    std::filesystem::path data_dir_;
    static constexpr uint32_t FORMAT_VERSION = 1;
    
    // Helper for atomic writes (write to temp, then rename)
    [[nodiscard]] std::expected<void, DataError>
    atomic_write(std::span<const std::byte> data, const std::filesystem::path& path) noexcept;
    
    // Validation helpers
    [[nodiscard]] bool validate_checksum(const std::filesystem::path& path) const noexcept;
    uint32_t compute_crc32(std::span<const std::byte> data) const noexcept;
};

} // namespace foaml#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <expected>
#include <span>
#include <memory>

namespace foaml {

// Custom error types for type-safe error handling
enum class DataError {
    FileNotFound,
    CorruptedData,
    InvalidFormat,
    WriteFailure,
    ReadFailure
};

class DataManager {
public:
    // Modern constructor with std::filesystem::path
    explicit DataManager(std::filesystem::path data_dir = "data");
    
    // Prevent copying (manage resources carefully)
    DataManager(const DataManager&) = delete;
    DataManager& operator=(const DataManager&) = delete;
    
    // Allow moving
    DataManager(DataManager&&) noexcept = default;
    DataManager& operator=(DataManager&&) noexcept = default;
    
    // Save with versioning and validation
    [[nodiscard]] std::expected<std::filesystem::path, DataError> 
    save_simulation_data(
        std::span<const double> data,
        std::string_view filename,
        bool compress = false
    ) noexcept;
    
    // Load with validation
    [[nodiscard]] std::expected<std::vector<double>, DataError>
    load_simulation_data(std::string_view filename) noexcept;
    
    // Template for generic model saving (decouple from PyTorch)
    template<typename ModelType>
    [[nodiscard]] std::expected<std::filesystem::path, DataError>
    save_model(const ModelType& model, std::string_view filename) noexcept;
    
    template<typename ModelType>
    [[nodiscard]] std::expected<void, DataError>
    load_model(ModelType& model, std::string_view filename) noexcept;
    
    // Async I/O support for large datasets
    [[nodiscard]] std::future<std::expected<void, DataError>>
    save_async(std::span<const double> data, std::string_view filename);
    
private:
    std::filesystem::path data_dir_;
    static constexpr uint32_t FORMAT_VERSION = 1;
    
    // Helper for atomic writes (write to temp, then rename)
    [[nodiscard]] std::expected<void, DataError>
    atomic_write(std::span<const std::byte> data, const std::filesystem::path& path) noexcept;
    
    // Validation helpers
    [[nodiscard]] bool validate_checksum(const std::filesystem::path& path) const noexcept;
    uint32_t compute_crc32(std::span<const std::byte> data) const noexcept;
};

} // namespace foaml