cmake_minimum_required(VERSION 3.14)
project(foaml VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(meep REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)  # For JSON handling

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add the subdirectories
add_subdirectory(src)
add_subdirectory(api)

# Testing
option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()