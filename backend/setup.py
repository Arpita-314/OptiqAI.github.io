cmake_minimum_required(VERSION 3.4...3.18)
project(fdtd_core)

find_package(pybind11 REQUIRED)
pybind11_add_module(fdtd_core fdtd_core.cpp)