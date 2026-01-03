#include "meep_simulation.h"
#include <stdexcept>
#include <vector>

MEEPSimulation::MEEPSimulation() = default;
MEEPSimulation::~MEEPSimulation() = default;

std::any MEEPSimulation::simulate(const std::map<std::string, std::any>& params) {
    try {
        // Extract parameters - notice the C++ type safety compared to Python
        double wavelength = getParam<double>(params, "wavelength");
        double resolution = getParam<double>(params, "resolution");
        double size_x = getParam<double>(params, "size_x");
        double size_y = getParam<double>(params, "size_y");
        
        // Define the computational cell
        meep::grid_volume vol = meep::vol2d(size_x, size_y, resolution);
        
        // Setup the structure
        meep::structure s(vol, meep::medium(3.0));
        
        // Setup the fields
        meep::fields f(&s);
        
        // Add a Gaussian source
        meep::vec src_pos = meep::vec(-0.5 * size_x + 0.5, 0.0);
        meep::continuous_src_time src(wavelength);
        f.add_point_source(meep::Ez, src, src_pos);
        
        // Add a block of material (similar to the Python version)
        meep::geometric_object block = meep::make_block(
            meep::vec(0.2 * size_x, size_y, meep::inf),
            meep::vec(0, 0, 0),
            meep::medium(3.0)
        );
        s.add_geometry(block);
        
        // Run the simulation
        while (f.time() < 200) {
            f.step();
        }
        
        // Get the field data at a specific point
        meep::vec point(0, 0, 0);
        double ez_data = f.get_field(meep::Ez, point);
        
        // In practice, you'd probably want to return a 2D field or a vector of values
        // This is simplified for demonstration
        std::vector<double> result = {ez_data};
        return result;
        
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("MEEP Simulation Failed: ") + e.what());
    }
}