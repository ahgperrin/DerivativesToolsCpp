#include <iostream>
#include "include/DerivativesTools/models.hpp"
#include "include/DerivativesTools/simulation.hpp"
#include <chrono>
#include <fstream>

void exportToCSV(const std::vector<std::vector<double>> &paths, const std::string &filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // Iterate over each path and write it to the file
    for (const auto &path: paths) {
        for (size_t i = 0; i < path.size(); ++i) {
            file << path[i];
            if (i < path.size() - 1) // Avoid trailing comma
                file << ",";
        }
        file << "\n"; // End of row
    }

    file.close();
    std::cout << "Exported paths to " << filename << std::endl;
}

int main() {
    std::cout << "Hello, World!" << std::endl;

    HestonModel hestonModel(100, 0.05, 1, 0.02, 3, 0.04, 0.5, -0.7, 0.04);
    hestonModel.showModel();


    BlackScholesModel bsm(100, 0.05, 1, 0.015, 0.2);
    bsm.showModel();

    VarGammaModel vg(100, 0.05, 1, 0.02, 0.2, -0.1, 0.1);
    vg.showModel();

    MertonModel merton(100, 0.05, 1, 0.015, 0.2, 1, 0.02, 0.20);
    merton.showModel();

    BatesModel bates(100, 0.05, 1, 0.015, 1.3, 0.15, 0.6, -0.7, 0.15, 0.12, 1);
    bates.showModel();

    NigModel nig(100, 0.05, 0.26, 0.015, 0.5, -0.2, 1.5);
    nig.showModel();

    KouModel kou(100, 0.05, 1, 0.02, 0.2, 0.05, 0.6, -0.07, 0.06);
    kou.showModel();


    CGMYModel cgmy(100, 0.05, 0.26, 0.015, 0.8, 3, 3, 0.5);
    cgmy.showModel();

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> simulationVG = simulatePaths(cgmy, 20, 252);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;


    exportToCSV(simulationVG, "/Users/antoineperrin/Desktop/vg_simulation_paths.csv");

    return 0;

}
