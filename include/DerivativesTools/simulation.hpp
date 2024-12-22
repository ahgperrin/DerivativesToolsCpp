//
// Created by Antoine Perrin on 12/9/24.
//

#ifndef DERIVATIVESTOOLS_SIMULATION_HPP
#define DERIVATIVESTOOLS_SIMULATION_HPP

#include <vector>
#include <future>
#include <thread>

template<typename ModelType>
std::vector<std::vector<double>> simulatePaths(ModelType &model, int nbSim, int nbSteps) {
    std::vector<std::vector<double>> Sims(nbSim);

    auto simulateSinglePath = [&model, nbSteps]() {
        return model.simulate(nbSteps);
    };

    std::vector<std::future<std::vector<double>>> futures;

    for (int i = 0; i < nbSim; ++i) {
        futures.push_back(std::async(std::launch::async, simulateSinglePath));
    }

    for (int i = 0; i < nbSim; ++i) {
        Sims[i] = futures[i].get();
    }

    return Sims;
};


#endif //DERIVATIVESTOOLS_SIMULATION_HPP
