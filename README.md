# DerivativesTools

A comprehensive C++ library for financial derivatives modeling and simulation. The library supports multiple stochastic processes and jump-diffusion models, allowing for flexible and accurate asset price simulations. It is designed to be reusable and easily integrated into larger projects.

---

## Features
- **Models**: Support for a variety of models, including:
    - Black-Scholes
    - Heston
    - Kou
    - CGMY
    - Variance Gamma
    - Bates
    - Merton
    - Non-Inverse Gaussian (NIG)
- **Simulation**: Generate realistic price paths based on specified model parameters.
- **Reusable Library**: Can be integrated into other projects or used standalone.
- **C++ Standard**: Compatible with C++14 and higher.

---

## Installation

### Prerequisites
- **CMake**: Version 3.20 or higher
- **C++ Compiler**: Supporting at least C++14 (e.g., GCC, Clang, MSVC)

### Steps to Build and Install

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/DerivativesTools.git
    cd DerivativesTools
    ```

2. Build the library:
    ```bash
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make
    ```

3. Install the package (optional):
    ```bash
    sudo make install
    ```

4. To use the package in another project:
    - Add the following to your `CMakeLists.txt`:
      ```cmake
      find_package(DerivativesTools REQUIRED)
      target_link_libraries(MyApp PRIVATE DerivativesTools::DerivativesTools)
      ```

---

## Models Description

### 1. **Black-Scholes**
A classic continuous-time stochastic process model for pricing options. Assumes:
- Log-normal distribution of asset prices.
- Constant volatility and risk-free rate.

#### Parameters:
- Spot price \( S_0 \)
- Risk-free rate \( r \)
- Volatility \( \sigma \)

---

### 2. **Heston Model**
A stochastic volatility model that introduces randomness to volatility:
$$
dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^S
$$
$$
dv_t = \kappa (\theta - v_t) dt + \xi \sqrt{v_t} dW_t^v
$$

#### Parameters:
- Spot price \( S_0 \)
- Long-run variance \( \theta \)
- Mean reversion speed \( \kappa \)
- Volatility of volatility \( \xi \)
- Correlation \( \rho \)

---

### 3. **Kou Model**
A double-exponential jump-diffusion model for capturing asymmetrical jumps:
$$
dS_t = \mu S_t dt + \sigma S_t dW_t + S_t dJ_t
$$

#### Parameters:
- Spot price \( S_0 \)
- Jump intensity \( \lambda \)
- Positive and negative jump rates \( p, q \)

---

### 4. **CGMY Model**
A pure jump Lévy process with flexibility in skewness and tail heaviness:
$$
\psi(u) = C \cdot \Gamma(-Y) \cdot [(M - iu)^Y - M^Y + (G + iu)^Y - G^Y]
$$

#### Parameters:
- \( C \): Jump intensity
- \( G, M \): Controls the steepness of positive and negative jumps
- \( Y \): Controls tail heaviness

---

### 5. **Variance Gamma (VG) Model**
A pure jump model where returns follow a gamma process:
$$
dS_t = S_t ( \mu dt + \sigma d\Gamma_t )
$$

#### Parameters:
- Drift \( \mu \)
- Volatility \( \sigma \)
- Variance \( \nu \)

---

### 6. **Bates Model**
Extends the Heston model with jumps:
$$
dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t + S_t dJ_t
$$

---

### 7. **Merton Model**
A jump-diffusion model with normal jump sizes:
$$
dS_t = \mu S_t dt + \sigma S_t dW_t + J_t S_t
$$

---

### 8. **Non-Inverse Gaussian (NIG)**
A Lévy process characterized by inverse Gaussian distributions of jumps.

---

## How to Simulate

Each model provides a `simulate(int nbSteps)` method to generate a price path.

### Example Code: CGMY Model
```cpp
#include <iostream>
#include "DerivativesTools/models.hpp"

int main() {
    // Initialize the CGMY model with parameters:
    // Spot price = 100, Risk-free rate = 5%, Maturity = 0.26 years, Dividends = 1.5%
    // C = 0.2, G = 3, M = 2, Y = 0.5
    CGMYModel model(100, 0.05, 0.26, 0.015, 0.2, 3, 2, 0.5);

    // Display model parameters
    model.showModel();

    // Simulate a price path with 252 time steps (daily steps for ~1 year)
    auto path = model.simulate(252);

    // Print the simulated path
    std::cout << "Simulated Path:" << std::endl;
    for (double price : path) {
        std::cout << price << std::endl;
    }

    return 0;
}
