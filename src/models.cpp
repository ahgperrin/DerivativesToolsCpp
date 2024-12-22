//
// Created by Antoine Perrin on 12/8/24.
//

#include "../include/DerivativesTools/models.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <thread>


// GenericModel Class
GenericModel::GenericModel(double spotZero, double rfrRate, double ttMaturity, double divRate) :
        m_spotZero(spotZero), m_ttMaturity(ttMaturity), m_rfrRate(rfrRate), m_divRate(divRate) {};

GenericModel::GenericModel(const GenericModel &p) :
        m_spotZero(p.m_spotZero), m_ttMaturity(p.m_ttMaturity), m_rfrRate(p.m_rfrRate), m_divRate(p.m_divRate) {};

GenericModel::~GenericModel() {};

GenericModel &GenericModel::operator=(const GenericModel &p) {
    if (this != &p) {
        m_divRate = p.m_divRate;
        m_rfrRate = p.m_rfrRate;
        m_ttMaturity = p.m_ttMaturity;
        m_spotZero = p.m_spotZero;
    }
    return *this;
}

double GenericModel::getDivRate() const { return m_divRate; };

double GenericModel::getRfrRate() const { return m_rfrRate; };

double GenericModel::getSpotZero() const { return m_spotZero; };

double GenericModel::getTtMaturity() const { return m_ttMaturity; };

// HestonModel Class

HestonModel::HestonModel(double spotZero, double rfrRate, double ttMaturity, double divRate, double kappa, double theta,
                         double volVol, double rho, double volZero) :
        GenericModel(spotZero, rfrRate, ttMaturity, divRate), m_kappa(kappa), m_theta(theta), m_rho(rho),
        m_volVol(volVol), m_volZero(volZero) {};

HestonModel::HestonModel(const HestonModel &p) :
        GenericModel(p), m_kappa(p.m_kappa), m_theta(p.m_theta), m_rho(p.m_rho),
        m_volVol(p.m_volVol), m_volZero(p.m_volZero) {};

HestonModel::~HestonModel() {};

HestonModel &HestonModel::operator=(const HestonModel &p) {
    if (this != &p) {
        GenericModel::operator=(p);
        m_kappa = p.m_kappa;
        m_theta = p.m_theta;
        m_volVol = p.m_volVol;
        m_rho = p.m_rho;
        m_volZero = p.m_volZero;
    }
    return *this;
}

std::vector<double> HestonModel::simulate(int nbSteps) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, 1.0);

    std::vector<double> spotPath(nbSteps, 0);
    std::vector<double> volPath(nbSteps, 0);
    spotPath[0] = getSpotZero();
    volPath[0] = m_volZero;
    double divider = getTtMaturity() / nbSteps;

    double drift = (getRfrRate() - getDivRate());
    for (int i = 1; i < nbSteps; i++) {
        double dW = d(gen);
        double dZ = d(gen);

        double corr = m_rho * dW + sqrt(1 - m_rho * m_rho) * dZ;
        volPath[i] = std::max(0.0, volPath[i - 1] + m_kappa * (m_theta - volPath[i - 1]) * divider +
                                   m_volVol * sqrt(volPath[i - 1]) * sqrt(divider) * corr);

        // Correcting spot price update
        spotPath[i] = spotPath[i - 1] *
                      exp((drift - 0.5 * volPath[i - 1]) * divider + sqrt(volPath[i - 1]) * sqrt(divider) * dW);
    }
    return spotPath;
}

void HestonModel::showModel() {
    std::cout << "Heston Model Parameters:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Spot Price (Spot Zero):    " << getSpotZero() << std::endl;
    std::cout << "Risk-Free Rate (rfrRate):  " << getRfrRate() << std::endl;
    std::cout << "Time to Maturity:          " << getTtMaturity() << " years" << std::endl;
    std::cout << "Dividend Rate (divRate):   " << getDivRate() << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Heston Specific Parameters:" << std::endl;
    std::cout << "Mean Reversion Speed (kappa): " << m_kappa << std::endl;
    std::cout << "Long-Term Variance (theta):   " << m_theta << std::endl;
    std::cout << "Volatility of Volatility:     " << m_volVol << std::endl;
    std::cout << "Correlation (rho):            " << m_rho << std::endl;
    std::cout << "Initial Variance (volZero):   " << m_volZero << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

// BlackScholes Class

BlackScholesModel::BlackScholesModel(double spotZero, double rfrRate, double ttMaturity, double divRate, double sigma)
        : GenericModel(spotZero, rfrRate, ttMaturity, divRate), m_sigma(sigma) {}

BlackScholesModel::~BlackScholesModel() {};

BlackScholesModel::BlackScholesModel(const BlackScholesModel &p) : GenericModel(p), m_sigma(p.m_sigma) {};

BlackScholesModel &BlackScholesModel::operator=(const BlackScholesModel &p) {
    if (this != &p) {
        GenericModel::operator=(p);
        m_sigma = p.m_sigma;
    }
    return *this;
}

std::vector<double> BlackScholesModel::simulate(int nbSteps) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, 1.0);

    std::vector<double> spotPath(nbSteps, 0);
    spotPath[0] = getSpotZero();
    double divider = getTtMaturity() / nbSteps;
    double drift = ((getRfrRate() - getDivRate()) - (pow(m_sigma, 2) / 2)) * divider;
    for (int i = 1; i < nbSteps; i++) {
        double dW = m_sigma * sqrt(divider) * d(gen);
        spotPath[i] = (exp(drift + dW) * spotPath[i - 1]);
    }
    return spotPath;
}

void BlackScholesModel::showModel() {
    std::cout << "Black Scholes Model Parameters:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Spot Price (Spot Zero):    " << getSpotZero() << std::endl;
    std::cout << "Risk-Free Rate (rfrRate):  " << getRfrRate() << std::endl;
    std::cout << "Time to Maturity:          " << getTtMaturity() << " years" << std::endl;
    std::cout << "Dividend Rate (divRate):   " << getDivRate() << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Black Scholes Specific Parameters:    " << std::endl;
    std::cout << "Volatility (sigma):                   " << m_sigma << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

// VarianceGamma Class

VarGammaModel::VarGammaModel(double spotZero, double rfrRate, double ttMaturity, double divRate, double sigma,
                             double theta, double nu)
        : GenericModel(spotZero, rfrRate, ttMaturity, divRate), m_sigma(sigma), m_theta(theta), m_nu(nu) {}

VarGammaModel::~VarGammaModel() {};

VarGammaModel::VarGammaModel(const VarGammaModel &p) : GenericModel(p), m_sigma(p.m_sigma), m_theta(p.m_theta),
                                                       m_nu(p.m_nu) {};

VarGammaModel &VarGammaModel::operator=(const VarGammaModel &p) {
    if (this != &p) {
        GenericModel::operator=(p);
        m_sigma = p.m_sigma;
        m_theta = p.m_theta;
        m_nu = p.m_nu;
    }
    return *this;
}

std::vector<double> VarGammaModel::simulate(int nbSteps) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, 1.0);

    double kappa = 1 / m_nu;
    double divider = getTtMaturity() / nbSteps; // Time step size
    std::gamma_distribution<> dG(divider / kappa, m_nu);
    std::vector<double> spotPath(nbSteps, 0);
    spotPath[0] = getSpotZero();
    double driftAdjustment = m_nu * std::log(1 - m_theta * m_nu - 0.5 * m_sigma * m_sigma * m_nu);

    for (int i = 1; i < nbSteps; ++i) {
        double gammaIncrement = dG(gen);
        double brownianScaled = m_sigma * std::sqrt(gammaIncrement) * d(gen);
        double logReturn = driftAdjustment * gammaIncrement + m_theta * gammaIncrement + brownianScaled;
        spotPath[i] = spotPath[i - 1] * std::exp(logReturn);
    }
    return spotPath;
}

void VarGammaModel::showModel() {
    std::cout << "Variance Gamma Model Parameters:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Spot Price (Spot Zero):    " << getSpotZero() << std::endl;
    std::cout << "Risk-Free Rate (rfrRate):  " << getRfrRate() << std::endl;
    std::cout << "Time to Maturity:          " << getTtMaturity() << " years" << std::endl;
    std::cout << "Dividend Rate (divRate):   " << getDivRate() << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Variance Gamma Specific Parameters:   " << std::endl;
    std::cout << "Volatility (sigma):                   " << m_sigma << std::endl;
    std::cout << "Brownian mean (theta):                " << m_theta << std::endl;
    std::cout << "Gamma Variance (nu):                  " << m_nu << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

// Merton Jump Diffusion Class

MertonModel::MertonModel(double spotZero, double rfrRate, double ttMaturity, double divRate, double sigma,
                         double lambda, double muJ, double sigJ) : GenericModel(spotZero, rfrRate, ttMaturity, divRate),
                                                                   m_sigma(sigma), m_lambda(lambda), m_muJ(muJ),
                                                                   m_sigJ(sigJ) {};

MertonModel::~MertonModel() {};

MertonModel::MertonModel(const MertonModel &p) : GenericModel(p), m_sigma(p.m_sigma), m_sigJ(p.m_sigJ),
                                                 m_muJ(p.m_muJ), m_lambda(p.m_lambda) {};

MertonModel &MertonModel::operator=(const MertonModel &p) {
    if (this != &p) {
        GenericModel::operator=(p);
        m_sigJ = p.m_sigJ;
        m_muJ = p.m_muJ;
        m_lambda = p.m_lambda;
        m_sigma = p.m_sigma;

    };
    return *this;
}

std::vector<double> MertonModel::simulate(int nbSteps) {
    double divider = getTtMaturity() / nbSteps; // Time step size

    // Random number generators
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, 1.0);
    std::poisson_distribution<> dP(m_lambda * divider);
    std::normal_distribution<> dJump(m_muJ, m_sigJ);
    std::vector<double> spotPath(nbSteps, 0);
    spotPath[0] = getSpotZero();
    double driftAdjustment = (getRfrRate() - getDivRate()) - pow(m_sigma, 2) / 2
                             - m_lambda * (std::exp(m_muJ + 0.5 * pow(m_sigJ, 2)) - 1);
    for (int i = 1; i < nbSteps; ++i) {
        int jumpCount = dP(gen);
        double jumpSum = 0.0;
        for (int j = 0; j < jumpCount; ++j) {
            jumpSum += dJump(gen);
        }
        double brownianScaled = m_sigma * sqrt(divider) * d(gen);
        spotPath[i] = spotPath[i - 1] * std::exp(driftAdjustment * divider + brownianScaled + jumpSum);
    }
    return spotPath;
}


void MertonModel::showModel() {
    std::cout << "Merton Model Parameters:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Spot Price (Spot Zero):    " << getSpotZero() << std::endl;
    std::cout << "Risk-Free Rate (rfrRate):  " << getRfrRate() << std::endl;
    std::cout << "Time to Maturity:          " << getTtMaturity() << " years" << std::endl;
    std::cout << "Dividend Rate (divRate):   " << getDivRate() << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Merton Specific Parameters:   " << std::endl;
    std::cout << "Volatility (sigma):           " << m_sigma << std::endl;
    std::cout << "Jump Probability (lambda):    " << m_lambda << std::endl;
    std::cout << "Jump Volatility (sigJ):       " << m_sigJ << std::endl;
    std::cout << "Jump Average (muJ):           " << m_muJ << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

// Bates Model Class

BatesModel::BatesModel(double spotZero, double rfrRate, double ttMaturity, double divRate, double kappa, double theta,
                       double volVol, double rho, double volZero, double sigJ, double lambda) :
        GenericModel(spotZero, rfrRate, ttMaturity, divRate), m_kappa(kappa), m_theta(theta), m_rho(rho),
        m_volVol(volVol), m_volZero(volZero), m_sigJ(sigJ), m_lambda(lambda) {};

BatesModel::~BatesModel() {};

BatesModel::BatesModel(const BatesModel &p) : GenericModel(p), m_kappa(p.m_kappa), m_theta(p.m_theta), m_rho(p.m_rho),
                                              m_volVol(p.m_volVol), m_volZero(p.m_volZero), m_sigJ(p.m_sigJ),
                                              m_lambda(p.m_lambda) {}

BatesModel &BatesModel::operator=(const BatesModel &p) {
    if (this != &p) {
        GenericModel::operator=(p);
        m_kappa = p.m_kappa;
        m_theta = p.m_theta;
        m_volVol = p.m_volVol;
        m_rho = p.m_rho;
        m_volZero = p.m_volZero;
        m_sigJ = p.m_sigJ;
        m_lambda = p.m_lambda;
    }
    return *this;
}

std::vector<double> BatesModel::simulate(int nbSteps) {
    double divider = getTtMaturity() / nbSteps; // Time increment
    double squaredVol = pow(m_volVol, 2);

    // Random number generators
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, 1.0); // Standard normal distribution
    std::poisson_distribution<> poisson_dist(m_lambda * divider); // Poisson for jumps

    // Output paths
    std::vector<double> spotPath(nbSteps, 0);
    std::vector<double> volPath(nbSteps, 0);

    // Initial values
    spotPath[0] = getSpotZero();
    volPath[0] = m_volZero;

    // Drift component
    double drift = getRfrRate() - getDivRate();

    for (int i = 1; i < nbSteps; i++) {
        double dW = d(gen); // Brownian motion for volatility
        double dZ = d(gen); // Independent Brownian motion
        double corr = m_rho * dW + sqrt(1 - m_rho * m_rho) * dZ; // Correlated motion

        // Volatility update
        double vol_prev = volPath[i - 1];
        double mean_reversion = m_theta + (vol_prev - m_theta) * exp(-m_kappa * divider);
        double vol_variance = sqrt(squaredVol * vol_prev * (1 - exp(-2 * m_kappa * divider)) / (2 * m_kappa));
        volPath[i] = std::max(0.0, mean_reversion + vol_variance * dW);

        // Jump component
        int jump_count = poisson_dist(gen); // Number of jumps
        double jump_sum = 0;
        for (int j = 0; j < jump_count; j++) {
            jump_sum += m_sigJ * d(gen); // Sum of jumps
        }

        // Spot price update
        double vol = volPath[i - 1]; // Use lagged volatility for Ito's Lemma
        spotPath[i] = spotPath[i - 1] *
                      exp((drift - 0.5 * vol) * divider +
                          sqrt(vol) * sqrt(divider) * corr +
                          jump_sum);
    }

    return spotPath;
}

void BatesModel::showModel() {
    std::cout << "Bates Model Parameters:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Spot Price (Spot Zero):    " << getSpotZero() << std::endl;
    std::cout << "Risk-Free Rate (rfrRate):  " << getRfrRate() << std::endl;
    std::cout << "Time to Maturity:          " << getTtMaturity() << " years" << std::endl;
    std::cout << "Dividend Rate (divRate):   " << getDivRate() << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Bates Specific Parameters:" << std::endl;
    std::cout << "Mean Reversion Speed (kappa): " << m_kappa << std::endl;
    std::cout << "Long-Term Variance (theta):   " << m_theta << std::endl;
    std::cout << "Volatility of Volatility:     " << m_volVol << std::endl;
    std::cout << "Correlation (rho):            " << m_rho << std::endl;
    std::cout << "Initial Variance (volZero):   " << m_volZero << std::endl;
    std::cout << "Jump Probability (Lambda):    " << m_lambda << std::endl;
    std::cout << "Jump Volatility (sigJ):       " << m_sigJ << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

// NIG Model Class

NigModel::NigModel(double spotZero, double rfrRate, double ttMaturity, double divRate, double delta, double beta,
                   double alpha) : GenericModel(spotZero, rfrRate, ttMaturity, divRate), m_delta(delta), m_beta(beta),
                                   m_alpha(alpha) {};

NigModel::NigModel(const NigModel &p) : GenericModel(p), m_delta(p.m_delta), m_beta(p.m_beta), m_alpha(p.m_alpha) {};

NigModel::~NigModel() {};

NigModel &NigModel::operator=(const NigModel &p) {
    if (this != &p) {
        GenericModel::operator=(p);
        m_delta = p.m_delta;
        m_alpha = p.m_alpha;
        m_beta = p.m_beta;
    }
    return *this;
}

std::vector<double> NigModel::simulate(int nbSteps) {
    double divider = getTtMaturity() / nbSteps;
    double drift = getRfrRate() - getDivRate(); // Risk-neutral drift
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal(0.0, 1.0);
    std::uniform_real_distribution<> uniform(0.0, 1.0);

    // Path storage
    std::vector<double> spotPath(nbSteps, 0.0);
    spotPath[0] = getSpotZero();

    // Parameters for Inverse Gaussian simulation
    double gamma = sqrt(m_alpha * m_alpha - m_beta * m_beta);
    double mean = m_delta * divider; // Mean adjusted for time step
    double lambda = m_delta * m_delta * divider;

    for (int i = 1; i < nbSteps; ++i) {
        // Simulate Inverse Gaussian variable
        double z = normal(gen);
        double y = z * z;

        double x1 = mean + (mean * mean * y) / (2 * lambda) -
                    (mean / (2 * lambda)) * sqrt(4 * mean * lambda * y + mean * mean * y * y);
        double x2 = (mean * mean) / x1;

        double v = (uniform(gen) < mean / (mean + x1)) ? x1 : x2;

        // Simulate NIG increment
        double w = normal(gen); // Brownian motion
        double x = m_beta * v + sqrt(v) * w;

        // Update spot price
        double increment = (drift - 0.5 * v * m_beta * m_beta) * divider + x; // Adjusted drift term
        spotPath[i] = spotPath[i - 1] * exp(increment);
    }

    return spotPath;
}

void NigModel::showModel() {
    std::cout << "NIG Model Parameters:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Spot Price (Spot Zero):    " << getSpotZero() << std::endl;
    std::cout << "Risk-Free Rate (rfrRate):  " << getRfrRate() << std::endl;
    std::cout << "Time to Maturity:          " << getTtMaturity() << " years" << std::endl;
    std::cout << "Dividend Rate (divRate):   " << getDivRate() << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "NIG Specific Parameters:      " << std::endl;
    std::cout << "Thickness of tails (delta):   " << m_delta << std::endl;
    std::cout << "Skewness (beta):              " << m_beta << std::endl;
    std::cout << "Steepness (alpha):            " << m_alpha << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

// Kou Model Class

KouModel::KouModel(double spotZero, double rfrRate, double ttMaturity, double divRate, double sigma, double lambda,
                   double rho, double etaOne, double etaTwo) :
        GenericModel(spotZero, rfrRate, ttMaturity, divRate), m_sigma(sigma), m_lambda(lambda), m_rho(rho),
        m_etaOne(etaOne),
        m_etaTwo(etaTwo) {};

KouModel::KouModel(const KouModel &p) : GenericModel(p), m_sigma(p.m_sigma), m_lambda(p.m_lambda), m_rho(p.m_rho),
                                        m_etaOne(p.m_etaOne),
                                        m_etaTwo(p.m_etaTwo) {};

KouModel::~KouModel() {};


KouModel &KouModel::operator=(const KouModel &p) {
    if (this != &p) {
        GenericModel::operator=(p);
        m_sigma = p.m_sigma;
        m_lambda = p.m_lambda;
        m_rho = p.m_rho;
        m_etaOne = p.m_etaOne;
        m_etaTwo = p.m_etaTwo;
    }
    return *this;
}

std::vector<double> KouModel::simulate(int nbSteps) {
    double divider = getTtMaturity() / nbSteps; // Time step size
    double drift = ((getRfrRate() - getDivRate()) - 0.5 * pow(m_sigma, 2)) * divider;
    double invEtaOne = (-1 / m_etaOne);
    double invEtaTwo = (1 / m_etaTwo);
    // Random number generators
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, 1.0);
    std::poisson_distribution<> dP(m_lambda * divider);
    std::uniform_real_distribution<> dPU(0.0, 1);

    std::vector<double> spotPath(nbSteps, 0);

    spotPath[0] = getSpotZero();

    for (int i = 1; i < nbSteps; ++i) {
        double uniform = dPU(gen);
        double jump;
        if (uniform >= m_rho) {
            jump = invEtaOne * log((1 - uniform) / m_rho);
        } else {
            jump = invEtaTwo * log(uniform / (1 - m_rho));
        };
        jump *= dP(gen);

        spotPath[i] = spotPath[i - 1] * std::exp(drift + sqrt(divider) * m_sigma * d(gen) + jump);
    }
    return spotPath;
}

void KouModel::showModel() {
    std::cout << "Kou Model Parameters:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Spot Price (Spot Zero):    " << getSpotZero() << std::endl;
    std::cout << "Risk-Free Rate (rfrRate):  " << getRfrRate() << std::endl;
    std::cout << "Time to Maturity:          " << getTtMaturity() << " years" << std::endl;
    std::cout << "Dividend Rate (divRate):   " << getDivRate() << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Kou Specific Parameters:" << std::endl;
    std::cout << "Brownian Volatility (sigma):           " << m_sigma << std::endl;
    std::cout << "Jump Probability (lambda):             " << m_lambda << std::endl;
    std::cout << "Probability of an upward jump (Rho):   " << m_rho << std::endl;
    std::cout << "Upward jump magnitude:                 " << m_etaOne << std::endl;
    std::cout << "Downward jump magnitude:               " << m_etaTwo << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}


// CGMY Model Class

CGMYModel::CGMYModel(double spotZero, double rfrRate, double ttMaturity, double divRate, double c, double g, double m,
                     double y) :
        GenericModel(spotZero, rfrRate, ttMaturity, divRate), m_c(c), m_g(g), m_m(m),
        m_y(y) {};

CGMYModel::CGMYModel(const CGMYModel &p) : GenericModel(p), m_c(p.m_c), m_g(p.m_g), m_m(p.m_m),
                                           m_y(p.m_y) {};

CGMYModel::~CGMYModel() {};

CGMYModel &CGMYModel::operator=(const CGMYModel &p) {
    if (this != &p) {
        GenericModel::operator=(p);
        m_c = p.m_c;
        m_g = p.m_g;
        m_m = p.m_m;
        m_y = p.m_y;
    }
    return *this;
}

std::vector<double> CGMYModel::simulate(int nbSteps) {
    double divider = getTtMaturity() / nbSteps;
    double drift = getRfrRate() - getDivRate(); // Risk-neutral drift
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    std::exponential_distribution<> expG(1.0 / m_g); // Exponential with mean 1/m_g
    std::exponential_distribution<> expM(1.0 / m_m); // Exponential with mean 1/m_m

    std::vector<double> spotPath(nbSteps, 0);
    spotPath[0] = getSpotZero();

    for (int i = 1; i < nbSteps; ++i) {
        double jumpSum = 0.0;

        // Poisson intensity for the number of jumps
        double lambda = m_c * pow(divider, 1.0 - m_y);
        std::poisson_distribution<int> poisson(lambda);
        int numJumps = poisson(gen);

        // Simulate jumps
        for (int j = 0; j < numJumps; ++j) {
            double u = uniform(gen);
            if (u < 0.5) {
                // Negative jump
                double jumpSize = -expM(gen);
                jumpSum += jumpSize;
            } else {
                // Positive jump
                double jumpSize = expG(gen);
                jumpSum += jumpSize;
            }
        }

        // Adjust jumpSum to ensure it's consistent with the CGMY density
        jumpSum *= pow(divider, m_y);

        // Update the path with drift and jump contributions
        double increment = (drift - 0.5 * pow(jumpSum, 2)) * divider + jumpSum;
        spotPath[i] = spotPath[i - 1] * exp(increment);
    }

    return spotPath;
}

void CGMYModel::showModel() {
    std::cout << "CGMY Model Parameters:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Spot Price (Spot Zero):    " << getSpotZero() << std::endl;
    std::cout << "Risk-Free Rate (rfrRate):  " << getRfrRate() << std::endl;
    std::cout << "Time to Maturity:          " << getTtMaturity() << " years" << std::endl;
    std::cout << "Dividend Rate (divRate):   " << getDivRate() << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "CGMY Specific Parameters:" << std::endl;
    std::cout << "Intensity of jumps (c)       " << m_c << std::endl;
    std::cout << "Left Skewness (g):           " << m_g << std::endl;
    std::cout << "Right Skewness (m):          " << m_m << std::endl;
    std::cout << "Downward jump magnitude (y): " << m_y << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}