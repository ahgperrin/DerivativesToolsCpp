//
// Created by Antoine Perrin on 12/8/24.
//

#ifndef UNTITLED_MODELS_HPP
#define UNTITLED_MODELS_HPP

#include <vector>


class GenericModel {

protected:
    GenericModel(double spotZero, double rfrRate, double ttMaturity, double divRate);

    GenericModel(const GenericModel &p);

    ~GenericModel();

    GenericModel &operator=(const GenericModel &p);

private:

    double m_spotZero;
    double m_rfrRate;
    double m_ttMaturity;
    double m_divRate;

protected:
    double getSpotZero() const;

    double getRfrRate() const;

    double getTtMaturity() const;

    double getDivRate() const;

};

// HestonModel Class

class HestonModel : public GenericModel {
    // Heston Model Class
    // Principle:
    // The Heston Model is a stochastic volatility model where the volatility follows a mean-reverting
    // square-root process.
    // It explains market phenomena like volatility clustering and implied volatility skew/smile.
    // Parameters:
    // - spotZero: Initial spot price of the asset.
    // - rfrRate: Risk-free interest rate.
    // - ttMaturity: Time to maturity of the derivative.
    // - divRate: Dividend rate of the underlying asset.
    // - kappa: Speed of mean reversion of the volatility.
    // - theta: Long-term mean level of volatility.
    // - volVol: Volatility of volatility (variance's volatility).
    // - rho: Correlation between asset returns and volatility.
    // - volZero: Initial volatility level.

public:
    HestonModel(double spotZero, double rfrRate, double ttMaturity, double divRate, double kappa, double theta,
                double volVol, double rho, double volZero);

    HestonModel(const HestonModel &p);

    ~HestonModel();

    HestonModel &operator=(const HestonModel &p);

    void showModel();

    std::vector<double> simulate(int nbSteps);

private:
    double m_kappa;
    double m_theta;
    double m_volVol;
    double m_rho;
    double m_volZero;

};

// BlackScholes Class

class BlackScholesModel : public GenericModel {
    // Black-Scholes Model Class
    // Principle:
    // The Black-Scholes Model is a classic option pricing model assuming constant volatility
    // and a log-normal asset price distribution.
    // Parameters:
    // - spotZero: Initial spot price of the asset.
    // - rfrRate: Risk-free interest rate.
    // - ttMaturity: Time to maturity of the derivative.
    // - divRate: Dividend rate of the underlying asset.
    // - sigma: Constant volatility of the asset's returns.

public:
    BlackScholesModel(double spotZero, double rfrRate, double ttMaturity, double divRate, double sigma);

    BlackScholesModel(const BlackScholesModel &p);

    ~BlackScholesModel();

    BlackScholesModel &operator=(const BlackScholesModel &p);

    void showModel();

    std::vector<double> simulate(int nbSteps);

private:
    double m_sigma;
};

// VarianceGamma Class

class VarGammaModel : public GenericModel {
    // Variance Gamma Model Class
    // Principle:
    // The Variance Gamma (VG) Model models asset returns with a Gamma-driven process,
    // incorporating skewness and kurtosis.
    // Parameters:
    // - spotZero: Initial spot price of the asset.
    // - rfrRate: Risk-free interest rate.
    // - ttMaturity: Time to maturity of the derivative.
    // - divRate: Dividend rate of the underlying asset.
    // - sigma: Volatility of the Brownian motion component.
    // - theta: Drift rate of the Brownian motion component.
    // - nu: Variance of the Gamma process, controlling jump intensity.

public:
    VarGammaModel(double spotZero, double rfrRate, double ttMaturity, double divRate, double sigma, double theta,
                  double nu);

    VarGammaModel(const VarGammaModel &p);

    ~VarGammaModel();

    VarGammaModel &operator=(const VarGammaModel &p);

    std::vector<double> simulate(int nbSteps);

    void showModel();

private:
    double m_sigma;
    double m_theta;
    double m_nu;

};

// Merton Jump Diffusion Class

class MertonModel : public GenericModel {
    // Merton Jump Diffusion Model Class
    // Principle:
    // The Merton Jump Diffusion Model extends the Black-Scholes model by adding jump dynamics
    // to account for large infrequent price changes.
    // Parameters:
    // - spotZero: Initial spot price of the asset.
    // - rfrRate: Risk-free interest rate.
    // - ttMaturity: Time to maturity of the derivative.
    // - divRate: Dividend rate of the underlying asset.
    // - sigma: Volatility of the Brownian motion component.
    // - lambda: Jump intensity (frequency of jumps).
    // - muJ: Mean of the log-normal distribution governing jump sizes.
    // - sigJ: Standard deviation of the jump size distribution.

public:
    MertonModel(double spotZero, double rfrRate, double ttMaturity, double divRate, double sigma, double lambda,
                double muJ, double sigJ);

    MertonModel(const MertonModel &p);

    ~MertonModel();

    MertonModel &operator=(const MertonModel &p);

    std::vector<double> simulate(int nbSteps);

    void showModel();

private:
    double m_sigma;
    double m_lambda;
    double m_muJ;
    double m_sigJ;

};


// Bates Model Class

class BatesModel : public GenericModel {
    // Bates Model Class
    // Principle:
    // The Bates Model combines the Heston stochastic volatility framework with jump dynamics
    // for the asset price process.
    // Parameters:
    // - spotZero: Initial spot price of the asset.
    // - rfrRate: Risk-free interest rate.
    // - ttMaturity: Time to maturity of the derivative.
    // - divRate: Dividend rate of the underlying asset.
    // - kappa: Speed of mean reversion of the volatility.
    // - theta: Long-term mean level of volatility.
    // - volVol: Volatility of volatility (variance's volatility).
    // - rho: Correlation between asset returns and volatility.
    // - volZero: Initial volatility level.
    // - sigJ: Standard deviation of the jump size distribution.
    // - lambda: Jump intensity (frequency of jumps).

public:
    BatesModel(double spotZero, double rfrRate, double ttMaturity, double divRate, double kappa, double theta,
               double volVol, double rho, double volZero, double sigJ, double lambda);

    BatesModel(const BatesModel &p);

    ~BatesModel();

    BatesModel &operator=(const BatesModel &p);

    std::vector<double> simulate(int nbSteps);

    void showModel();

private:
    double m_kappa;
    double m_theta;
    double m_volVol;
    double m_rho;
    double m_volZero;
    double m_sigJ;
    double m_lambda;
};

// NIG Model Class

class NigModel : public GenericModel {
    // NIG (Normal Inverse Gaussian) Model Class
    // Principle:
    // The NIG Model is a Lévy process used to capture heavy tails and skewness in asset return
    // distributions for more accurate option pricing.
    // Parameters:
    // - spotZero: Initial spot price of the asset.
    // - rfrRate: Risk-free interest rate.
    // - ttMaturity: Time to maturity of the derivative.
    // - divRate: Dividend rate of the underlying asset.
    // - delta: Scale parameter controlling the thickness of the tails.
    // - beta: Asymmetry parameter controlling skewness.
    // - alpha: Tail decay parameter controlling steepness.


public:
    NigModel(double spotZero, double rfrRate, double ttMaturity, double divRate, double delta, double beta,
             double alpha);

    NigModel(const NigModel &p);

    ~NigModel();

    NigModel &operator=(const NigModel &p);

    std::vector<double> simulate(int nbSteps);

    void showModel();

private:
    double m_delta;
    double m_beta;
    double m_alpha;

};

// Kou Model Class

class KouModel : public GenericModel {
    // Kou Model Class
    // Principle:
    // The Kou Model is a jump diffusion model that uses a double exponential distribution for
    // jump sizes, capturing realistic skewness and kurtosis.
    // Parameters:
    // - spotZero: Initial spot price of the asset.
    // - rfrRate: Risk-free interest rate.
    // - ttMaturity: Time to maturity of the derivative.
    // - divRate: Dividend rate of the underlying asset.
    // - sigma: Volatility of the Brownian motion component.
    // - lambda: Jump intensity (frequency of jumps).
    // - rho: Probability of an upward jump.
    // - etaOne: Rate parameter for upward jumps' magnitude.
    // - etaTwo: Rate parameter for downward jumps' magnitude.

public:
    KouModel(double spotZero, double rfrRate, double ttMaturity, double divRate, double sigma, double lambda,
             double rho, double etaOne, double etaTwo);

    KouModel(const KouModel &p);

    ~KouModel();

    KouModel &operator=(const KouModel &p);

    std::vector<double> simulate(int nbSteps);

    void showModel();

private:
    double m_sigma;
    double m_lambda;
    double m_rho;
    double m_etaOne;
    double m_etaTwo;
};

// CGMY Model Class

class CGMYModel : public GenericModel {
    // CGMY Model Class
    // Principle:
    // The CGMY Model is a general Lévy process model that captures infinite activity and finite variation in asset
    // price changes, accounting for skewness, kurtosis, and fat tails.
    // Parameters:
    // - spotZero: Initial spot price of the asset.
    // - rfrRate: Risk-free interest rate.
    // - ttMaturity: Time to maturity of the derivative.
    // - divRate: Dividend rate of the underlying asset.
    // - c: Scale parameter controlling the overall intensity of jumps.
    // - g: Parameter for jump distribution skewness (left tails).
    // - m: Parameter for jump distribution skewness (right tails).
    // - y: Parameter controlling tail behavior and activity of small jumps.

public:
    CGMYModel(double spotZero, double rfrRate, double ttMaturity, double divRate, double c, double g,
              double m, double y);

    CGMYModel(const CGMYModel &p);

    ~CGMYModel();

    CGMYModel &operator=(const CGMYModel &p);

    std::vector<double> simulate(int nbSteps);

    void showModel();

private:
    double m_c;
    double m_g;
    double m_m;
    double m_y;
};

#endif //UNTITLED_MODELS_HPP
