#include <vector>
#include "jacobi_polynomial.hpp"

JacobiP::JacobiP(const int &n_in, const double &alpha_in, const double &beta_in, const int domain_in)
: integral_sc_fac(sqrt(2.0)), n(n_in), alpha(alpha_in), beta(beta_in), domain(domain_in)
{
}

inline double JacobiP::change_coords(double x_inp) const
{
  return (2L * x_inp - 1L);
}

std::vector<double> JacobiP::value(double x) const
{
  std::vector<double> result = compute(x);
  if (domain & Domain::From_0_to_1)
  {
    for (double &y : result)
      y *= integral_sc_fac;
  }
  return result;
}

std::vector<double> JacobiP::derivative(double x)
{
  std::vector<double> dP(n + 1);

  if (n == 0)
  {
    dP[0] = 0.0;
  }

  else
  {
    JacobiP JP0(n - 1, alpha + 1, beta + 1, domain);
    std::vector<double> P = JP0.compute(x);
    for (unsigned n1 = 0; n1 < n + 1; ++n1)
    {
      if (n1 == 0)
      {
        dP[0] = 0.0;
      }
      else
      {
        dP[n1] = sqrt(n1 * (n1 + alpha + beta + 1)) * P[n1 - 1];
        if (domain & Domain::From_0_to_1)
          dP[n1] *= 2 * integral_sc_fac;
      }
    }
  }
  return dP;
}

std::vector<double> JacobiP::compute(const double x_inp) const
{
  /* The Jacobi polynomial is evaluated using a recursion formula.
   * x     : The input point which should be in -1 <= x <= 1
   * alpha : ...
   * beta  : ...
   * n     : ...
   */
  double x = x_inp;
  if (domain & From_0_to_1)
    x = change_coords(x_inp);
  std::vector<double> p(n + 1);

  double aold = 0.0L, anew = 0.0L, bnew = 0.0L, h1 = 0.0L, prow, x_bnew;
  double gamma0 = 0.0L, gamma1 = 0.0L;
  double ab = alpha + beta, ab1 = alpha + beta + 1.0L, a1 = alpha + 1.0L,
         b1 = beta + 1.0L;

  gamma0 = pow(2.0L, ab1) / (ab1) * tgamma(a1) * tgamma(b1) / tgamma(ab1);

  // initial values P_0(x), P_1(x):
  p[0] = 1.0L / sqrt(gamma0);
  if (n == 0)
    return p;

  gamma1 = (a1) * (b1) / (ab + 3.0L) * gamma0;
  prow = ((ab + 2.0L) * x / 2.0L + (alpha - beta) / 2.0L) / sqrt(gamma1);
  p[1] = prow;
  if (n == 1)
    return p;

  aold = 2.0L / (2.0L + ab) * sqrt((a1) * (b1) / (ab + 3.0L));
  for (unsigned int i = 1; i <= (n - 1); ++i)
  {
    h1 = 2.0L * i + alpha + beta;
    anew = 2.0L / (h1 + 2.0L) * sqrt((i + 1) * (i + ab1) * (i + a1) * (i + b1) /
                                     (h1 + 1.0L) / (h1 + 3.0L));
    bnew = -(pow(alpha, 2) - pow(beta, 2)) / h1 / (h1 + 2.0L);
    x_bnew = x - bnew;
    p[i + 1] = 1.0L / anew * (-aold * p[i - 1] + x_bnew * p[i]);
    aold = anew;
  }
  return p;
}
