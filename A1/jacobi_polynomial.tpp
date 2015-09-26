#include <vector>
#include "jacobi_polynomial.hpp"

template <int dim>
JacobiP<dim>::JacobiP()
  : integral_sc_fac(sqrt(2.0))
{
}

template <int dim>
JacobiP<dim>::JacobiP(const int &n_in,
                      const double &alpha_in,
                      const double &beta_in,
                      const int domain_in)
  : integral_sc_fac(sqrt(2.0)), n(n_in), alpha(alpha_in), beta(beta_in), domain(domain_in)
{
}

/*
 * This initializer is only required for empty constructor. When empty
 * constructor is
 * used, one has to set the values of alpha, beta and domain_type.
 */
template <int dim>
void JacobiP<dim>::init(const std::vector<dealii::Point<dim>> &Supp_Points)
{
  alpha = 0;
  beta = 0;
  n = Supp_Points.size() - 1;
  domain = From_0_to_1;
}

template <int dim>
inline double JacobiP<dim>::change_coords(double x_inp) const
{
  return (2L * x_inp - 1L);
}

template <int dim>
std::vector<double> JacobiP<dim>::value(double x) const
{
  std::vector<double> result = compute(x);
  if (domain & Domain::From_0_to_1)
  {
    for (double &y : result)
      y *= integral_sc_fac;
  }
  return result;
}

template <int dim>
std::vector<double> JacobiP<dim>::derivative(double x) const
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

template <int dim>
std::vector<double> JacobiP<dim>::value(const dealii::Point<dim> &P0) const
{
  std::vector<double> result;
  result.reserve(pow(n + 1, dim));

  std::vector<std::vector<double>> one_D_values;
  for (unsigned i1 = 0; i1 < dim; i1++)
    one_D_values.push_back(std::move(value(P0(i1))));

  switch (dim)
  {
  case 1:
    for (unsigned i1 = 0; i1 < n + 1; ++i1)
      result.push_back(one_D_values[0][i1]);
    break;
  case 2:
    for (unsigned i2 = 0; i2 < n + 1; ++i2)
      for (unsigned i1 = 0; i1 < n + 1; ++i1)
        result.push_back(one_D_values[0][i1] * one_D_values[1][i2]);
    break;
  case 3:
    for (unsigned i3 = 0; i3 < n + 1; ++i3)
      for (unsigned i2 = 0; i2 < n + 1; ++i2)
        for (unsigned i1 = 0; i1 < n + 1; ++i1)
          result.push_back(one_D_values[0][i1] * one_D_values[1][i2] *
                           one_D_values[2][i3]);
    break;
  }
  return result;
}

template <int dim>
std::vector<double>
JacobiP<dim>::value(const dealii::Point<dim> &P0, const unsigned half_range) const
{
  assert(half_range <= pow(2, P0.dimension));
  std::vector<double> result;
  if (half_range == 0)
    result = value(P0);
  else
  {
    if (P0.dimension == 1)
    {
      if (half_range == 1)
      {
        dealii::Point<dim> P0_mod(P0(0) / 2.0);
        result = value(P0_mod);
      }
      if (half_range == 2)
      {
        dealii::Point<dim> P0_mod(0.5 + P0(0) / 2.0);
        result = value(P0_mod);
      }
    }
    if (P0.dimension == 2)
    {
      if (half_range == 1)
      {
        dealii::Point<dim> P0_mod(P0(0) / 2.0, P0(1) / 2.0);
        result = value(P0_mod);
      }
      if (half_range == 2)
      {
        dealii::Point<dim> P0_mod(0.5 + P0(0) / 2.0, P0(1) / 2.0);
        result = value(P0_mod);
      }
      if (half_range == 3)
      {
        dealii::Point<dim> P0_mod(P0(0) / 2.0, 0.5 + P0(1) / 2.0);
        result = value(P0_mod);
      }
      if (half_range == 4)
      {
        dealii::Point<dim> P0_mod(0.5 + P0(0) / 2.0, 0.5 + P0(1) / 2.0);
        result = value(P0_mod);
      }
    }
  }
  return result;
}

template <int dim>
std::vector<dealii::Tensor<1, dim>> JacobiP<dim>::grad(const dealii::Point<dim> &P0) const
{
  std::vector<dealii::Tensor<1, dim>> grad;
  grad.reserve(pow(n + 1, dim));

  std::vector<std::vector<double>> one_D_values;
  for (unsigned i1 = 0; i1 < dim; i1++)
    one_D_values.push_back(std::move(value(P0(i1))));

  std::vector<std::vector<double>> one_D_grads;
  for (unsigned i1 = 0; i1 < dim; i1++)
    one_D_grads.push_back(std::move(derivative(P0(i1))));

  dealii::Tensor<1, dim> grad_N;
  switch (dim)
  {
  case 1:
    for (unsigned i1 = 0; i1 < n + 1; ++i1)
    {
      grad_N[0] = one_D_grads[0][i1];
      grad.push_back(std::move(grad_N));
    }
    break;
  case 2:
    for (unsigned i2 = 0; i2 < n + 1; ++i2)
      for (unsigned i1 = 0; i1 < n + 1; ++i1)
      {
        grad_N[0] = one_D_grads[0][i1] * one_D_values[1][i2];
        grad_N[1] = one_D_values[0][i1] * one_D_grads[1][i2];
        grad.push_back(std::move(grad_N));
      }
    break;
  case 3:
    for (unsigned i3 = 0; i3 < n + 1; ++i3)
      for (unsigned i2 = 0; i2 < n + 1; ++i2)
        for (unsigned i1 = 0; i1 < n + 1; ++i1)
        {
          grad_N[0] = one_D_grads[0][i1] * one_D_values[1][i2] * one_D_values[2][i3];
          grad_N[1] = one_D_values[0][i1] * one_D_grads[1][i2] * one_D_values[2][i3];
          grad_N[2] = one_D_values[0][i1] * one_D_values[1][i2] * one_D_grads[2][i3];
          grad.push_back(std::move(grad_N));
        }

    break;
  }
  return grad;
}

template <int dim>
std::vector<double> JacobiP<dim>::compute(const double x_inp) const
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

template <int dim>
JacobiP<dim>::~JacobiP()
{
}

Derived_Factory<JacobiP<1>, Poly_Basis<1, 1>, std::string>
Poly_Factory_1D("legendre1");

Derived_Factory<JacobiP<2>, Poly_Basis<2, 2>, std::string>
Poly_Factory_2D("legendre2");

Derived_Factory<JacobiP<3>, Poly_Basis<3, 3>, std::string>
Poly_Factory_3D("legendre3");
