#include "jacobi_polynomial.hpp"

template <int dim>
Jacobi_Poly_Basis<dim>::Jacobi_Poly_Basis(const std::vector<dealii::Point<1>> &Supp_Points,
                                          const int &domain_)
  : integral_sc_fac(sqrt(2.0)), domain(domain_)
{
  alpha = 0;
  beta = 0;
  polyspace_order = Supp_Points.size() - 1;
}

template <int dim>
Jacobi_Poly_Basis<dim>::Jacobi_Poly_Basis(const unsigned &polyspace_order_,
                                          const double &alpha_,
                                          const double &beta_,
                                          const int &domain_)
  : integral_sc_fac(sqrt(2.0)),
    polyspace_order(polyspace_order_),
    alpha(alpha_),
    beta(beta_),
    domain(domain_)
{
}

template <int dim>
inline double Jacobi_Poly_Basis<dim>::change_coords(const double &x_inp)
{
  return (2L * x_inp - 1L);
}

template <int dim>
std::vector<double> Jacobi_Poly_Basis<dim>::value(const double &x)
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
std::vector<double> Jacobi_Poly_Basis<dim>::derivative(const double &x)
{
  std::vector<double> dP(polyspace_order + 1);

  if (polyspace_order == 0)
  {
    dP[0] = 0.0;
  }

  else
  {
    Jacobi_Poly_Basis JP0(polyspace_order - 1, alpha + 1, beta + 1, domain);
    std::vector<double> P = JP0.compute(x);
    for (unsigned n1 = 0; n1 < polyspace_order + 1; ++n1)
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
std::vector<double> Jacobi_Poly_Basis<dim>::value(const dealii::Point<dim, double> &P0)
{
  std::vector<double> result;
  result.reserve(pow(polyspace_order + 1, dim));

  std::vector<std::vector<double>> one_D_values;
  for (unsigned i1 = 0; i1 < dim; i1++)
    one_D_values.push_back(std::move(value(P0(i1))));

  switch (dim)
  {
  case 1:
    for (unsigned i1 = 0; i1 < polyspace_order + 1; ++i1)
      result.push_back(one_D_values[0][i1]);
    break;
  case 2:
    for (unsigned i2 = 0; i2 < polyspace_order + 1; ++i2)
      for (unsigned i1 = 0; i1 < polyspace_order + 1; ++i1)
        result.push_back(one_D_values[0][i1] * one_D_values[1][i2]);
    break;
  case 3:
    for (unsigned i3 = 0; i3 < polyspace_order + 1; ++i3)
      for (unsigned i2 = 0; i2 < polyspace_order + 1; ++i2)
        for (unsigned i1 = 0; i1 < polyspace_order + 1; ++i1)
          result.push_back(one_D_values[0][i1] * one_D_values[1][i2] *
                           one_D_values[2][i3]);
    break;
  }
  return result;
}

template <int dim>
std::vector<double> Jacobi_Poly_Basis<dim>::value(const dealii::Point<dim, double> &P0,
                                                  const unsigned &half_range)
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
        dealii::Point<dim, double> P0_mod(P0(0) / 2.0);
        result = value(P0_mod);
      }
      if (half_range == 2)
      {
        dealii::Point<dim, double> P0_mod(0.5 + P0(0) / 2.0);
        result = value(P0_mod);
      }
    }
    if (P0.dimension == 2)
    {
      if (half_range == 1)
      {
        dealii::Point<dim, double> P0_mod(P0(0) / 2.0, P0(1) / 2.0);
        result = value(P0_mod);
      }
      if (half_range == 2)
      {
        dealii::Point<dim, double> P0_mod(0.5 + P0(0) / 2.0, P0(1) / 2.0);
        result = value(P0_mod);
      }
      if (half_range == 3)
      {
        dealii::Point<dim, double> P0_mod(P0(0) / 2.0, 0.5 + P0(1) / 2.0);
        result = value(P0_mod);
      }
      if (half_range == 4)
      {
        dealii::Point<dim, double> P0_mod(0.5 + P0(0) / 2.0, 0.5 + P0(1) / 2.0);
        result = value(P0_mod);
      }
    }
  }
  return result;
}

template <int dim>
std::vector<dealii::Tensor<1, dim>>
 Jacobi_Poly_Basis<dim>::grad(const dealii::Point<dim, double> &P0)
{
  std::vector<dealii::Tensor<1, dim>> grad;
  grad.reserve(pow(polyspace_order + 1, dim));

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
    for (unsigned i1 = 0; i1 < polyspace_order + 1; ++i1)
    {
      grad_N[0] = one_D_grads[0][i1];
      grad.push_back(std::move(grad_N));
    }
    break;
  case 2:
    for (unsigned i2 = 0; i2 < polyspace_order + 1; ++i2)
      for (unsigned i1 = 0; i1 < polyspace_order + 1; ++i1)
      {
        grad_N[0] = one_D_grads[0][i1] * one_D_values[1][i2];
        grad_N[1] = one_D_values[0][i1] * one_D_grads[1][i2];
        grad.push_back(std::move(grad_N));
      }
    break;
  case 3:
    for (unsigned i3 = 0; i3 < polyspace_order + 1; ++i3)
      for (unsigned i2 = 0; i2 < polyspace_order + 1; ++i2)
        for (unsigned i1 = 0; i1 < polyspace_order + 1; ++i1)
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
std::vector<double> Jacobi_Poly_Basis<dim>::compute(const double &x_inp)
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
  std::vector<double> p(polyspace_order + 1);

  double aold = 0.0L, anew = 0.0L, bnew = 0.0L, h1 = 0.0L, prow, x_bnew;
  double gamma0 = 0.0L, gamma1 = 0.0L;
  double ab = alpha + beta, ab1 = alpha + beta + 1.0L, a1 = alpha + 1.0L,
         b1 = beta + 1.0L;

  gamma0 = pow(2.0L, ab1) / (ab1) * tgamma(a1) * tgamma(b1) / tgamma(ab1);

  // initial values P_0(x), P_1(x):
  p[0] = 1.0L / sqrt(gamma0);
  if (polyspace_order == 0)
    return p;

  gamma1 = (a1) * (b1) / (ab + 3.0L) * gamma0;
  prow = ((ab + 2.0L) * x / 2.0L + (alpha - beta) / 2.0L) / sqrt(gamma1);
  p[1] = prow;
  if (polyspace_order == 1)
    return p;

  aold = 2.0L / (2.0L + ab) * sqrt((a1) * (b1) / (ab + 3.0L));
  for (unsigned int i = 1; i <= (polyspace_order - 1); ++i)
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
Jacobi_Poly_Basis<dim>::~Jacobi_Poly_Basis()
{
}

/*
template <int dim>
template <int func_dim>
void Jacobi_Poly_Basis<dim>::project_to(const Function<func_dim, double> &func,
                                        const
std::vector<dealii::Point<func_dim>> &integration_points_,
                                        const std::vector<double> &weights,
                                        Eigen::MatrixXd &vec)
{
}

template <int dim>
template <int func_dim, typename T>
void Jacobi_Poly_Basis<dim>::project_to(
 const Function<func_dim, T> &func,
 const std::vector<dealii::Point<func_dim>> &integration_points,
 const std::vector<dealii::Point<func_dim>> &normals_at_integration,
 const std::vector<double> &weights,
 Eigen::MatrixXd &vec)
{
}
*/
