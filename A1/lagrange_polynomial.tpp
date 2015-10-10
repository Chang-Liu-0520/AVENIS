#include <float.h>
#include <vector>
#include "lagrange_polynomial.hpp"

template <int dim>
Lagrange_Polys<dim>::Lagrange_Polys(const std::vector<dealii::Point<1, double>> &support_points_,
                                    int domain_)
  : support_points(support_points_),
    polyspace_order(support_points_.size() - 1),
    domain(domain_)
{
}

template <int dim>
std::vector<double> Lagrange_Polys<dim>::value(const double &x) const
{
  std::vector<double> result = compute(x);
  return result;
}

template <int dim>
std::vector<double> Lagrange_Polys<dim>::derivative(double x) const
{
  std::vector<double> dL(polyspace_order + 1);
  /* We use the formula
   *
   * L'_i(x) = L_i(x) \sum_{m=0, m\ne i}^n \frac {1}{x-x_m}
   *
   */
  for (unsigned i_poly = 0; i_poly < polyspace_order + 1; ++i_poly)
  {
    double x_i = support_points[i_poly][0];
    double numinator = 0.0;
    double denuminator = 1.0;
    for (unsigned m_poly = 0; m_poly < polyspace_order + 1; ++m_poly)
    {
      double x_m = support_points[m_poly][0];
      double product_without_m = 1;
      if (i_poly != m_poly)
      {
        denuminator *= (x_i - x_m);
        for (unsigned j_poly = 0; j_poly < polyspace_order + 1; ++j_poly)
        {
          double x_j = support_points[j_poly][0];
          if (j_poly != m_poly && j_poly != i_poly)
            product_without_m *= (x - x_j);
        }
        numinator += product_without_m;
      }
    }
    dL[i_poly] = numinator / denuminator;
  }
  return dL;
}

template <int dim>
std::vector<double> Lagrange_Polys<dim>::compute(const double &x_) const
{
  double x = x_;
  std::vector<double> L(polyspace_order + 1);
  for (unsigned i_poly = 0; i_poly < polyspace_order + 1; ++i_poly)
  {
    L[i_poly] = compute_Li(x, i_poly);
  }
  return L;
}

template <int dim>
double Lagrange_Polys<dim>::compute_Li(const double &x_, const unsigned &i_poly) const
{
  double x = x_;
  double L;
  double Numinator = 1.0;
  double Denuminator = 1.0;
  for (unsigned j_poly = 0; j_poly < polyspace_order + 1; ++j_poly)
  {
    if (j_poly != i_poly)
    {
      Numinator *= (x - support_points[j_poly][0]);
      Denuminator *= (support_points[i_poly][0] - support_points[j_poly][0]);
    }
  }
  L = Numinator / Denuminator;
  return L;
}

/*!
 * This function calculates the basis function at a given point.
 * For example for 2nd order 2D element. The first function corresponds
 * to (0,0), the second function corresponds to (0.5,0), 3rd (1,0),
 * 4th (0,0.5), 5th (0.5,0.5), ... .
 */
template <int dim>
std::vector<double> Lagrange_Polys<dim>::value(const dealii::Point<dim, double> &P0) const
{
  unsigned n_polys = polyspace_order + 1;
  std::vector<double> result;
  result.reserve(pow(n_polys, dim));

  std::vector<std::vector<double>> one_D_values;
  for (unsigned i1 = 0; i1 < dim; i1++)
    one_D_values.push_back(std::move(compute(P0(i1))));

  switch (dim)
  {
  case 1:
    for (unsigned i1 = 0; i1 < n_polys; ++i1)
      result.push_back(one_D_values[0][i1]);
    break;
  case 2:
    for (unsigned i2 = 0; i2 < n_polys; ++i2)
      for (unsigned i1 = 0; i1 < n_polys; ++i1)
        result.push_back(one_D_values[0][i1] * one_D_values[1][i2]);
    break;
  case 3:
    for (unsigned i3 = 0; i3 < n_polys; ++i3)
      for (unsigned i2 = 0; i2 < n_polys; ++i2)
        for (unsigned i1 = 0; i1 < n_polys; ++i1)
          result.push_back(one_D_values[0][i1] * one_D_values[1][i2] *
                           one_D_values[2][i3]);
    break;
  }
  return result;
}

template <int dim>
std::vector<double> Lagrange_Polys<dim>::value(const dealii::Point<dim, double> &P0,
                                               const unsigned &half_range) const
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
 Lagrange_Polys<dim>::grad(const dealii::Point<dim, double> &P0) const
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
Lagrange_Polys<dim>::~Lagrange_Polys()
{
}
