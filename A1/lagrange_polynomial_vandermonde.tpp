#include <float.h>
#include <vector>
#include "lagrange_polynomial_vandermonde.hpp"

template <int dim>
Lagrange_Polys_Vandermonde<dim>::Lagrange_Polys_Vandermonde(
 const std::vector<dealii::Point<1, double>> &support_points_, int domain_)
  : support_points(support_points_),
    polyspace_order(support_points_.size() - 1),
    domain(domain_),
    jacobi_poly(support_points_, domain_),
    Vandermonde_T_inv(polyspace_order + 1, polyspace_order + 1)
{
  Eigen::MatrixXd Vandermonde_T(polyspace_order + 1, polyspace_order + 1);
  for (int i_point = 0; i_point < polyspace_order + 1; ++i_point)
  {
    std::vector<double> Legandre_at_Point_i =
     jacobi_poly.value(support_points[i_point]);
    for (int j_poly = 0; j_poly < polyspace_order + 1; ++j_poly)
    {
      Vandermonde_T(j_poly, i_point) = Legandre_at_Point_i[j_poly];
    }
  }
  Vandermonde_T_inv = Vandermonde_T.inverse();
}

template <int dim>
std::vector<double> Lagrange_Polys_Vandermonde<dim>::value(const double &x)
{
  std::vector<double> result = compute(x);
  return result;
}

template <int dim>
std::vector<double> Lagrange_Polys_Vandermonde<dim>::derivative(const double &x)
{
  std::vector<double> dL = jacobi_poly.derivative(x);
  Eigen::MatrixXd dP_js(polyspace_order + 1, 1);
  for (unsigned i_poly = 0; i_poly < polyspace_order + 1; ++i_poly)
    dP_js(i_poly, 0) = dL[i_poly];
  Eigen::MatrixXd dL_Mat = Vandermonde_T_inv * dP_js;
  for (unsigned i_poly = 0; i_poly < polyspace_order + 1; ++i_poly)
    dL[i_poly] = dL_Mat(i_poly, 0);
  return dL;
}

template <int dim>
std::vector<double> Lagrange_Polys_Vandermonde<dim>::compute(const double &x)
{
  std::vector<double> L = jacobi_poly.value(x);
  Eigen::MatrixXd P_js(polyspace_order + 1, 1);
  for (unsigned i_poly = 0; i_poly < polyspace_order + 1; ++i_poly)
    P_js(i_poly, 0) = L[i_poly];
  Eigen::MatrixXd L_Mat = Vandermonde_T_inv * P_js;
  for (unsigned i_poly = 0; i_poly < polyspace_order + 1; ++i_poly)
    L[i_poly] = L_Mat(i_poly, 0);
  return L;
}

template <int dim>
std::vector<double>
 Lagrange_Polys_Vandermonde<dim>::value(const dealii::Point<dim, double> &P0)
{
  unsigned n_polys = polyspace_order + 1;
  std::vector<double> result;
  result.reserve(pow(n_polys, dim));

  std::vector<std::vector<double>> one_D_values;
  for (unsigned i1 = 0; i1 < dim; i1++)
    one_D_values.push_back(std::move(value(P0(i1))));

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
std::vector<dealii::Tensor<1, dim>>
 Lagrange_Polys_Vandermonde<dim>::grad(const dealii::Point<dim, double> &P0)
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
Lagrange_Polys_Vandermonde<dim>::~Lagrange_Polys_Vandermonde()
{
}
