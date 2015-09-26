#include "lagrange_polys.hpp"
#include <vector>

template <int dim, int spacedim>
Lagrange_Polys<dim, spacedim>::Lagrange_Polys(const std::vector<double> &support_points_,
                                              int domain_)
  : support_points(support_points_), n_polys(support_points_.size())
{
}

template <int dim, int spacedim>
std::vector<double> Lagrange_Polys<dim, spacedim>::value(const double &x) const
{
  std::vector<double> result = compute(x);
  return result;
}

template <int dim, int spacedim>
std::vector<double> Lagrange_Polys<dim, spacedim>::derivative(double x)
{
  std::vector<double> dL(n_polys);
  return dL;
}

template <int dim, int spacedim>
std::vector<double> Lagrange_Polys<dim, spacedim>::compute(const double &x_) const
{
  double x = x_;
  std::vector<double> L(n_polys);
  for (unsigned i_poly = 0; i_poly < n_polys; ++i_poly)
  {
    double Numinator = 1.0;
    double Denuminator = 1.0;
    for (unsigned j_poly = 0; j_poly < n_polys; ++j_poly)
    {
      if (j_poly != i_poly)
      {
        Numinator *= (x - support_points[j_poly]);
        Denuminator *= (support_points[i_poly] - support_points[j_poly]);
      }
    }
    L[i_poly] = Numinator / Denuminator;
  }
  return L;
}

template <int dim, int spacedim>
std::vector<double>
Lagrange_Polys<dim, spacedim>::value(const dealii::Point<dim, double> &P0) const
{
  unsigned n_polys = n_polys - 1;
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

template <int dim, int spacedim>
std::vector<dealii::Tensor<1, dim>>
Lagrange_Polys<dim, spacedim>::grad(const dealii::Point<dim, double> &P0) const
{
  std::vector<dealii::Tensor<1, dim>> grad;
  grad.reserve(pow(n_polys, dim));

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
    for (unsigned i1 = 0; i1 < n_polys; ++i1)
    {
      grad_N[0] = one_D_grads[0][i1];
      grad.push_back(std::move(grad_N));
    }
    break;
  case 2:
    for (unsigned i2 = 0; i2 < n_polys + 1; ++i2)
      for (unsigned i1 = 0; i1 < n_polys + 1; ++i1)
      {
        grad_N[0] = one_D_grads[0][i1] * one_D_values[1][i2];
        grad_N[1] = one_D_values[0][i1] * one_D_grads[1][i2];
        grad.push_back(std::move(grad_N));
      }
    break;
  case 3:
    for (unsigned i3 = 0; i3 < n_polys + 1; ++i3)
      for (unsigned i2 = 0; i2 < n_polys + 1; ++i2)
        for (unsigned i1 = 0; i1 < n_polys + 1; ++i1)
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
