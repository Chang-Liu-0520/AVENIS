#include "lagrange_polies.hpp"
#include <vector>

template <int dim, int spacedim>
double Lagrange_Polies<dim, spacedim>::change_coords(double x_inp) const
{
  return (2L * x_inp - 1L);
}

template <int dim, int spacedim>
std::vector<double> Lagrange_Polies<dim, spacedim>::value(double x) const
{
  std::vector<double> result = compute(x);
  return result;
}

template <int dim, int spacedim>
std::vector<double> Lagrange_Polies<dim, spacedim>::derivative(double x)
{
  std::vector<double> dL(n_polys + 1);
  return dL;
}

template <int dim, int spacedim>
std::vector<double> Lagrange_Polies<dim, spacedim>::compute(const double x_) const
{
  double x = x_;
  if (domain & From_0_to_1)
    x = change_coords(x_);
  std::vector<double> L(n_polys + 1);
  return L;
}
