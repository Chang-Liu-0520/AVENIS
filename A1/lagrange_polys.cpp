#include "lagrange_polys.hpp"
#include <vector>

template <int dim, int spacedim>
Lagrange_Polies<dim, spacedim>::Lagrange_Polies(const std::vector<double> &support_points_,
                                                int domain_)
  : support_points(support_points_), n_polys(support_points_.size())
{
}

template <int dim, int spacedim>
std::vector<double> Lagrange_Polies<dim, spacedim>::value(const double& x) const
{
  std::vector<double> result = compute(x);
  return result;
}

template <int dim, int spacedim>
std::vector<double> Lagrange_Polies<dim, spacedim>::derivative(double x)
{
  std::vector<double> dL(n_polys);
  return dL;
}

template <int dim, int spacedim>
std::vector<double> Lagrange_Polies<dim, spacedim>::compute(const double& x_) const
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
