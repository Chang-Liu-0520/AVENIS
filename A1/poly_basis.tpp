#include "poly_basis.hpp"

template <typename Derived_Basis, int dim>
Poly_Basis<Derived_Basis, dim>::Poly_Basis(
 const std::vector<dealii::Point<1, double>> &support_points, const int &domain_)
  : poly_basis(new Derived_Basis(support_points, domain_))
{
}

template <typename Derived_Basis, int dim>
std::vector<double>
 Poly_Basis<Derived_Basis, dim>::value(const dealii::Point<dim, double> &P0)
{
  return poly_basis->value(P0);
}

template <typename Derived_Basis, int dim>
std::vector<double>
 Poly_Basis<Derived_Basis, dim>::value(const dealii::Point<dim, double> &P0,
                                       const unsigned half_range)
{
  return poly_basis->value(P0, half_range);
}

template <typename Derived_Basis, int dim>
std::vector<dealii::Tensor<1, dim>>
 Poly_Basis<Derived_Basis, dim>::grad(const dealii::Point<dim, double> &P0)
{
  return poly_basis->grad(P0);
}

template <typename Derived_Basis, int dim>
Poly_Basis<Derived_Basis, dim>::~Poly_Basis()
{
}
