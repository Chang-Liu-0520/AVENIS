#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <deal.II/base/tensor.h>

#ifndef POLY_BASIS
#define POLY_BASIS

enum Domain
{
  From_0_to_1 = 1 << 0,
  From_minus_1_to_1 = 1 << 1
};

#include "jacobi_polynomial.hpp"
#include "lagrange_polynomial.hpp"
#include "lagrange_polynomial_vandermonde.hpp"

template <typename Derived_Basis, int dim>
class Poly_Basis
{
 public:
  Poly_Basis() = delete;
  Poly_Basis(const std::vector<dealii::Point<1, double>> &support_points,
             const int &domain_);
  std::vector<double> value(const dealii::Point<dim, double> &P0);
  /*
   * This function gives you the values of half-range basis functions, which
   * will be used in the adaptive meshing. The approach is to give the basis
   * corresponding to the unrefined element neghboring current element. For
   * example consider point x on the edge of element 1, instead of giving the
   * value of bases corresponding to element 1, we will give the value of
   *basis
   * functions of the element 0.
   *
   *               |   0   |
   *               |_______|
   *
   *               |\
   *               | *  <------  we will give this value !
   *               |  \
   *               |-x-\---|
   *                    \  |
   *                     \ |
   *                      \|
   *
   *               |---|---|
   *               | 1 | 2 |
   */

  std::vector<double>
   value(const dealii::Point<dim, double> &P0, const unsigned half_range);
  std::vector<dealii::Tensor<1, dim>> grad(const dealii::Point<dim, double> &P0);
  ~Poly_Basis();

 private:
  std::unique_ptr<Derived_Basis> poly_basis;
};

#include "poly_basis.tpp"

#endif // POLY_BASIS
