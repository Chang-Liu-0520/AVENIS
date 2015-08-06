#include <string>
#include <vector>
#include <deal.II/base/tensor.h>
#include <cmath>

#ifndef POLY_BASIS
#define POLY_BASIS

template <int dim, int spacedim = dim>
class Poly_Basis
{
 public:
  Poly_Basis();
  Poly_Basis(const Poly_Basis &);
  Poly_Basis &operator=(const Poly_Basis &);

  std::vector<double> value(const dealii::Point<dim> &P0) const;

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
  value(const dealii::Point<dim> &P0, const unsigned half_range) const;

  std::vector<dealii::Tensor<1, dim>> grad(const dealii::Point<dim> &P0) const;

  ~Poly_Basis();
};

#endif // POLY_BASIS
