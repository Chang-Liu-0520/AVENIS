#include "class_factory.hpp"
#include <string>
#include <vector>
#include <deal.II/base/tensor.h>
#include <cmath>

#ifndef POLY_BASIS
#define POLY_BASIS

template <int dim, int spacedim = dim>
class Poly_Basis : public Base_Template<Poly_Basis<dim, spacedim>, std::string>
{
 public:
  Poly_Basis()
  {
  }

  virtual std::string get_type() const = 0;

  /*
   * The function init is only called when we create the object with the empty
   * constructor.
   */
  virtual void init(const std::vector<dealii::Point<dim>> &Points) = 0;

  virtual std::vector<double> value(const dealii::Point<dim> &P0) const = 0;

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
  virtual std::vector<double>
  value(const dealii::Point<dim> &P0, const unsigned half_range) const = 0;

  virtual std::vector<dealii::Tensor<1, dim>>
  grad(const dealii::Point<dim> &P0) const = 0;
  virtual ~Poly_Basis()
  {
  }
};

#endif // POLY_BASIS
