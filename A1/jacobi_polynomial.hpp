#include "poly_basis.hpp"
#include <string>
#include <vector>
#include <deal.II/base/tensor.h>
#include <cmath>

#ifndef Jacobi_Polynomials
#define Jacobi_Polynomials

template <int dim, int spacedim = dim>
class JacobiP : public Poly_Basis<dim, spacedim>
{
 public:
  JacobiP();
  JacobiP(const JacobiP &) = delete;
  JacobiP &operator=(const JacobiP &) = delete;
  JacobiP(const int &, const double &, const double &, const int);
  ~JacobiP();

  std::string get_type()
  {
    return "Using Legendre polynomials.";
  }

  std::vector<double> value(double) const;
  std::vector<double> derivative(double) const;

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

  enum Domain
  {
    From_0_to_1 = 1 << 0,
    From_minus_1_to_1 = 1 << 1
  };

 private:
  const double integral_sc_fac;
  unsigned int n;
  double alpha, beta;
  int domain;
  std::vector<double> compute(const double x_inp) const;
  inline double change_coords(double x_inp) const;
};

#include "jacobi_polynomial.tpp"

#endif
