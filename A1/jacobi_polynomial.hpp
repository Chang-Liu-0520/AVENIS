#include <string>
#include <vector>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <cmath>

#ifndef Jacobi_Polynomials
#define Jacobi_Polynomials

class JacobiP
{
 public:
  JacobiP() = delete;
  JacobiP(const JacobiP &) = delete;
  JacobiP &operator=(const JacobiP &) = delete;
  JacobiP(const int &, const double &, const double &, const int);
  std::vector<double> value(double) const;
  std::vector<double> derivative(double);

  template <int dim>
  std::vector<double> value(const dealii::Point<dim> &P0) const
  {
    std::vector<double> result;
    result.reserve(pow(n + 1, dim));

    std::vector<std::vector<double>> one_D_values;
    for (unsigned i1 = 0; i1 < dim; i1++)
      one_D_values.push_back(std::move(value(P0(i1))));

    switch (dim)
    {
    case 1:
      for (unsigned i1 = 0; i1 < n + 1; ++i1)
        result.push_back(one_D_values[0][i1]);
      break;
    case 2:
      for (unsigned i2 = 0; i2 < n + 1; ++i2)
        for (unsigned i1 = 0; i1 < n + 1; ++i1)
          result.push_back(one_D_values[0][i1] * one_D_values[1][i2]);
      break;
    case 3:
      for (unsigned i3 = 0; i3 < n + 1; ++i3)
        for (unsigned i2 = 0; i2 < n + 1; ++i2)
          for (unsigned i1 = 0; i1 < n + 1; ++i1)
            result.push_back(one_D_values[0][i1] * one_D_values[1][i2] *
                             one_D_values[2][i3]);
      break;
    }
    return result;
  }

  /*
   * This function gives you the values of half-range basis functions, which
   * will be used in the adaptive meshing. The approach is to give the basis
   * corresponding to the unrefined element neghboring current element. For
   * example consider point x on the edge of element 1, instead of giving the
   * value of bases corresponding to element 1, we will give the value of basis
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
  template <int dim>
  std::vector<double> value(const dealii::Point<dim> &P0, const unsigned half_range) const
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
          dealii::Point<dim> P0_mod(P0(0) / 2.0);
          result = value(P0_mod);
        }
        if (half_range == 2)
        {
          dealii::Point<dim> P0_mod(0.5 + P0(0) / 2.0);
          result = value(P0_mod);
        }
      }
      if (P0.dimension == 2)
      {
        if (half_range == 1)
        {
          dealii::Point<dim> P0_mod(P0(0) / 2.0, P0(1) / 2.0);
          result = value(P0_mod);
        }
        if (half_range == 2)
        {
          dealii::Point<dim> P0_mod(0.5 + P0(0) / 2.0, P0(1) / 2.0);
          result = value(P0_mod);
        }
        if (half_range == 3)
        {
          dealii::Point<dim> P0_mod(P0(0) / 2.0, 0.5 + P0(1) / 2.0);
          result = value(P0_mod);
        }
        if (half_range == 4)
        {
          dealii::Point<dim> P0_mod(0.5 + P0(0) / 2.0, 0.5 + P0(1) / 2.0);
          result = value(P0_mod);
        }
      }
    }
    return result;
  }

  template <int dim>
  std::vector<dealii::Tensor<1, dim>> grad(const dealii::Point<dim> &P0)
  {
    std::vector<dealii::Tensor<1, dim>> grad;
    grad.reserve(pow(n + 1, dim));

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
      for (unsigned i1 = 0; i1 < n + 1; ++i1)
      {
        grad_N[0] = one_D_grads[0][i1];
        grad.push_back(std::move(grad_N));
      }
      break;
    case 2:
      for (unsigned i2 = 0; i2 < n + 1; ++i2)
        for (unsigned i1 = 0; i1 < n + 1; ++i1)
        {
          grad_N[0] = one_D_grads[0][i1] * one_D_values[1][i2];
          grad_N[1] = one_D_values[0][i1] * one_D_grads[1][i2];
          grad.push_back(std::move(grad_N));
        }
      break;
    case 3:
      for (unsigned i3 = 0; i3 < n + 1; ++i3)
        for (unsigned i2 = 0; i2 < n + 1; ++i2)
          for (unsigned i1 = 0; i1 < n + 1; ++i1)
          {
            grad_N[0] =
              one_D_grads[0][i1] * one_D_values[1][i2] * one_D_values[2][i3];
            grad_N[1] =
              one_D_values[0][i1] * one_D_grads[1][i2] * one_D_values[2][i3];
            grad_N[2] =
              one_D_values[0][i1] * one_D_values[1][i2] * one_D_grads[2][i3];
            grad.push_back(std::move(grad_N));
          }

      break;
    }
    return grad;
  }

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

#endif
