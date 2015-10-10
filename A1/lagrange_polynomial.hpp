#include <vector>
#include <deal.II/base/tensor.h>

#ifndef LAGRANGE_POLIES_H
#define LAGRANGE_POLIES_H

template <int dim>
class Lagrange_Polys
{
 public:
  Lagrange_Polys() = delete;
  Lagrange_Polys(const std::vector<dealii::Point<1, double>> &support_points_,
                 int domain_);
  ~Lagrange_Polys();
  std::vector<double> value(const dealii::Point<dim, double> &P0) const;
  std::vector<double> value(const dealii::Point<dim, double> &P0,
                            const unsigned &half_range) const;
  std::vector<double> value(const double &) const;
  std::vector<dealii::Tensor<1, dim>> grad(const dealii::Point<dim, double> &P0) const;

 private:
  std::vector<double> compute(const double &x_) const;
  double compute_Li(const double &x_, const unsigned &i_poly) const;
  std::vector<double> derivative(double) const;

  std::vector<dealii::Point<1>> support_points;
  unsigned int polyspace_order;
  int domain;
};

#include "lagrange_polynomial.tpp"

#endif // LAGRANGE_POLIES_H
