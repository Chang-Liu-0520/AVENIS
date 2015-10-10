#include <vector>
#include <deal.II/base/tensor.h>
#include <Eigen/Dense>
#include "jacobi_polynomial.hpp"

#ifndef LAGRANGE_POLIES_VANDERMONDE_H
#define LAGRANGE_POLIES_VANDERMONDE_H

template <int dim>
class Lagrange_Polys_Vandermonde
{
 public:
  Lagrange_Polys_Vandermonde() = delete;
  Lagrange_Polys_Vandermonde(const std::vector<dealii::Point<1, double>> &support_points_,
                             int domain_);
  ~Lagrange_Polys_Vandermonde();
  std::vector<double> value(const dealii::Point<dim, double> &P0);
  std::vector<double> value(const double &);
  std::vector<dealii::Tensor<1, dim>> grad(const dealii::Point<dim, double> &P0);

 private:
  std::vector<double> compute(const double &x_);
  std::vector<double> derivative(const double &);

  std::vector<dealii::Point<1>> support_points;
  unsigned int polyspace_order;
  int domain;
  Jacobi_Poly_Basis<1> jacobi_poly;
  Eigen::MatrixXd Vandermonde_T_inv;
};

#include "lagrange_polynomial_vandermonde.tpp"

#endif // LAGRANGE_POLIES_H
