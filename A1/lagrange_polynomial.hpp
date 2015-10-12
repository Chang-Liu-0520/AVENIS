#include <vector>
#include <deal.II/base/tensor.h>
#include <Eigen/Dense>

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

  /*
  template <int func_dim>
  void project_to(const Function<func_dim, double> &func,
                  const std::vector<dealii::Point<func_dim>> &support_points_,
                  const std::vector<double> &weights,
                  Eigen::MatrixXd &vec);

  template <int func_dim, typename T>
  void project_to(const Function<func_dim, T> &func,
                  const std::vector<dealii::Point<func_dim>> &support_points_,
                  const std::vector<dealii::Point<func_dim>>
  &normals_at_supports_,
                  const std::vector<double> &weights_,
                  Eigen::MatrixXd &vec);
                  */

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
