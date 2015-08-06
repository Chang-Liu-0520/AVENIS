#include <vector>
#include <deal.II/base/tensor.h>

#ifndef LAGRANGE_POLIES_H
#define LAGRANGE_POLIES_H

template <int dim, int spacedim = dim>
class Lagrange_Polys
{
 public:
  Lagrange_Polys() = delete;
  Lagrange_Polys(const Lagrange_Polys &) = delete;
  Lagrange_Polys &operator=(const Lagrange_Polys &) = delete;
  Lagrange_Polys(const std::vector<double> &support_points_, int domain_);
  ~Lagrange_Polys();

  std::vector<double> value(const double &) const;
  std::vector<double> derivative(double);

  std::vector<double> value(const dealii::Point<dim> &P0) const;
  std::vector<dealii::Tensor<1, dim>> grad(const dealii::Point<dim> &P0) const;

 private:
  std::vector<double> support_points;
  unsigned int n_polys;
  std::vector<double> compute(const double &x_) const;
};

#endif // LAGRANGE_POLIES_H
