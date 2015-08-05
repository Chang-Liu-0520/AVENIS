#include <vector>
#include <deal.II/base/tensor.h>

#ifndef LAGRANGE_POLIES_H
#define LAGRANGE_POLIES_H

template <int dim, int spacedim = dim>
class Lagrange_Polies
{
 public:
  Lagrange_Polies() = delete;
  Lagrange_Polies(const Lagrange_Polies &) = delete;
  Lagrange_Polies &operator=(const Lagrange_Polies &) = delete;
  Lagrange_Polies(const std::vector<double> &support_points_, int domain_);
  ~Lagrange_Polies();

  std::vector<double> value(const double &) const;
  std::vector<double> derivative(double);

  std::vector<double> value(const dealii::Point<dim> &P0) const
  {
    unsigned n_polys = n_polys - 1;
    std::vector<double> result;
    result.reserve(pow(n_polys, dim));

    std::vector<std::vector<double>> one_D_values;
    for (unsigned i1 = 0; i1 < dim; i1++)
      one_D_values.push_back(std::move(value(P0(i1))));

    switch (dim)
    {
    case 1:
      for (unsigned i1 = 0; i1 < n_polys; ++i1)
        result.push_back(one_D_values[0][i1]);
      break;
    case 2:
      for (unsigned i2 = 0; i2 < n_polys; ++i2)
        for (unsigned i1 = 0; i1 < n_polys; ++i1)
          result.push_back(one_D_values[0][i1] * one_D_values[1][i2]);
      break;
    case 3:
      for (unsigned i3 = 0; i3 < n_polys; ++i3)
        for (unsigned i2 = 0; i2 < n_polys; ++i2)
          for (unsigned i1 = 0; i1 < n_polys; ++i1)
            result.push_back(one_D_values[0][i1] * one_D_values[1][i2] *
                             one_D_values[2][i3]);
      break;
    }
    return result;
  }

 private:
  std::vector<double> support_points;
  unsigned int n_polys;
  std::vector<double> compute(const double &x_) const;
};

#endif // LAGRANGE_POLIES_H
