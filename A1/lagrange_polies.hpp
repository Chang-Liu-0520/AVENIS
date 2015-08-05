#include <vector>

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

  std::vector<double> value(double) const;
  std::vector<double> derivative(double);

  enum Domain
  {
    From_0_to_1 = 1 << 0,
    From_minus_1_to_1 = 1 << 1
  };

 private:
  std::vector<double> support_points;
  const double integral_sc_fac;
  unsigned int n_polys;
  int domain;
  std::vector<double> compute(const double x_) const;
  double change_coords(double x) const;
};

#endif // LAGRANGE_POLIES_H
