#include <string>
#include <vector>
#include <cmath>

#ifndef Jacobi_Polynomials
#define Jacobi_Polynomials

#include "poly_basis.hpp"
#include "support_classes.hpp"

/*!
 * \brief The Legendre polynomial basis.
 * \ingroup basis_funcs
 */
template <int dim>
class Jacobi_Poly_Basis //: public Poly_Basis<Jacobi_Poly_Basis<dim>, dim>
{
 public:
  Jacobi_Poly_Basis() = delete;
  Jacobi_Poly_Basis(const std::vector<dealii::Point<1>> &Supp_Points, const int &domain_);
  Jacobi_Poly_Basis(const unsigned &polyspace_order_,
                    const double &alpha_,
                    const double &beta_,
                    const int &domain_);
  ~Jacobi_Poly_Basis();

  std::vector<double> value(const dealii::Point<dim, double> &P0);
  std::vector<double>
   value(const dealii::Point<dim, double> &P0, const unsigned &half_range);
  std::vector<dealii::Tensor<1, dim>> grad(const dealii::Point<dim, double> &P0);
  std::vector<double> value(const double &);
  std::vector<double> derivative(const double &);

  template <int func_dim>
  void project_to(const Function<func_dim, double> &func,
                  const std::vector<dealii::Point<func_dim>> &integration_points_,
                  const std::vector<double> &weights,
                  Eigen::MatrixXd &vec);

  template <int func_dim, typename T>
  void project_to(const Function<func_dim, T> &func,
                  const std::vector<dealii::Point<func_dim>> &integration_points,
                  const std::vector<dealii::Point<func_dim>> &normals_at_integration,
                  const std::vector<double> &weights,
                  Eigen::MatrixXd &vec);

 private:
  const double integral_sc_fac;
  unsigned int polyspace_order;
  double alpha, beta;
  int domain;
  std::vector<double> compute(const double &x_inp);
  inline double change_coords(const double &x_inp);
};

#include "jacobi_polynomial.tpp"

#endif
