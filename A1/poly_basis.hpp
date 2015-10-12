#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <deal.II/base/tensor.h>
#include <boost/numeric/mtl/mtl.hpp>

#ifndef POLY_BASIS
#define POLY_BASIS

enum Domain
{
  From_0_to_1 = 1 << 0,
  From_minus_1_to_1 = 1 << 1
};

#include "jacobi_polynomial.hpp"
#include "lagrange_polynomial.hpp"
#include "lagrange_polynomial_vandermonde.hpp"
#include "support_classes.hpp"

/*!
 * This structure contains the basis functions, their gradients,
 * and their divergence. The main motivation behind this is to avoid the
 * repeated calculation of bases on a unit cell for every element. This
 * structure has a constructor which takes quadrature points as inputs and
 * stores the corresponding basis.
 */
template <typename Derived_Basis, int dim>
class Poly_Basis
{
 public:
  Poly_Basis() = delete;
  Poly_Basis(const std::vector<dealii::Point<dim>> &integration_points,
             const std::vector<dealii::Point<1, double>> &support_points,
             const int &domain_);
  std::vector<double> value(const dealii::Point<dim, double> &P0);
  std::vector<double>
   value(const dealii::Point<dim, double> &P0, const unsigned half_range);
  std::vector<dealii::Tensor<1, dim>> grad(const dealii::Point<dim, double> &P0);
  ~Poly_Basis();

  unsigned n_polys;
  std::vector<std::vector<double>> bases;
  std::vector<std::vector<dealii::Tensor<1, dim>>> bases_grads;
  Eigen::MatrixXd the_bases;
  mtl::dense2D<dealii::Tensor<1, dim>> the_bases_grads;

  template <int func_dim, typename T>
  void Project_to_Basis(const Function<func_dim, T> &func,
                        const std::vector<dealii::Point<func_dim>> &integration_points,
                        const std::vector<dealii::Point<func_dim>> &support_points,
                        const std::vector<double> &weights,
                        Eigen::MatrixXd &vec);

  template <int func_dim, typename T>
  void
   Project_to_Basis(const Function<func_dim, T> &func,
                    const std::vector<dealii::Point<func_dim>> &integration_points,
                    const std::vector<dealii::Point<func_dim>> &support_points,
                    const std::vector<dealii::Point<func_dim>> &normals_at_integration,
                    const std::vector<dealii::Point<func_dim>> &normals_at_supports,
                    const std::vector<double> &weights,
                    Eigen::MatrixXd &vec);

 private:
  Derived_Basis poly_basis;
};

#include "poly_basis.tpp"

#endif // POLY_BASIS
