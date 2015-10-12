#include "poly_basis.hpp"

/*!
 * As you can see, you can have different values for number of quadrature
 * points and number of basis functions.
 * By execution of this constructor the bases[i][j] will contain the value
 * of jth basis function at ith point. You can assume that functions are
 * stored in different columns of the same row. Also, different rows
 * correspond to different points.
 *
 *                                function j
 *                                    |
 *                                    |
 *
 *                            x  ...  x  ...  x
 *                            .       .       .
 *                            .       :       .
 *                            .       x       .
 *          point i ---->     x      Bij ...  x
 *                            .       x       .
 *                            .       :       .
 *                            .       .       .
 *                            x  ...  x  ...  x
 *
 *
 *    ======  Now let us consider the matrix B_ij = bases[i][j]  ======
 * To convert from modal to nodal (where modes are stored in a column vector),
 * use:
 *
 *                               Ni = Bij * Mj
 *
 */
template <typename Derived_Basis, int dim>
poly_space_basis<Derived_Basis, dim>::poly_space_basis(
 const std::vector<dealii::Point<dim>> &integration_points_,
 const std::vector<dealii::Point<1, double>> &support_points_,
 const int &domain_)
  : n_polys(pow(support_points_.size(), dim)),
    the_bases(integration_points_.size(), n_polys),
    the_bases_grads(integration_points_.size(), n_polys),
    poly_basis(Derived_Basis(support_points_, domain_))
{
  for (unsigned i1 = 0; i1 < integration_points_.size(); ++i1)
  {
    dealii::Point<dim, double> p0 = integration_points_[i1];
    std::vector<double> Ni = value(p0);
    std::vector<dealii::Tensor<1, dim>> Ni_grad = grad(p0);
    bases.push_back(Ni);
    bases_grads.push_back(Ni_grad);
    for (unsigned i_poly = 0; i_poly < Ni.size(); ++i_poly)
    {
      the_bases(i1, i_poly) = Ni[i_poly];
      the_bases_grads(i1, i_poly) = Ni_grad[i_poly];
    }
  }
}

/*!
 * This function projects the function "func" to the current basis.
 * Since, this is a regular integration, we do not need the JxW. We
 * need only the weights.
 * Note for example that: \f[f(x)=\sum_{i} \alpha_i N_i(x) \Longrightarrow
 * \left(f(x),N_j(x)\right) = \sum_i \alpha_i \left(N_i(x),N(x)\right).\f]
 * Now, if \f$N_i\f$'s are orthonormal to each other, then \f$ \alpha_i =
 * (f,N_i)\f$. Otherwise, we are using a Lagrangian basis and in order to
 * project a function onto this basis, we need to just calculate the value
 * of the function at the corresponding nodal points. Finally, it is
 * worthwhile
 * noting that, the parameter @c func_dim is not always the same as
 * <code>dim</code> (from the containing class). The reason is we want to for
 * example calculte \f$\hat u\f$ with the same function as we calculate
 * \f$u\f$. So, although the space of \f$\hat u\f$ is a \f$d-1\f$-dimonsional
 * space, we evaluate its value using a \f$d\f$-dimensional function.
 * \tparam func_dim is the dimension of the @c func argument. This dimension
 * has no utility whatsoever!
 * @param func is
 * @param integration_points is the input integration points into this
 * function. We will assert if the number of the integration points in this
 * function is the same as the number of rows of the:
 * BasisIntegrator_Matrix#bases.
 * @param weights is the integration weights corresponding to different points
 * in the @c integration_points argument.
 */
template <typename Derived_Basis, int dim>
template <int func_dim, typename T>
void poly_space_basis<Derived_Basis, dim>::Project_to_Basis(
 const Function<func_dim, T> &func,
 const std::vector<dealii::Point<func_dim>> &integration_points_,
 const std::vector<dealii::Point<func_dim>> &support_points_,
 const std::vector<double> &weights,
 Eigen::MatrixXd &vec)
{
  if (std::is_same<Jacobi_Poly_Basis<dim>, Derived_Basis>::value)
  {
    assert(bases.size() == integration_points_.size());
    assert(integration_points_.size() == weights.size());
    vec = Eigen::MatrixXd::Zero(n_polys, 1);
    for (unsigned i1 = 0; i1 < weights.size(); ++i1)
    {
      Eigen::MatrixXd Nj(n_polys, 1);
      Nj = Eigen::VectorXd::Map(bases[i1].data(), n_polys);
      vec += weights[i1] *
             func.value(integration_points_[i1], integration_points_[i1]) * Nj;
    }
  }
  else if (std::is_same<Lagrange_Polys<dim>, Derived_Basis>::value)
  {
    assert(support_points_.size() == n_polys);
    vec = Eigen::MatrixXd::Zero(n_polys, 1);
    unsigned counter = 0;
    for (auto &&support_point : support_points_)
      vec(counter++, 0) = func.value(support_point, support_point);
  }
}

template <typename Derived_Basis, int dim>
template <int func_dim, typename T>
void poly_space_basis<Derived_Basis, dim>::Project_to_Basis(
 const Function<func_dim, T> &func,
 const std::vector<dealii::Point<func_dim>> &integration_points_,
 const std::vector<dealii::Point<func_dim>> &support_points_,
 const std::vector<dealii::Point<func_dim>> &normals_at_integration_,
 const std::vector<dealii::Point<func_dim>> &normals_at_supports_,
 const std::vector<double> &weights_,
 Eigen::MatrixXd &vec)
{
  if (std::is_same<Jacobi_Poly_Basis<dim>, Derived_Basis>::value)
  {
    assert(bases.size() == integration_points_.size());
    assert(integration_points_.size() == weights_.size());
    vec = Eigen::MatrixXd::Zero(n_polys, 1);
    for (unsigned i1 = 0; i1 < weights_.size(); ++i1)
    {
      Eigen::MatrixXd Nj(n_polys, 1);
      Nj = Eigen::VectorXd::Map(bases[i1].data(), n_polys);
      vec += weights_[i1] *
             func.value(integration_points_[i1], normals_at_integration_[i1]) * Nj;
    }
  }
  else if (std::is_same<Lagrange_Polys<dim>, Derived_Basis>::value)
  {
    assert(support_points_.size() == n_polys);
    vec = Eigen::MatrixXd::Zero(n_polys, 1);
    unsigned counter = 0;
    for (auto &&support_point : support_points_)
    {
      vec(counter, 0) = func.value(support_point, normals_at_supports_[counter]);
      counter++;
    }
  }
}

template <typename Derived_Basis, int dim>
std::vector<double>
 poly_space_basis<Derived_Basis, dim>::value(const dealii::Point<dim, double> &P0)
{
  return poly_basis.value(P0);
}

/*!
 * This function gives you the values of half-range basis functions, which
 * will be used in the adaptive meshing. The approach is to give the basis
 * corresponding to the unrefined element neghboring current element. For
 * example consider point x on the edge of element 1, instead of giving the
 * value of bases corresponding to element 1, we will give the value of
 * basis functions of the element 0.
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
template <typename Derived_Basis, int dim>
std::vector<double>
 poly_space_basis<Derived_Basis, dim>::value(const dealii::Point<dim, double> &P0,
                                       const unsigned half_range)
{
  return poly_basis.value(P0, half_range);
}

template <typename Derived_Basis, int dim>
std::vector<dealii::Tensor<1, dim>>
 poly_space_basis<Derived_Basis, dim>::grad(const dealii::Point<dim, double> &P0)
{
  return poly_basis.grad(P0);
}

template <typename Derived_Basis, int dim>
poly_space_basis<Derived_Basis, dim>::~poly_space_basis()
{
}
