#include "support_classes.hpp"
#include <Eigen/Dense>
#include <deal.II/base/tensor.h>

#ifndef INPUT_DATA_HPP
#define INPUT_DATA_HPP

template <int dim, typename T>
struct kappa_inv_class : public Function<dim, T>
{
  virtual T value(const dealii::Point<dim> &x, const dealii::Point<dim> &n) const
  {
    Eigen::Matrix2d kappa_inv_;
    kappa_inv_ << 1.0 / exp(x[0] + x[1]), 0.0, 0.0, 1.0 / exp(x[0] - x[1]);

    //   Result set 1.
    /*
    kappa_inv_ << 1, 0.0, 0.0, 1;
    if (x[0] > -Mat_Change)
      kappa_inv_ << 1E4, 0.0, 0.0, 1E4;
    */

    //   Result set 2.
    /*
    Eigen::MatrixXd kappa_inv_ = Eigen::MatrixXd::Zero(dim, dim);
    kappa_inv_ << 10, 0.0, 0.0, 10;
    if (x[0] < -Mat_Change || x[0] > Mat_Change)
      kappa_inv_ << 0.1, 0.0, 0.0, 0.1;
    if (x[0] > -Mat_Change && x[0] < Mat_Change && x[1] > -Mat_Change / 10.0 &&
        x[1] < Mat_Change / 10.0)
      kappa_inv_ << 0.01, 0.0, 0.0, 10.0;
    */

    //    Result set 3.
    /*
    Eigen::MatrixXd kappa_inv_ = Eigen::MatrixXd::Zero(dim, dim);
    double Mat_Change = M_PI / 10.0;
    kappa_inv_ << 1, 0.0, 0.0, 0.001;
    double fac1 = sqrt(2.0) / 2.0;
    dealii::Tensor<2, 2> Rot_Mat({ { fac1, fac1 }, { -fac1, fac1 } });
    Eigen::Matrix2d Rot_Mat2;
    Rot_Mat2 << fac1, fac1, -fac1, fac1;
    dealii::Point<2> x_prime = Rot_Mat * x;
    if (x_prime[1] > -Mat_Change / 10 && x_prime[1] < Mat_Change / 10 &&
        x[0] > -Mat_Change * 2.5 && x[0] < Mat_Change * 2.5)
    {
      Eigen::Matrix2d kappa_inv_2;
      kappa_inv_2 << 0.001, 0.0, 0.0, 1000;
      kappa_inv_ = Rot_Mat2.transpose() * kappa_inv_2 * Rot_Mat2;
    }
    */

    //    Result set 4.
    /*
    Eigen::MatrixXd kappa_inv_ = Eigen::MatrixXd::Zero(dim, dim);
    double Mat_Change = M_PI / 10.0;
    kappa_inv_ << 10.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1;
    double fac1 = sqrt(2.0) / 2.0;
    double fac2 = sqrt(3.0) / 3.0;
    dealii::Tensor<2, 3> Rot_Mat({ { fac2, fac2, fac2 }, { -fac1, fac1, 0.0 },
                                   { -fac1 * fac2, -fac1 * fac2, 2 * fac1 *
    fac2
    } });
    Eigen::Matrix3d Rot_Mat2;
    Rot_Mat2 << fac2, fac2, fac2, -fac1, fac1, 0.0, -fac1 *fac2, -fac1 *fac2,
     2 * fac1 *fac2;
    dealii::Point<3> x_prime = Rot_Mat * x;
    double pipe_radius = sqrt(x_prime[1] * x_prime[1] + x_prime[2] *
    x_prime[2]);
    if (pipe_radius < Mat_Change / 5 && x[0] > -Mat_Change * 2.5 &&
        x[0] < Mat_Change * 2.5)
    {
      Eigen::Matrix3d kappa_inv_2;
      kappa_inv_2 << 0.1, 0.0, 0.0, 0.0, 10, 0.0, 0.0, 0.0, 10;
      kappa_inv_ = Rot_Mat2.transpose() * kappa_inv_2 * Rot_Mat2;
    }
    */
    return kappa_inv_;
  }
};

template <int dim, typename T>
struct u_func_class : public Function<dim, T>
{
  virtual T value(const dealii::Point<dim> &x, const dealii::Point<dim> &n) const
  {
    double u_func = 0;
    if (dim == 2)
      u_func = sin(M_PI * x[0]) * cos(M_PI * x[1]);
    if (dim == 3)
      u_func = sin(M_PI * x[0]) * cos(M_PI * x[1]) * sin(M_PI * x[2]);
    return u_func;
  }
};

template <int dim, typename T>
struct q_func_class : public Function<dim, T>
{
  virtual T value(const dealii::Point<dim> &x, const dealii::Point<dim> &n) const
  {
    dealii::Tensor<1, dim> q_func;
    if (dim == 2)
    {
      q_func[0] = -exp(x[0] + x[1]) * M_PI * cos(M_PI * x[0]) * cos(M_PI * x[1]);
      q_func[1] = exp(x[0] - x[1]) * M_PI * sin(M_PI * x[0]) * sin(M_PI * x[1]);
    }
    if (dim == 3)
    {
      q_func[0] = -M_PI * cos(M_PI * x[0]) * cos(M_PI * x[1]) * sin(M_PI * x[2]);
      q_func[1] = M_PI * sin(M_PI * x[0]) * sin(M_PI * x[1]) * sin(M_PI * x[2]);
      q_func[2] = -M_PI * sin(M_PI * x[0]) * cos(M_PI * x[1]) * cos(M_PI * x[2]);
    }

    return q_func;
  }
};

template <int dim, typename T>
struct divq_func_class : public Function<dim, T>
{
  virtual T value(const dealii::Point<dim> &x, const dealii::Point<dim> &n) const
  {
    if (dim == 3)
      return 3 * M_PI * M_PI * sin(M_PI * x[0]) * cos(M_PI * x[1]) * sin(M_PI * x[2]);
    return 2 * M_PI * M_PI * sin(M_PI * x(0)) * cos(M_PI * x(1));
  }
};

template <int dim, typename T>
struct f_func_class : public Function<dim, T>
{
  virtual T value(const dealii::Point<dim> &x, const dealii::Point<dim> &n) const
  {
    double f_func = 0;
    if (dim == 2)
      f_func = M_PI * M_PI * sin(M_PI * x[0]) * cos(M_PI * x[1]) *
                 (exp(x[0] + x[1]) + exp(x[0] - x[1])) -
               M_PI * exp(x[0] + x[1]) * cos(M_PI * x[0]) * cos(M_PI * x[1]) -
               M_PI * exp(x[0] - x[1]) * sin(M_PI * x[0]) * sin(M_PI * x[1]);
    if (dim == 3)
      f_func =
        3 * M_PI * M_PI * sin(M_PI * x[0]) * cos(M_PI * x[1]) * sin(M_PI * x[2]);

    //    f_func = 0;

    return f_func;
  }
};

template <int dim, typename T>
struct Dirichlet_BC_func_class : public Function<dim, T>
{
  u_func_class<dim, T> u_func;
  virtual T value(const dealii::Point<dim> &x, const dealii::Point<dim> &n) const
  {
    double gD;
    gD = 0;
    if (x[0] < -1 + 1E-10)
      gD = 10;

    gD = u_func.value(x, x);

    return gD;
  }
};

template <int dim, typename T>
struct Neumann_BC_func_class : public Function<dim, T>
{
  virtual T value(const dealii::Point<dim> &x, const dealii::Point<dim> &n) const
  {
    q_func_class<dim, dealii::Tensor<1, dim>> q_func;
    double gN;
    gN = 0;
    if (x[1] < -1.0 + 1.0E-10 || x[1] > 1.0 - 1.0E-10)
      gN = 0;

    gN = q_func.value(x, x) * n;

    return gN;
  }
};

#endif // INPUT_DATA_HPP
