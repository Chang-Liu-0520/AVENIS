//#define EIGEN_USE_MKL_ALL

#include <fstream>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <functional>
#include <memory>
#include <cstdio>
#include <unistd.h>
#include <getopt.h>
#include <memory>

#include <mpi.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscis.h>
#include <petscksp.h>
#include <slepc.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/generic_linear_algebra.h>

#define USE_PETSC_LA
namespace LA
{
#ifdef USE_PETSC_LA
using namespace dealii::LinearAlgebraPETSc;
#else
using namespace ::LinearAlgebraTrilinos;
#endif
}
#include <deal.II/lac/petsc_parallel_vector.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/derivative_form.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/base/polynomials_abf.h>
#include <deal.II/base/function.h>
#include <deal.II/base/multithread_info.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/data_component_interpretation.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <Eigen/SVD>

#include "poly_basis.hpp"
#include "jacobi_polynomial.hpp"
#include "input_data.hpp"
#include "support_classes.hpp"

#ifndef O_N_DIFFUSION
#define O_N_DIFFUSION

/**
 * We want to solve the diffusion equation with hybridized DG. We want to solve,
 * the following equation in \f$\Omega \subset \mathbb R^{d}\f$ (with \f$ d=\f$
 * \c dim):
 * \f[\left\{\begin{aligned}\nabla u + \boldsymbol \kappa^{-1} \mathbf q = 0 &
 *                          \\
 *                          \nabla \cdot \mathbf q = f &
 *           \end{aligned}
 *    \right. \quad \text{in } \Omega.\f]
 * with boundary conditions:
 * \f[
 *   \begin{aligned}
 *     u = g_D & \quad \text{on } \Gamma_D ,\\
 *     \mathbf q \cdot \boldsymbol n = g_N & \quad \text{on } \Gamma_N.
 *   \end{aligned}
 * \f]
 * I will add all the formulation (specially different matrices) will be added
 * here.
 */
template <int dim>
struct Diffusion
{
  static const unsigned n_faces_per_cell = dealii::GeometryInfo<dim>::faces_per_cell;
  typedef typename Cell_Class<dim>::dealii_Cell_Type Cell_Type;
  typedef Jacobi_Poly_Basis<dim> elem_basis_type;
  typedef Jacobi_Poly_Basis<dim - 1> face_basis_type;
  //  typedef Lagrange_Polys<dim> elem_basis_type;
  //  typedef Lagrange_Polys<dim - 1> face_basis_type;

  /*!
   * @brief The constructor of the main class of the program. This constructor
   * takes 6 arguments.
   * @param order The order of the elements.
   * @param comm_ The MPI communicator.
   * @param comm_size_ Number of MPI procs.
   * @param comm_rank_ ID_Num of the current proc.
   * @param n_threads Number of OpenMP threads.
   * @param Adaptive_ON_ A flag which tell to turn on the AMR.
   */
  Diffusion(const unsigned &order,
            const MPI_Comm &comm_,
            const unsigned &comm_size_,
            const unsigned &comm_rank_,
            const unsigned &n_threads,
            const bool &Adaptive_ON_);
  ~Diffusion();

  void FreeUpContainers();
  void OutLogger(std::ostream &logger, const std::string &log, bool insert_eol = true);
  void Refine_Grid(int);
  void Write_Grid_Out();
  void Setup_System(unsigned);
  void Set_Boundary_Indicator();
  PetscErrorCode Solve_Linear_Systam();
  void vtk_visualizer();

  std::vector<Cell_Class<dim>> All_Owned_Cells;
  MPI_Comm comm;
  unsigned comm_size, comm_rank;
  const unsigned poly_order;
  const unsigned quad_order;
  const unsigned n_internal_unknowns;
  const unsigned n_trace_unknowns;
  dealii::parallel::distributed::Triangulation<dim> Grid1;
  dealii::MappingQ1<dim> Elem_Mapping;
  dealii::QGauss<dim> elem_integration_capsul;
  dealii::QGauss<dim - 1> face_integration_capsul;
  dealii::QGaussLobatto<1> LGL_quad_1D;
  const std::vector<dealii::Point<1>> support_points_1D;
  dealii::FE_DGQ<dim> DG_Elem1;
  dealii::FESystem<dim> DG_System1;
  dealii::DoFHandler<dim> DoF_H_Refine;
  dealii::DoFHandler<dim> DoF_H1_System;

  Poly_Basis<elem_basis_type, dim> the_elem_basis;
  Poly_Basis<face_basis_type, dim - 1> the_face_basis;
  unsigned refn_cycle;

  kappa_inv_class<dim, Eigen::MatrixXd> kappa_inv;
  u_func_class<dim, double> u_func;
  q_func_class<dim, dealii::Tensor<1, dim>> q_func;
  divq_func_class<dim, double> divq_func;
  f_func_class<dim, double> f_func;
  Dirichlet_BC_func_class<dim, double> Dirichlet_BC_func;
  Neumann_BC_func_class<dim, double> Neumann_BC_func;

 private:
  const int Dirichlet_BC_Index = 1;
  const int Neumann_BC_Index = 2;
  const bool Adaptive_ON = true;
  void Init_Mesh_Containers();
  void Count_Globals();
  void Assemble_Globals();
  void Calculate_Internal_Unknowns(double *const &local_uhat_vec);

  template <typename T1, typename T2>
  void Internal_Vars_Errors(const Cell_Class<dim> &cell,
                            const T1 &solved_u_vec,
                            const T1 &solved_q_vec,
                            const T2 &Mode_to_Node_Matrix,
                            double &Error_u,
                            double &Error_q,
                            double &Error_div_q);

  void CalculateMatrices(Cell_Class<dim> &cell,
                         Poly_Basis<elem_basis_type, dim> &the_elem_basis);

  template <typename T>
  void Calculate_Postprocess_Matrices(Cell_Class<dim> &cell,
                                      const Poly_Basis<elem_basis_type, dim> &PostProcess_Elem_Basis,
                                      T &DM_star,
                                      T &DB2);

  template <typename T1>
  void PostProcess(Cell_Class<dim> &cell,
                   const Poly_Basis<elem_basis_type, dim> &PostProcess_Elem_Basis,
                   const T1 &u,
                   const T1 &q,
                   T1 &ustar,
                   const T1 &PostProcess_Mode_to_Node_Matrix,
                   double &error_ustar);

  template <typename T>
  void uhat_u_q_to_jth_col(const T &C,
                           const T &E,
                           const T &H,
                           const T &H2,
                           const T &uhat,
                           const T &u,
                           const T &q,
                           const T &g_N,
                           const double &multiplier,
                           std::vector<double> &jth_col);

  template <typename T, typename U>
  void u_from_uhat_f(const U &LDLT_of_BT_Ainv_B_plus_D,
                     const T &BT_Ainv,
                     const T &C,
                     const T &E,
                     const T &M,
                     const T &uhat,
                     const T &lambda,
                     const T &f,
                     T &u);

  template <typename T, typename U>
  void q_from_u_uhat(
   const U &LDLT_of_A, const T &B, const T &C, const T &uhat, const T &u, T &q);

  void Compute_Error(const Function<dim, double> &func,
                     const std::vector<dealii::Point<dim>> &points_loc,
                     const std::vector<double> &JxWs,
                     const Eigen::MatrixXd &modal_vector,
                     const Eigen::MatrixXd &mode_to_Qpoint_matrix,
                     double &error);

  void Compute_Error(const Function<dim, dealii::Tensor<1, dim>> &func,
                     const std::vector<dealii::Point<dim>> &points_loc,
                     const std::vector<double> &JxWs,
                     const Eigen::MatrixXd &modal_vector,
                     const Eigen::MatrixXd &mode_to_Qpoint_matrix,
                     double &error);

  unsigned n_ghost_cell;
  unsigned n_active_cell;
  unsigned num_global_DOFs_on_this_rank;
  unsigned num_local_DOFs_on_this_rank;
  unsigned num_global_DOFs_on_all_ranks;
  unsigned n_threads;

  /* The next two variables contain num faces from rank zero to the
   * current rank, including and excluding current rank
   */
  std::vector<unsigned> face_count_before_rank;
  std::vector<unsigned> face_count_up_to_rank;
  std::vector<int> n_local_DOFs_connected_to_DOF;
  std::vector<int> n_nonlocal_DOFs_connected_to_DOF;
  std::vector<int> scatter_from, scatter_to;
  std::vector<double> taus;
  std::map<std::string, int> cell_ID_to_num;
  std::map<unsigned, std::vector<std::string>> face_to_rank_sender;
  std::map<unsigned, unsigned> face_to_rank_recver;

  Vec solution_vec, RHS_vec, exact_solution;
  Mat global_mat;
  LA::MPI::Vector refn_solu, elem_solu;

  std::ofstream Convergence_Result;
  std::ofstream Execution_Time;
};

void Tokenize(const std::string &str_in,
              std::vector<std::string> &tokens,
              const std::string &delimiters);

#include "grid_operations.tpp"
#include "diffusion.tpp"

#endif // O_N_DIFFUSION
