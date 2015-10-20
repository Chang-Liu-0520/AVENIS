#include <type_traits>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <Eigen/Dense>

#include "poly_basis.hpp"

#ifndef SUPPORT_CLASSES
#define SUPPORT_CLASSES


const std::string currentDateTime();

/*!
 * \defgroup Functions
 * \brief
 * All of the predefined functions in the code.
 * \details
 * As a reminder we are solving \f[\begin{aligned}\kappa_{ij} u_{,i} &= q_j \\
 * q_{j,j} &= f \end{aligned} \quad \text{in } \Omega\f] with boundary
 * conditions:
 * \f[\begin{aligned} u &= g_D \quad \text{on } \Gamma_D, \\
 *         q_{,i}n_{,i} &= g_N \quad \text{on } \Gamma_N.
 * \end{aligned}\f]
 *
 * This group contains classes which will be used as predefined functions of
 * space or time (like \f$\kappa_{ij}\f$ in diffusion or the boundary condition
 * or the forcing function). This function will be used in such methods as
 * project_to_basis or compute_error.
 */

/*!
 * \defgroup cells Cell data
 * \brief
 * This group contains the classes which encapsulate data corresponding to each
 * cell in the mesh.
 */

/*!
 * \ingroup Functions
 * \details This is the generic abstract base struct for all other functions.
 */
template <int dim, typename T, int spacedim = dim>
struct Function
{
  Function();
  virtual ~Function();
  virtual T value(const dealii::Point<dim> &x, const dealii::Point<dim> &n) const = 0;
};

/*!
 * \brief The \c Cell_Class contains most of the required data about a generic
 * element in the mesh.
 *
 * \ingroup cells
 */
template <int dim, int spacedim = dim>
struct Cell_Class
{
  /*!
   * \details This enum contains the boundary condition on a given face of the
   * element. When there is no BC applied, its value is zero.
   */
  enum BC
  {
    /// Duh! the Dirichlet BC.
    Dirichlet = 1 << 0,
    /// Yes! the Neumann BC.
    Neumann = 1 << 1
  };

  /*!
   * \details
   * We define this type to access the deal.II cell class easily.
   */
  typedef dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim>> dealii_Cell_Type;
  /*!
   * \details
   * This typedef is also for easy access to vector iterator type.
   */
  typedef typename std::vector<Cell_Class>::iterator vec_iterator_type;
  /*!
   * \details
   * We remove the default constructor to avoid uninitialized creation of Cell
   * objects.
   */
  Cell_Class() = delete;
  /*!
   * \details
   * The constructor of this class takes a deal.II cell and its deal.II ID.
   * \param inp_cell The iterator to the deal.II cell in the mesh.
   * \param id_num_  The unique ID (\c dealii_Cell::id()) of the dealii_Cell.
   * This is necessary when working on a distributed mesh.
   */
  Cell_Class(const dealii_Cell_Type &inp_cell, unsigned id_num_);
  /*!
   * \details
   * We remove the copy constructor of this class to avoid unnecessary copies
   * (specially unintentional ones). Up to October 2015, this copy constructor
   * was not useful anywhere in the code.
   */
  Cell_Class(const Cell_Class &inp_cell) = delete;
  /*!
   * \details
   * We need a move constructor, to be able to pass this class as function
   * arguments efficiently. Maybe, you say that this does not help efficiency
   * that much, but we are using it for semantic constraints.
   * \param inp_cell An object of the \c Cell_Class type which we steal its
   * guts.
   */
  Cell_Class(Cell_Class &&inp_cell) noexcept;
  /*!
   * Obviously, the destructor.
   */
  ~Cell_Class();
  /*!
   * \details We attach a \c unique_ptr of dealii::FEValues and
   * dealii::FEFaceValues to the current object.
   * \param cell_quad_fe_vals_ The dealii::FEValues which is used for location
   * of quadrature points in cells.
   * \param face_quad_fe_vals_ The dealii::FEValues which is used for lacation
   * of support points in cells.
   * \param cell_supp_fe_vals_ The dealii::FEValues which is used for location
   * of quadrature points on faces.
   * \param face_supp_fe_vals_ The dealii::FEValues which is used for location
   * of support points on faces.
   */
  void attach_FEValues(std::unique_ptr<dealii::FEValues<dim>> &cell_quad_fe_vals_,
                       std::unique_ptr<dealii::FEFaceValues<dim>> &face_quad_fe_vals_,
                       std::unique_ptr<dealii::FEValues<dim>> &cell_supp_fe_vals_,
                       std::unique_ptr<dealii::FEFaceValues<dim>> &face_supp_fe_vals_);
  /*!
   * \details We detach the \c unique_ptr of dealii::FEValues and
   * dealii::FEFaceValues from the current object. parameters are similar to the
   * \c Cell_Class::attach_FEValues.
  */
  void detach_FEValues(std::unique_ptr<dealii::FEValues<dim>> &cell_quad_fe_vals_,
                       std::unique_ptr<dealii::FEFaceValues<dim>> &face_quad_fe_vals_,
                       std::unique_ptr<dealii::FEValues<dim>> &cell_supp_fe_vals_,
                       std::unique_ptr<dealii::FEFaceValues<dim>> &face_supp_fe_vals_);
  /*!
   * \details Updates the FEValues which are connected to the current element
   * (not the FEFaceValues.)
   */
  void reinit_Cell_FEValues();
  /*!
   * \details Updates the FEFaceValues which are connected to a given face of
   * the current element.
   * \param i_face the face which we want to update the connected FEFaceValues.
   * \c i_face\f$\in\{1,2,3,4\}\f$
   */
  void reinit_Face_FEValues(unsigned i_face);

  template <typename T>
  void assign_matrices(T &&A_, T &&B_, T &&C_, T &&D_, T &&E_, T &&H_, T &&H2_, T &&M_);
  template <typename T>
  void get_matrices(T &A_, T &B_, T &C_, T &D_, T &E_, T &H_, T &H2_, T &M_);
  Eigen::MatrixXd A, B, C, D, E, H, H2, M;
  bool matrices_calculated;

  const unsigned n_faces;
  unsigned id_num;
  std::string cell_id;
  std::vector<unsigned> half_range_flag;
  std::vector<unsigned> face_owner_rank;
  dealii_Cell_Type dealii_Cell;
  std::vector<int> Face_ID_in_this_rank;
  std::vector<int> Face_ID_in_all_ranks;
  std::vector<BC> BCs;
  std::unique_ptr<dealii::FEValues<dim>> cell_quad_fe_vals, cell_supp_fe_vals;
  std::unique_ptr<dealii::FEFaceValues<dim>> face_quad_fe_vals, face_supp_fe_vals;
};

template <int dim, int spacedim = dim>
struct Face_Class
{
  Face_Class();
  unsigned n_local_connected_faces;
  unsigned n_nonlocal_connected_faces;
  unsigned n_local_connected_DOFs;
  unsigned n_nonlocal_connected_DOFs;
  unsigned num_global_DOFs;
  int owner_rank_id;
  std::vector<typename Cell_Class<dim>::vec_iterator_type> Parent_Cells;
  std::vector<unsigned> connected_face_of_parent_cell;
  std::vector<typename Cell_Class<dim>::vec_iterator_type> Parent_Ghosts;
  std::vector<unsigned> connected_face_of_parent_ghost;
};

#include "support_classes.tpp"

#endif // SUPPORT_CLASSES
