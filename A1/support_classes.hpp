#include <type_traits>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <Eigen/Dense>

#include "poly_basis.hpp"

#ifndef SUPPORT_CLASSES
#define SUPPORT_CLASSES


const std::string currentDateTime();

template <int dim, typename T, int spacedim = dim>
struct Function
{
  Function();
  virtual ~Function();
  virtual T value(const dealii::Point<dim> &x, const dealii::Point<dim> &n) const = 0;
};

template <int dim, int spacedim = dim>
struct Cell_Class
{
  enum BC
  {
    Dirichlet = 1 << 0,
    Neumann = 1 << 1
  };

  typedef dealii::TriaActiveIterator<dealii::CellAccessor<dim, spacedim>> dealii_Cell_Type;
  typedef typename std::vector<Cell_Class>::iterator vec_iterator_type;
  Cell_Class() = delete;
  Cell_Class(const dealii_Cell_Type &inp_cell, unsigned id_num_);
  Cell_Class(const Cell_Class &inp_cell) = delete;
  Cell_Class(Cell_Class &&inp_cell) noexcept;
  ~Cell_Class();
  void attach_FEValues(std::unique_ptr<dealii::FEValues<dim>> &cell_quad_fe_vals_,
                       std::unique_ptr<dealii::FEFaceValues<dim>> &face_quad_fe_vals_,
                       std::unique_ptr<dealii::FEValues<dim>> &cell_supp_fe_vals_,
                       std::unique_ptr<dealii::FEFaceValues<dim>> &face_supp_fe_vals_);
  void detach_FEValues(std::unique_ptr<dealii::FEValues<dim>> &cell_quad_fe_vals_,
                       std::unique_ptr<dealii::FEFaceValues<dim>> &face_quad_fe_vals_,
                       std::unique_ptr<dealii::FEValues<dim>> &cell_supp_fe_vals_,
                       std::unique_ptr<dealii::FEFaceValues<dim>> &face_supp_fe_vals_);
  void reinit_Cell_FEValues();
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
