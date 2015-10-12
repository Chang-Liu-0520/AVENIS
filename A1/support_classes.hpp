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
  Function()
  {
  }
  virtual ~Function()
  {
  }
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
  Cell_Class(const dealii_Cell_Type &inp_cell, unsigned id_num_)
    : n_faces(dealii::GeometryInfo<dim>::faces_per_cell),
      id_num(id_num_),
      half_range_flag(n_faces, 0),
      face_owner_rank(n_faces, -1),
      dealii_Cell(inp_cell),
      Face_ID_in_this_rank(n_faces, -2),
      Face_ID_in_all_ranks(n_faces, -2),
      BCs(n_faces)
  {
    pCell_FEValues = nullptr;
    pFace_FEValues = nullptr;
    std::stringstream ss_id;
    ss_id << inp_cell->id();
    cell_id = ss_id.str();
  }
  Cell_Class(const Cell_Class &inp_cell) = delete;
  Cell_Class(Cell_Class &&inp_cell) noexcept
   : matrices_calculated(false),
     n_faces(inp_cell.n_faces),
     id_num(inp_cell.id_num),
     cell_id(inp_cell.cell_id),
     half_range_flag(inp_cell.half_range_flag),
     face_owner_rank(inp_cell.face_owner_rank),
     dealii_Cell(std::move(inp_cell.dealii_Cell)),
     Face_ID_in_this_rank(inp_cell.Face_ID_in_this_rank),
     Face_ID_in_all_ranks(inp_cell.Face_ID_in_all_ranks),
     BCs(inp_cell.BCs)
  {
  }
  ~Cell_Class()
  {
    pCell_FEValues = nullptr;
    pFace_FEValues = nullptr;
  }
  void attach_FEValues(dealii::FEValues<dim> &FEValues_inp,
                       dealii::FEFaceValues<dim> &FEFaceValues_inp)
  {
    pCell_FEValues = &FEValues_inp;
    pFace_FEValues = &FEFaceValues_inp;
  }
  void reinit_Cell_FEValues()
  {
    assert(pCell_FEValues != nullptr);
    pCell_FEValues->reinit(dealii_Cell);
  }
  void reinit_Face_FEValues(unsigned i_face)
  {
    assert(pFace_FEValues != nullptr);
    pFace_FEValues->reinit(dealii_Cell, i_face);
  }
  template <typename T>
  void assign_matrices(T &&A_, T &&B_, T &&C_, T &&D_, T &&E_, T &&H_, T &&H2_, T &&M_)
  {
    A = std::move(A_);
    B = std::move(B_);
    C = std::move(C_);
    D = std::move(D_);
    E = std::move(E_);
    H = std::move(H_);
    H2 = std::move(H2_);
    M = std::move(M_);
    matrices_calculated = true;
  }
  template <typename T>
  void get_matrices(T &A_, T &B_, T &C_, T &D_, T &E_, T &H_, T &H2_, T &M_)
  {
    A_ = std::move(A);
    B_ = std::move(B);
    C_ = std::move(C);
    D_ = std::move(D);
    E_ = std::move(E);
    H_ = std::move(H);
    H2_ = std::move(H2);
    M_ = std::move(M);
    matrices_calculated = false;
  }

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
  dealii::FEValues<dim> *pCell_FEValues;
  dealii::FEFaceValues<dim> *pFace_FEValues;
};

template <int dim, int spacedim = dim>
struct Face_Class
{
  Face_Class()
    : n_local_connected_faces(0),
      n_nonlocal_connected_faces(0),
      n_local_connected_DOFs(0),
      n_nonlocal_connected_DOFs(0),
      num_global_DOFs(0),
      owner_rank_id(-1)
  {
  }
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
