#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <Eigen/Dense>

#include "jacobi_polynomial.hpp"

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

/* BasisFuncs is the structure containing the basis functions, their gradients,
 * and their divergence. The main motivation behind this is to avoid the
 * repeated calculation of bases on a unit cell for every element. This
 * structure has a constructor which takes quadrature points as inputs and
 * stores the corresponding basis.
 */
template <int dim>
struct BasisFuncs
{
  unsigned poly_order;
  std::vector<std::vector<double>> bases;
  std::vector<std::vector<dealii::Tensor<1, dim>>> bases_grads;

  /*
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
  BasisFuncs(const std::vector<dealii::Point<dim>> &Points_inp, const unsigned &n_poly_bases)
    : poly_order(n_poly_bases)
  {
    JacobiP<dim> Jacobi_P(poly_order, 0, 0, JacobiP<dim>::From_0_to_1);
    for (unsigned i1 = 0; i1 < Points_inp.size(); ++i1)
    {
      dealii::Point<dim> p0 = Points_inp[i1];
      std::vector<double> Ni = Jacobi_P.value(p0);
      std::vector<dealii::Tensor<1, dim>> Ni_grad = Jacobi_P.grad(p0);
      bases.push_back(Ni);
      bases_grads.push_back(Ni_grad);
    }
  }

  /**
   * This function projects the function "func" to the current basis.
   * Since, this is a regular integration, we do not need the JxW. We
   * need only the weights.
   */
  template <int func_dim, typename T>
  void Project_to_Basis(const Function<func_dim, T> &func,
                        const std::vector<dealii::Point<func_dim>> &points,
                        const std::vector<double> &weights,
                        Eigen::MatrixXd &vec)
  {
    assert(bases.size() == points.size());
    assert(points.size() == weights.size());
    unsigned n_polys = bases[0].size();
    vec = Eigen::MatrixXd::Zero(n_polys, 1);
    for (unsigned i1 = 0; i1 < weights.size(); ++i1)
    {
      Eigen::MatrixXd Nj(n_polys, 1);
      Nj = Eigen::VectorXd::Map(bases[i1].data(), n_polys);
      vec += weights[i1] * func.value(points[i1], points[i1]) * Nj;
    }
  }

  template <int func_dim, typename T>
  void Project_to_Basis(const Function<func_dim, T> &func,
                        const std::vector<dealii::Point<func_dim>> &points,
                        const std::vector<dealii::Point<func_dim>> &normals,
                        const std::vector<double> &weights,
                        Eigen::MatrixXd &vec)
  {
    assert(bases.size() == points.size());
    assert(points.size() == weights.size());
    unsigned n_polys = bases[0].size();
    vec = Eigen::MatrixXd::Zero(n_polys, 1);
    for (unsigned i1 = 0; i1 < weights.size(); ++i1)
    {
      Eigen::MatrixXd Nj(n_polys, 1);
      Nj = Eigen::VectorXd::Map(bases[i1].data(), n_polys);
      vec += weights[i1] * func.value(points[i1], normals[i1]) * Nj;
    }
  }
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
    //    pFace_FEValues = nullptr;
    std::stringstream ss_id;
    ss_id << inp_cell->id();
    cell_id = ss_id.str();
  }
  Cell_Class(const Cell_Class &inp_cell) = delete;
  Cell_Class(Cell_Class &&inp_cell) noexcept
    : n_faces(inp_cell.n_faces),
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

#endif // SUPPORT_CLASSES
