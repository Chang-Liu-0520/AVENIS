#include "support_classes.hpp"

template <int dim, typename T, int spacedim>
Function<dim, T, spacedim>::Function()
{
}

template <int dim, typename T, int spacedim>
Function<dim, T, spacedim>::~Function()
{
}

template <int dim, int spacedim>
Cell_Class<dim, spacedim>::Cell_Class(const dealii_Cell_Type &inp_cell, unsigned id_num_)
  : n_faces(dealii::GeometryInfo<dim>::faces_per_cell),
    id_num(id_num_),
    half_range_flag(n_faces, 0),
    face_owner_rank(n_faces, -1),
    dealii_Cell(inp_cell),
    Face_ID_in_this_rank(n_faces, -2),
    Face_ID_in_all_ranks(n_faces, -2),
    BCs(n_faces)
{
  //  cell_quad_fe_vals = nullptr;
  //  face_quad_fe_vals = nullptr;
  std::stringstream ss_id;
  ss_id << inp_cell->id();
  cell_id = ss_id.str();
}

template <int dim, int spacedim>
Cell_Class<dim, spacedim>::Cell_Class(Cell_Class &&inp_cell) noexcept
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

template <int dim, int spacedim>
Cell_Class<dim, spacedim>::~Cell_Class()
{
}

template <int dim, int spacedim>
void Cell_Class<dim, spacedim>::attach_FEValues(
 std::unique_ptr<dealii::FEValues<dim>> &cell_quad_fe_vals_,
 std::unique_ptr<dealii::FEFaceValues<dim>> &face_quad_fe_vals_,
 std::unique_ptr<dealii::FEValues<dim>> &cell_supp_fe_vals_,
 std::unique_ptr<dealii::FEFaceValues<dim>> &face_supp_fe_vals_)
{

  cell_quad_fe_vals = std::move(cell_quad_fe_vals_);
  face_quad_fe_vals = std::move(face_quad_fe_vals_);
  cell_supp_fe_vals = std::move(cell_supp_fe_vals_);
  face_supp_fe_vals = std::move(face_supp_fe_vals_);
}

template <int dim, int spacedim>
void Cell_Class<dim, spacedim>::detach_FEValues(
 std::unique_ptr<dealii::FEValues<dim>> &cell_quad_fe_vals_,
 std::unique_ptr<dealii::FEFaceValues<dim>> &face_quad_fe_vals_,
 std::unique_ptr<dealii::FEValues<dim>> &cell_supp_fe_vals_,
 std::unique_ptr<dealii::FEFaceValues<dim>> &face_supp_fe_vals_)
{

  cell_quad_fe_vals_ = std::move(cell_quad_fe_vals);
  face_quad_fe_vals_ = std::move(face_quad_fe_vals);
  cell_supp_fe_vals_ = std::move(cell_supp_fe_vals);
  face_supp_fe_vals_ = std::move(face_supp_fe_vals);
}

template <int dim, int spacedim>
void Cell_Class<dim, spacedim>::reinit_Cell_FEValues()
{
  cell_quad_fe_vals->reinit(dealii_Cell);
  cell_supp_fe_vals->reinit(dealii_Cell);
}

template <int dim, int spacedim>
void Cell_Class<dim, spacedim>::reinit_Face_FEValues(unsigned i_face)
{
  face_quad_fe_vals->reinit(dealii_Cell, i_face);
  face_supp_fe_vals->reinit(dealii_Cell, i_face);
}

template <int dim, int spacedim>
template <typename T>
void Cell_Class<dim, spacedim>::assign_matrices(
 T &&A_, T &&B_, T &&C_, T &&D_, T &&E_, T &&H_, T &&H2_, T &&M_)
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

template <int dim, int spacedim>
template <typename T>
void Cell_Class<dim, spacedim>::get_matrices(
 T &A_, T &B_, T &C_, T &D_, T &E_, T &H_, T &H2_, T &M_)
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

template <int dim, int spacedim>
Face_Class<dim, spacedim>::Face_Class()
  : n_local_connected_faces(0),
    n_nonlocal_connected_faces(0),
    n_local_connected_DOFs(0),
    n_nonlocal_connected_DOFs(0),
    num_global_DOFs(0),
    owner_rank_id(-1)
{
}
