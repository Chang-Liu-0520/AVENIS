#include "diffusion.hpp"

template <int dim>
Diffusion<dim>::Diffusion(const unsigned &order,
                          const MPI_Comm &comm_,
                          const unsigned &comm_size_,
                          const unsigned &comm_rank_,
                          const unsigned &n_threads,
                          const bool &Adaptive_ON_)
  : comm(comm_),
    comm_size(comm_size_),
    comm_rank(comm_rank_),
    poly_order(order),
    quad_order((order * 2 + 6) / 2),
    n_internal_unknowns(pow(poly_order + 1, dim)),
    n_trace_unknowns((poly_order + 1) * n_faces_per_cell),
    Grid1(comm,
          typename dealii::Triangulation<dim>::MeshSmoothing(
           dealii::Triangulation<dim>::smoothing_on_refinement |
           dealii::Triangulation<dim>::smoothing_on_coarsening)),
    Elem_Mapping(),
    elem_integration_capsul(quad_order),
    face_integration_capsul(quad_order),
    LGL_quad_1D(poly_order + 1),
    support_points_1D(LGL_quad_1D.get_points()),
    DG_Elem1(poly_order),
    DG_System1(DG_Elem1, 1 + dim),
    DoF_H_Refine(Grid1),
    DoF_H1_System(Grid1),
    the_elem_basis(elem_integration_capsul.get_points(),
                   LGL_quad_1D.get_points(),
                   Domain::From_0_to_1),
    the_face_basis(face_integration_capsul.get_points(),
                   LGL_quad_1D.get_points(),
                   Domain::From_0_to_1),
    refn_cycle(0),
    Adaptive_ON(Adaptive_ON_),
    n_threads(n_threads)
{
  if (comm_rank == 0)
  {
    Convergence_Result.open("Convergence_Result.txt",
                            std::ofstream::out | std::fstream::app);
    Execution_Time.open("Execution_Time.txt", std::ofstream::out | std::fstream::app);
  }
  std::vector<unsigned> repeats(dim, 1);
  dealii::Point<dim> point_1, point_2;
  for (int i_dim = 0; i_dim < dim; ++i_dim)
  {
    point_1[i_dim] = -1.0;
    point_2[i_dim] = 1.0;
  }
  dealii::GridGenerator::subdivided_hyper_rectangle(Grid1, repeats, point_1, point_2);

  //  Set_Boundary_Indicator(Grid1);
  Set_Boundary_Indicator();

  //  dealii::GridTools::rotate(asin(1.0) / 3.0 * 1.0, Grid1);
}

template <int dim>
Diffusion<dim>::~Diffusion()
{
  DoF_H1_System.clear();
  DoF_H_Refine.clear();
  if (comm_rank == 0)
  {
    Convergence_Result.close();
    Execution_Time.close();
  }
}

template <int dim>
void Diffusion<dim>::Compute_Error(const Function<dim, double> &func,
                                   const std::vector<dealii::Point<dim>> &points_loc,
                                   const std::vector<double> &JxWs,
                                   const Eigen::MatrixXd &modal_vector,
                                   const Eigen::MatrixXd &mode_to_Qpoint_matrix,
                                   double &error)
{
  error = 0;
  assert(points_loc.size() == JxWs.size());
  assert(modal_vector.rows() == mode_to_Qpoint_matrix.cols());
  assert((long int)points_loc.size() == mode_to_Qpoint_matrix.rows());
  Eigen::MatrixXd values_at_Nodes = mode_to_Qpoint_matrix * modal_vector;
  for (unsigned i_point = 0; i_point < JxWs.size(); ++i_point)
  {
    error += (func.value(points_loc[i_point], points_loc[i_point]) -
              values_at_Nodes(i_point, 0)) *
             (func.value(points_loc[i_point], points_loc[i_point]) -
              values_at_Nodes(i_point, 0)) *
             JxWs[i_point];
  }
}

template <int dim>
void Diffusion<dim>::Compute_Error(const Function<dim, dealii::Tensor<1, dim>> &func,
                                   const std::vector<dealii::Point<dim>> &points_loc,
                                   const std::vector<double> &JxWs,
                                   const Eigen::MatrixXd &modal_vector,
                                   const Eigen::MatrixXd &mode_to_Qpoint_matrix,
                                   double &error)
{
  error = 0;
  unsigned n_unknowns = mode_to_Qpoint_matrix.cols();
  assert(points_loc.size() == JxWs.size());
  assert(modal_vector.rows() == dim * n_unknowns);
  assert((long int)points_loc.size() == mode_to_Qpoint_matrix.rows());
  std::vector<Eigen::MatrixXd> values_at_Nodes(3);
  for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
    values_at_Nodes[i_dim] =
     mode_to_Qpoint_matrix * modal_vector.block(i_dim * n_unknowns, 0, n_unknowns, 1);
  for (unsigned i_point = 0; i_point < JxWs.size(); ++i_point)
  {
    dealii::Tensor<1, dim> temp_val;
    for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
    {
      temp_val[i_dim] = values_at_Nodes[i_dim](i_point, 0);
    }
    error += (func.value(points_loc[i_point], points_loc[i_point]) - temp_val) *
             (func.value(points_loc[i_point], points_loc[i_point]) - temp_val) *
             JxWs[i_point];
  }
}

/**
 * In this function we calculate the matrices used in all other methods.
 * In this calculation we choose to use the nodal or modal basis for the faces
 * and elements.
 */
template <int dim>
void Diffusion<dim>::CalculateMatrices(Cell_Class<dim> &cell,
                                       Poly_Basis<elem_basis_type, dim> &the_elem_basis)
{
  const unsigned n_polys = pow(poly_order + 1, dim);
  const unsigned n_polyfaces = pow(poly_order + 1, dim - 1);
  typedef Eigen::MatrixXd T;

  Poly_Basis<face_basis_type, dim - 1> face_poly_basis(
   face_integration_capsul.get_points(), support_points_1D, Domain::From_0_to_1);
  std::vector<dealii::DerivativeForm<1, dim, dim>> D_Forms =
   cell.pCell_FEValues->get_inverse_jacobians();
  std::vector<dealii::Point<dim>> QPoints_Locs =
   cell.pCell_FEValues->get_quadrature_points();
  std::vector<double> cell_JxW = cell.pCell_FEValues->get_JxW_values();

  T A = T::Zero(dim * n_polys, dim * n_polys);
  T B = T::Zero(dim * n_polys, n_polys);
  T C = T::Zero(dim * n_polys, n_faces_per_cell * n_polyfaces);
  T D = T::Zero(n_polys, n_polys);
  T E = T::Zero(n_polys, n_faces_per_cell * n_polyfaces);
  T H = T::Zero(n_faces_per_cell * n_polyfaces, n_faces_per_cell * n_polyfaces);
  T H2 = T::Zero(n_faces_per_cell * n_polyfaces, n_faces_per_cell * n_polyfaces);
  T M = T::Zero(n_polys, n_polys);

  Eigen::MatrixXd Ni_grad, NjT, Ni_vec;
  for (unsigned i1 = 0; i1 < elem_integration_capsul.size(); ++i1)
  {
    Ni_grad = Eigen::MatrixXd::Zero(dim * n_polys, 1);
    NjT = the_elem_basis.the_bases.block(i1, 0, 1, n_polys);
    Ni_vec = Eigen::MatrixXd::Zero(dim * n_polys, dim);
    for (unsigned i_poly = 0; i_poly < n_polys; ++i_poly)
    {
      dealii::Tensor<2, dim> d_form = D_Forms[i1];
      dealii::Tensor<1, dim> N_grads_X = the_elem_basis.bases_grads[i1][i_poly] * d_form;
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      {
        Ni_vec(n_polys * i_dim + i_poly, i_dim) = NjT(0, i_poly);
        Ni_grad(n_polys * i_dim + i_poly, 0) = N_grads_X[i_dim];
      }
    }
    Eigen::MatrixXd kappa_inv_ = kappa_inv.value(QPoints_Locs[i1], QPoints_Locs[i1]);
    A += cell_JxW[i1] * Ni_vec * kappa_inv_ * Ni_vec.transpose();
    M += cell_JxW[i1] * NjT.transpose() * NjT;
    B += cell_JxW[i1] * Ni_grad * NjT;
  }

  Eigen::MatrixXd normal(dim, 1);
  std::vector<dealii::Point<dim - 1>> Face_Q_Points =
   face_integration_capsul.get_points();
  for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
  {
    cell.reinit_Face_FEValues(i_face);
    Eigen::MatrixXd C_On_Face = Eigen::MatrixXd::Zero(dim * n_polys, n_polyfaces);
    Eigen::MatrixXd E_On_Face = Eigen::MatrixXd::Zero(n_polys, n_polyfaces);
    Eigen::MatrixXd H_On_Face = Eigen::MatrixXd::Zero(n_polyfaces, n_polyfaces);
    Eigen::MatrixXd H2_On_Face = Eigen::MatrixXd::Zero(n_polyfaces, n_polyfaces);
    std::vector<dealii::Point<dim>> Projected_Face_Q_Points(
     face_integration_capsul.size());
    dealii::QProjector<dim>::project_to_face(face_integration_capsul,
                                             i_face,
                                             Projected_Face_Q_Points);
    std::vector<dealii::Point<dim>> Normals =
     cell.pFace_FEValues->get_normal_vectors();
    std::vector<double> Face_JxW = cell.pFace_FEValues->get_JxW_values();
    Eigen::MatrixXd NjT_Face = Eigen::MatrixXd::Zero(1, n_polyfaces);
    Eigen::MatrixXd Nj_vec;
    Eigen::MatrixXd Nj = Eigen::MatrixXd::Zero(n_polys, 1);
    for (unsigned i_Q_face = 0; i_Q_face < face_integration_capsul.size(); ++i_Q_face)
    {
      Nj_vec = Eigen::MatrixXd::Zero(dim * n_polys, dim);
      std::vector<double> N_valus =
       the_elem_basis.value(Projected_Face_Q_Points[i_Q_face]);
      std::vector<double> half_range_face_basis, face_basis;

      face_basis = the_face_basis.bases[i_Q_face];
      half_range_face_basis =
       face_poly_basis.value(Face_Q_Points[i_Q_face], cell.half_range_flag[i_face]);

      for (unsigned i_polyface = 0; i_polyface < n_polyfaces; ++i_polyface)
      {
        if (cell.half_range_flag[i_face] == 0)
          NjT_Face(0, i_polyface) = face_basis[i_polyface];
        else
          NjT_Face(0, i_polyface) = half_range_face_basis[i_polyface];
      }
      for (unsigned i_poly = 0; i_poly < n_polys; ++i_poly)
      {
        Nj(i_poly, 0) = N_valus[i_poly];
        for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
          Nj_vec(i_dim * n_polys + i_poly, i_dim) = N_valus[i_poly];
      }
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        normal(i_dim, 0) = Normals[i_Q_face](i_dim);
      C_On_Face += Face_JxW[i_Q_face] * Nj_vec * normal * NjT_Face;
      D += Face_JxW[i_Q_face] * taus[i_face] * Nj * Nj.transpose();
      E_On_Face += Face_JxW[i_Q_face] * taus[i_face] * Nj * NjT_Face;
      H_On_Face += Face_JxW[i_Q_face] * taus[i_face] * NjT_Face.transpose() * NjT_Face;
      H2_On_Face += Face_JxW[i_Q_face] * NjT_Face.transpose() * NjT_Face;
    }
    H.block(i_face * n_polyfaces, i_face * n_polyfaces, n_polyfaces, n_polyfaces) =
     H_On_Face;
    H2.block(i_face * n_polyfaces, i_face * n_polyfaces, n_polyfaces, n_polyfaces) =
     H2_On_Face;
    C.block(0, i_face * n_polyfaces, dim * n_polys, n_polyfaces) = C_On_Face;
    E.block(0, i_face * n_polyfaces, n_polys, n_polyfaces) = E_On_Face;
  }
  cell.assign_matrices(A, B, C, D, E, H, H2, M);
}

template <int dim>
void Diffusion<dim>::Assemble_Globals()
{
  unsigned n_polys = pow(poly_order + 1, dim);
  unsigned n_polyfaces = pow(poly_order + 1, dim - 1);
  Poly_Basis<elem_basis_type, dim> elem_poly_basis(
   elem_integration_capsul.get_points(), support_points_1D, Domain::From_0_to_1);
  std::vector<double> Q_Weights = elem_integration_capsul.get_weights();
  std::vector<double> Face_Q_Weights = face_integration_capsul.get_weights();
  dealii::QGaussLobatto<dim> LGL_elem_support_points(poly_order + 1);
  dealii::QGaussLobatto<dim - 1> LGL_face_support_points(poly_order + 1);
#ifdef _OPENMP
#pragma omp parallel
  {
    unsigned thread_id = omp_get_thread_num();
#else
  unsigned thread_id = 0;
  {
#endif
    dealii::FEValues<dim> FEValues_Elem1(Elem_Mapping,
                                         DG_Elem1,
                                         elem_integration_capsul,
                                         dealii::update_JxW_values |
                                          dealii::update_quadrature_points |
                                          dealii::update_inverse_jacobians |
                                          dealii::update_jacobians);
    dealii::FEFaceValues<dim> FEValues_Face1(Elem_Mapping,
                                             DG_Elem1,
                                             face_integration_capsul,
                                             dealii::update_values |
                                              dealii::update_JxW_values |
                                              dealii::update_quadrature_points |
                                              dealii::update_face_normal_vectors |
                                              dealii::update_inverse_jacobians);

    dealii::FEValues<dim> FEValues_Elem2(Elem_Mapping,
                                         DG_Elem1,
                                         LGL_elem_support_points,
                                         dealii::update_quadrature_points);
    dealii::FEFaceValues<dim> FEValues_Face2(Elem_Mapping,
                                             DG_Elem1,
                                             LGL_face_support_points,
                                             dealii::update_quadrature_points |
                                              dealii::update_face_normal_vectors);
    for (unsigned i_cell = thread_id; i_cell < All_Owned_Cells.size();
         i_cell = i_cell + n_threads)
    {
      Cell_Class<dim> &cell = All_Owned_Cells[i_cell];
      char buffer[100];
      std::snprintf(buffer,
                    100,
                    "I am thread num. %d on rank %d on element: %d from %zu",
                    thread_id,
                    comm_rank,
                    i_cell,
                    All_Owned_Cells.size());

      FEValues_Elem2.reinit(cell.dealii_Cell);
      cell.attach_FEValues(FEValues_Elem1, FEValues_Face1);
      cell.reinit_Cell_FEValues();

      Eigen::MatrixXd A, B, C, D, E, H, H2, M;
      CalculateMatrices(cell, elem_poly_basis);
      cell.get_matrices(A, B, C, D, E, H, H2, M);

      Eigen::MatrixXd Ainv = A.inverse();
      Eigen::MatrixXd BT_Ainv = B.transpose() * Ainv;
      Eigen::LDLT<Eigen::MatrixXd, Eigen::Lower> LDLT_of_A = A.ldlt();
      Eigen::LDLT<Eigen::MatrixXd, Eigen::Lower> LDLT_of_BT_Ainv_B_plus_D =
       (BT_Ainv * B + D).ldlt();

      std::vector<dealii::Point<dim>> Q_Points_Loc =
       cell.pCell_FEValues->get_quadrature_points();

      std::vector<double> cell_mat;
      std::vector<int> row_nums, col_nums;

      Eigen::MatrixXd f_vec = Eigen::MatrixXd::Zero(n_polys, 1);
      for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
      {
        for (unsigned i_polyface = 0; i_polyface < n_polyfaces; ++i_polyface)
        {
          int global_face_number = cell.Face_ID_in_all_ranks[i_face];
          int global_dof_number = -1;
          if (global_face_number >= 0)
            global_dof_number = global_face_number * n_polyfaces + i_polyface;

          if (global_face_number < -1)
            std::cout << global_face_number << std::endl;

          assert(global_face_number >= -1);
          assert(global_dof_number < INT_MAX);

          row_nums.push_back(global_dof_number);
          col_nums.push_back(global_dof_number);

          Eigen::MatrixXd uhat_vec =
           Eigen::MatrixXd::Zero(n_faces_per_cell * n_polyfaces, 1);
          Eigen::MatrixXd gN_vec =
           Eigen::MatrixXd::Zero(n_faces_per_cell * n_polyfaces, 1);
          uhat_vec(i_face * n_polyfaces + i_polyface, 0) = 1.0;
          Eigen::MatrixXd u_vec, q_vec;
          std::vector<double> jth_col;
          u_from_uhat_f(
           LDLT_of_BT_Ainv_B_plus_D, BT_Ainv, C, E, M, uhat_vec, uhat_vec, f_vec, u_vec);
          q_from_u_uhat(LDLT_of_A, B, C, uhat_vec, u_vec, q_vec);
          uhat_u_q_to_jth_col(C, E, H, H2, uhat_vec, u_vec, q_vec, gN_vec, -1, jth_col);
          cell_mat.insert(cell_mat.end(),
                          std::make_move_iterator(jth_col.begin()),
                          std::make_move_iterator(jth_col.end()));
        }
      }

#ifdef _OPENMP
#pragma omp critical
#endif
      {
        MatSetValues(global_mat,
                     row_nums.size(),
                     row_nums.data(),
                     col_nums.size(),
                     col_nums.data(),
                     cell_mat.data(),
                     ADD_VALUES);
      }

      {
        Eigen::MatrixXd gD_vec;
        Eigen::MatrixXd gN_vec =
         Eigen::MatrixXd::Zero(n_polyfaces * n_faces_per_cell, 1);
        Eigen::MatrixXd uhat_vec =
         Eigen::MatrixXd::Zero(n_polyfaces * n_faces_per_cell, 1);
        for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
        {
          FEValues_Face2.reinit(cell.dealii_Cell, i_face);
          if (cell.BCs[i_face] == Cell_Class<dim>::Dirichlet)
          {
            cell.reinit_Face_FEValues(i_face);
            std::vector<dealii::Point<dim>> FaceQ_Points_Loc =
             cell.pFace_FEValues->get_quadrature_points();
            std::vector<dealii::Point<dim>> face_supp_points_loc =
             FEValues_Face2.get_quadrature_points();
            if (cell.half_range_flag[i_face] == 0)
            {
              the_face_basis.Project_to_Basis(Dirichlet_BC_func,
                                              FaceQ_Points_Loc,
                                              face_supp_points_loc,
                                              Face_Q_Weights,
                                              gD_vec);
            }
            else
              std::cout << "There is something wrong dude!\n";
            uhat_vec.block(i_face * n_polyfaces, 0, n_polyfaces, 1) = gD_vec;
          }
          if (cell.BCs[i_face] == Cell_Class<dim>::Neumann)
          {
            cell.reinit_Face_FEValues(i_face);
            Eigen::MatrixXd gN_vec_face;
            std::vector<dealii::Point<dim>> FaceQ_Points_Loc =
             cell.pFace_FEValues->get_quadrature_points();
            std::vector<dealii::Point<dim>> face_supp_points_loc =
             FEValues_Face2.get_quadrature_points();
            std::vector<dealii::Point<dim>> face_normals_at_support =
             FEValues_Face2.get_normal_vectors();
            std::vector<dealii::Point<dim>> Normal_Vec_Dir =
             cell.pFace_FEValues->get_normal_vectors();
            if (cell.half_range_flag[i_face] == 0)
              the_face_basis.Project_to_Basis(Neumann_BC_func,
                                              FaceQ_Points_Loc,
                                              face_supp_points_loc,
                                              Normal_Vec_Dir,
                                              face_normals_at_support,
                                              Face_Q_Weights,
                                              gN_vec_face);
            gN_vec.block(i_face * n_polyfaces, 0, n_polyfaces, 1) = gN_vec_face;
          }
        }

        std::vector<dealii::Point<dim>> elem_supp_points_loc =
         FEValues_Elem2.get_quadrature_points();
        the_elem_basis.Project_to_Basis(
         f_func, Q_Points_Loc, elem_supp_points_loc, Q_Weights, f_vec);
        std::vector<double> rhs_col;
        Eigen::MatrixXd u_vec, q_vec;
        u_from_uhat_f(
         LDLT_of_BT_Ainv_B_plus_D, BT_Ainv, C, E, M, uhat_vec, uhat_vec, f_vec, u_vec);
        q_from_u_uhat(LDLT_of_A, B, C, uhat_vec, u_vec, q_vec);
        uhat_u_q_to_jth_col(C, E, H, H2, uhat_vec, u_vec, q_vec, gN_vec, 1, rhs_col);
#ifdef _OPENMP
#pragma omp critical
#endif
        {
          VecSetValues(RHS_vec, row_nums.size(), row_nums.data(), rhs_col.data(), ADD_VALUES);
        }
      }

      {
        std::vector<double> exact_uhat_vec;
        Eigen::MatrixXd face_exact_uhat_vec;
        for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
        {
          cell.reinit_Face_FEValues(i_face);
          std::vector<dealii::Point<dim>> Face_Q_Points_Loc =
           cell.pFace_FEValues->get_quadrature_points();
          std::vector<dealii::Point<dim>> face_supp_points_loc =
           FEValues_Face2.get_quadrature_points();
          the_face_basis.Project_to_Basis(u_func,
                                          Face_Q_Points_Loc,
                                          face_supp_points_loc,
                                          face_integration_capsul.get_weights(),
                                          face_exact_uhat_vec);
          exact_uhat_vec.insert(exact_uhat_vec.end(),
                                face_exact_uhat_vec.data(),
                                face_exact_uhat_vec.data() +
                                 face_exact_uhat_vec.rows());
#ifdef _OPENMP
#pragma omp critical
#endif
          {
            VecSetValues(exact_solution,
                         row_nums.size(),
                         row_nums.data(),
                         exact_uhat_vec.data(),
                         INSERT_VALUES);
          }
        }
      }
    }
  }
}

template <int dim>
template <typename T, typename U>
void Diffusion<dim>::q_from_u_uhat(
 const U &LDLT_of_A, const T &B, const T &C, const T &uhat, const T &u, T &q)
{
  q = B * u - C * uhat;
  q = LDLT_of_A.solve(q);
}

template <int dim>
template <typename T, typename U>
void Diffusion<dim>::u_from_uhat_f(const U &LDLT_of_BT_Ainv_B_plus_D,
                                   const T &BT_Ainv,
                                   const T &C,
                                   const T &E,
                                   const T &M,
                                   const T &uhat,
                                   const T &lambda,
                                   const T &f,
                                   T &u)
{
  u = LDLT_of_BT_Ainv_B_plus_D.solve(M * f + BT_Ainv * C * uhat + E * lambda);
}

template <int dim>
template <typename T>
void Diffusion<dim>::uhat_u_q_to_jth_col(const T &C,
                                         const T &E,
                                         const T &H,
                                         const T &H2,
                                         const T &uhat,
                                         const T &u,
                                         const T &q,
                                         const T &g_N,
                                         const double &multiplier,
                                         std::vector<double> &jth_col)
{
  T jth_col_vec =
   multiplier * (C.transpose() * q + E.transpose() * u - H * uhat) - H2 * g_N;
  jth_col.assign(jth_col_vec.data(), jth_col_vec.data() + jth_col_vec.rows());
}

template <int dim>
void Diffusion<dim>::Calculate_Internal_Unknowns(double *const &local_uhat_vec)
{
  dealii::IndexSet refn_ghost_indices, elem_ghost_indices;
  dealii::IndexSet refn_owned_indices = DoF_H_Refine.locally_owned_dofs();
  dealii::IndexSet elem_owned_indices = DoF_H1_System.locally_owned_dofs();
  dealii::DoFTools::extract_locally_relevant_dofs(DoF_H_Refine, refn_ghost_indices);
  dealii::DoFTools::extract_locally_relevant_dofs(DoF_H1_System, elem_ghost_indices);
  LA::MPI::Vector refn_sol_temp(refn_owned_indices, comm);
  LA::MPI::Vector elem_sol_temp(elem_owned_indices, comm);
  std::vector<unsigned> refn_owned_indices_vec;
  std::vector<unsigned> elem_owned_indices_vec;
  refn_owned_indices.fill_index_vector(refn_owned_indices_vec);
  elem_owned_indices.fill_index_vector(elem_owned_indices_vec);
  std::vector<PetscScalar> refn_owned_values(refn_owned_indices_vec.size());
  std::vector<PetscScalar> elem_owned_values(elem_owned_indices_vec.size());
  refn_solu.reinit(refn_owned_indices, refn_ghost_indices, comm);
  elem_solu.reinit(elem_owned_indices, elem_ghost_indices, comm);

  unsigned n_polys = pow(poly_order + 1, dim);
  unsigned n_polyfaces = pow(poly_order + 1, dim - 1);
  Poly_Basis<elem_basis_type, dim> the_elem_basis(
   elem_integration_capsul.get_points(), support_points_1D, Domain::From_0_to_1);

  double Error_u = 0;
  double Error_q = 0;
  double Error_ustar = 0;
  double Error_div_q = 0;
  double Error_div_qstar = 0;

  std::vector<double> Q_Weights = elem_integration_capsul.get_weights();
  std::vector<double> Face_Q_Weights = face_integration_capsul.get_weights();

  dealii::QGaussLobatto<1> postproc_support_points(poly_order + 2);
  Poly_Basis<elem_basis_type, dim> the_postproc_elem_basis(
   elem_integration_capsul.get_points(),
   postproc_support_points.get_points(),
   Domain::From_0_to_1);
  Eigen::MatrixXd PostProcess_Mode_to_QPoint_Matrix = the_postproc_elem_basis.the_bases;

  dealii::QGaussLobatto<dim> LGL_elem_support_points(poly_order + 1);
  dealii::QGaussLobatto<dim - 1> LGL_face_support_points(poly_order + 1);

  Eigen::MatrixXd Mode_to_QPoint_Matrix = the_elem_basis.the_bases;

  dealii::FE_DGQ<1> DG_Elem_1D(poly_order);
  Poly_Basis<elem_basis_type, dim> the_elem_qual_dist_basis(
   DG_Elem1.get_unit_support_points(), LGL_quad_1D.get_points(), Domain::From_0_to_1);
  Eigen::MatrixXd Mode_to_Node_Matrix = the_elem_qual_dist_basis.the_bases;

#ifdef _OPENMP
#pragma omp parallel
  {
    unsigned thread_id = omp_get_thread_num();
#else
  unsigned thread_id = 0;
  {
#endif
    dealii::FEValues<dim> FEValues_Elem1(Elem_Mapping,
                                         DG_Elem1,
                                         elem_integration_capsul,
                                         dealii::update_values | dealii::update_gradients |
                                          dealii::update_JxW_values |
                                          dealii::update_quadrature_points |
                                          dealii::update_inverse_jacobians |
                                          dealii::update_jacobians);
    dealii::FEFaceValues<dim> FEValues_Face1(
     Elem_Mapping,
     DG_Elem1,
     face_integration_capsul,
     dealii::update_values | dealii::update_gradients |
      dealii::update_JxW_values | dealii::update_quadrature_points |
      dealii::update_face_normal_vectors | dealii::update_inverse_jacobians);

    dealii::FEValues<dim> FEValues_Elem2(Elem_Mapping,
                                         DG_Elem1,
                                         LGL_elem_support_points,
                                         dealii::update_quadrature_points);
    dealii::FEFaceValues<dim> FEValues_Face2(Elem_Mapping,
                                             DG_Elem1,
                                             LGL_face_support_points,
                                             dealii::update_quadrature_points);

    for (unsigned i_cell = thread_id; i_cell < All_Owned_Cells.size();
         i_cell = i_cell + n_threads)
    {
      Cell_Class<dim> &cell = All_Owned_Cells[i_cell];
      FEValues_Elem2.reinit(cell.dealii_Cell);
      std::vector<dealii::Point<dim>> elem_supp_points_loc =
       FEValues_Elem2.get_quadrature_points();

      char buffer[100];
      std::snprintf(buffer,
                    100,
                    "I am thread num. %d on element: %d from %zu",
                    thread_id,
                    i_cell,
                    All_Owned_Cells.size());
      //      std::cout << buffer << std::endl;

      cell.attach_FEValues(FEValues_Elem1, FEValues_Face1);
      cell.reinit_Cell_FEValues();

      Eigen::MatrixXd A, B, C, D, E, H, H2, M;
      CalculateMatrices(cell, the_elem_basis);
      cell.get_matrices(A, B, C, D, E, H, H2, M);

      Eigen::MatrixXd BT_Ainv = B.transpose() * A.inverse();
      Eigen::LDLT<Eigen::MatrixXd, Eigen::Lower> LDLT_of_A = A.ldlt();
      Eigen::LDLT<Eigen::MatrixXd, Eigen::Lower> LDLT_of_BT_Ainv_B_plus_D =
       (BT_Ainv * B + D).ldlt();

      std::vector<dealii::Point<dim>> Q_Points_Loc =
       cell.pCell_FEValues->get_quadrature_points();
      Eigen::MatrixXd exact_f_vec;
      the_elem_basis.Project_to_Basis(
       f_func, Q_Points_Loc, elem_supp_points_loc, Q_Weights, exact_f_vec);

      Eigen::MatrixXd solved_uhat_vec =
       Eigen::MatrixXd::Zero(n_polyfaces * n_faces_per_cell, 1);
      Eigen::MatrixXd solved_lambda_vec =
       Eigen::MatrixXd::Zero(n_polyfaces * n_faces_per_cell, 1);
      for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
      {
        int global_face_number = cell.Face_ID_in_this_rank[i_face];
        if (global_face_number < 0)
        {
          Eigen::MatrixXd face_uhat_vec;
          cell.reinit_Face_FEValues(i_face);
          FEValues_Face2.reinit(cell.dealii_Cell, i_face);
          std::vector<dealii::Point<dim>> Face_Q_Points_Loc =
           cell.pFace_FEValues->get_quadrature_points();
          std::vector<dealii::Point<dim>> face_supp_points_loc =
           FEValues_Face2.get_quadrature_points();
          the_face_basis.Project_to_Basis(Dirichlet_BC_func,
                                          Face_Q_Points_Loc,
                                          face_supp_points_loc,
                                          Face_Q_Weights,
                                          face_uhat_vec);
          solved_uhat_vec.block(i_face * n_polyfaces, 0, n_polyfaces, 1) = face_uhat_vec;
          solved_lambda_vec.block(i_face * n_polyfaces, 0, n_polyfaces, 1) =
           face_uhat_vec;
        }
        else
        {
          for (unsigned i_polyface = 0; i_polyface < n_polyfaces; ++i_polyface)
          {
            int global_dof_number = global_face_number * n_polyfaces + i_polyface;
            solved_uhat_vec(i_face * n_polyfaces + i_polyface, 0) =
             local_uhat_vec[global_dof_number];
          }
        }
      }

      Eigen::MatrixXd solved_q_vec, solved_u_vec;
      u_from_uhat_f(LDLT_of_BT_Ainv_B_plus_D,
                    BT_Ainv,
                    C,
                    E,
                    M,
                    solved_uhat_vec,
                    solved_uhat_vec,
                    exact_f_vec,
                    solved_u_vec);
      q_from_u_uhat(LDLT_of_A, B, C, solved_uhat_vec, solved_u_vec, solved_q_vec);

      Internal_Vars_Errors(
       cell, solved_u_vec, solved_q_vec, Mode_to_QPoint_Matrix, Error_u, Error_q, Error_div_q);

      Eigen::MatrixXd ustar;
      PostProcess(cell,
                  the_postproc_elem_basis,
                  solved_u_vec,
                  solved_q_vec,
                  ustar,
                  PostProcess_Mode_to_QPoint_Matrix,
                  Error_ustar);

      Eigen::MatrixXd solved_u_at_nodes = Mode_to_Node_Matrix * solved_u_vec;
      Eigen::MatrixXd solved_q_at_nodes(dim * n_polys, 1);
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      {
        solved_q_at_nodes.block(i_dim * n_polys, 0, n_polys, 1) =
         Mode_to_Node_Matrix * solved_q_vec.block(i_dim * n_polys, 0, n_polys, 1);
      }
      unsigned n_local_unknown = solved_u_at_nodes.rows();

      for (unsigned i_local_unknown = 0; i_local_unknown < n_local_unknown; ++i_local_unknown)
      {
        double temp_val = 0;
        for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        {
          temp_val +=
           std::pow(solved_q_at_nodes(i_dim * n_local_unknown + i_local_unknown, 0), 2);
        }
        refn_owned_values[i_cell * n_local_unknown + i_local_unknown] = sqrt(temp_val);
      }

      for (unsigned i_local_unknown = 0; i_local_unknown < n_local_unknown; ++i_local_unknown)
      {
        elem_owned_values[(i_cell * n_local_unknown) * (dim + 1) + i_local_unknown] =
         solved_u_at_nodes(i_local_unknown, 0);
        for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        {
          elem_owned_values[(i_cell * n_local_unknown) * (dim + 1) +
                            (i_dim + 1) * n_local_unknown + i_local_unknown] =
           solved_q_at_nodes(i_dim * n_local_unknown + i_local_unknown, 0);
        }
      }
    }
  }

  refn_sol_temp.set(refn_owned_indices_vec, refn_owned_values);
  refn_sol_temp.compress(dealii::VectorOperation::insert);
  elem_sol_temp.set(elem_owned_indices_vec, elem_owned_values);
  elem_sol_temp.compress(dealii::VectorOperation::insert);
  refn_solu = refn_sol_temp;
  elem_solu = elem_sol_temp;

  double global_Error_u, global_Error_q, global_Error_ustar;
  MPI_Reduce(&Error_u, &global_Error_u, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&Error_q, &global_Error_q, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&Error_ustar, &global_Error_ustar, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

  if (comm_rank == 0)
  {
    char buffer[200];
    std::snprintf(buffer,
                  200,
                  " NEl : %10d, || uh - u ||_L2 : %12.4e; || q - qh ||_L2 "
                  "is: "
                  "%12.4e; || "
                  "div(q - qh) ||_L2 : %12.4e; || u - uh* ||_L2 : %12.4e; || "
                  "div "
                  "(q - qh*) || : %12.4e",
                  Grid1.n_global_active_cells(),
                  sqrt(Error_u),
                  sqrt(Error_q),
                  sqrt(Error_div_q),
                  sqrt(Error_ustar),
                  sqrt(Error_div_qstar));
    //    Convergence_Result << buffer << std::endl;
    std::snprintf(buffer,
                  200,
                  " NEl : %10d, || uh - u ||_L2 : %12.4e; || q - qh ||_L2 "
                  "is: "
                  "%12.4e; || "
                  "div(q - qh) ||_L2 : %12.4e; || u - uh* ||_L2 : %12.4e; || "
                  "div "
                  "(q - qh*) || : %12.4e",
                  Grid1.n_global_active_cells(),
                  sqrt(global_Error_u),
                  sqrt(global_Error_q),
                  sqrt(Error_div_q),
                  sqrt(global_Error_ustar),
                  sqrt(Error_div_qstar));
    Convergence_Result << buffer << std::endl;
  }
}

template <int dim>
template <typename T1, typename T2>
void Diffusion<dim>::Internal_Vars_Errors(const Cell_Class<dim> &cell,
                                          const T1 &solved_u_vec,
                                          const T1 &solved_q_vec,
                                          const T2 &Mode_to_Qpoints_Matrix,
                                          double &Error_u,
                                          double &Error_q,
                                          double &Error_div_q)
{
  unsigned n_polys = pow(poly_order + 1, dim);
  std::vector<dealii::DerivativeForm<1, dim, dim>> D_Forms =
   cell.pCell_FEValues->get_inverse_jacobians();

  std::vector<dealii::Point<dim>> Q_Points_Loc =
   cell.pCell_FEValues->get_quadrature_points();
  std::vector<double> Q_JxWs = cell.pCell_FEValues->get_JxW_values();

  double Error_u2, Error_q2;
  Compute_Error(u_func, Q_Points_Loc, Q_JxWs, solved_u_vec, Mode_to_Qpoints_Matrix, Error_u2);
  for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
  {
    Compute_Error(q_func, Q_Points_Loc, Q_JxWs, solved_q_vec, Mode_to_Qpoints_Matrix, Error_q2);
  }

  Error_u += Error_u2;
  for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
    Error_q += Error_q2;

  for (unsigned i_Qpoint = 0; i_Qpoint < Q_Points_Loc.size(); ++i_Qpoint)
  {
    dealii::Tensor<2, dim> d_form = D_Forms[i_Qpoint];
    double divq_at_i_Qpoint = 0;
    for (unsigned i_poly = 0; i_poly < n_polys; ++i_poly)
    {
      dealii::Tensor<1, dim> grad_X =
       the_elem_basis.bases_grads[i_Qpoint][i_poly] * d_form;
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      {
        divq_at_i_Qpoint += grad_X[i_dim] * solved_q_vec(i_dim * n_polys + i_poly, 0);
      }
    }
    Error_div_q += (divq_at_i_Qpoint - divq_func.value(Q_Points_Loc[i_Qpoint],
                                                       Q_Points_Loc[i_Qpoint])) *
                   (divq_at_i_Qpoint - divq_func.value(Q_Points_Loc[i_Qpoint],
                                                       Q_Points_Loc[i_Qpoint])) *
                   Q_JxWs[i_Qpoint];
  }
}

/* This method produces the matrices which will be used for a very basic
 * postprocessing of u to obtain u*.
 *
 * This postprocessing comprised of two equations:
 *
 *                    \kappa (\grad w , \grad u) = (\grad w , q)
 *                                   (1 , u) = (1 , u*)
 *
 * with u,w \in Q(p+1) and q as obtained in the analysis.
 * It will be formulated as:
 *
 *                             DM* . u* = DB2 . q
 *
 * But it should be noted that the first rows of M* and B2 are zeros.
 * Hence we need another equation to make the system of equation complete.
 */

template <int dim>
template <typename T>
void Diffusion<dim>::Calculate_Postprocess_Matrices(
 Cell_Class<dim> &cell,
 const Poly_Basis<elem_basis_type, dim> &PostProcess_Elem_Basis,
 T &DM_star,
 T &DB2)
{
  const unsigned n_polys = pow(poly_order + 1, dim);
  const unsigned n_polys_plus1 = pow(poly_order + 2, dim);

  std::vector<dealii::DerivativeForm<1, dim, dim>> D_Forms =
   cell.pCell_FEValues->get_inverse_jacobians();
  std::vector<double> cell_JxW = cell.pCell_FEValues->get_JxW_values();
  std::vector<dealii::Point<dim>> QPoints_Locs =
   cell.pCell_FEValues->get_quadrature_points();

  DM_star = T::Zero(n_polys_plus1, n_polys_plus1);
  DB2 = T::Zero(n_polys_plus1, dim * n_polys);

  Eigen::MatrixXd grad_Ni, Ni_vec;
  for (unsigned i_point = 0; i_point < elem_integration_capsul.size(); ++i_point)
  {
    dealii::Tensor<2, dim> d_form = D_Forms[i_point];
    grad_Ni = Eigen::MatrixXd::Zero(n_polys_plus1, dim);
    Ni_vec = Eigen::MatrixXd::Zero(dim * n_polys, dim);
    for (unsigned i_poly = 0; i_poly < n_polys_plus1; ++i_poly)
    {
      dealii::Tensor<1, dim> grad_Ni_at_Qpoint =
       PostProcess_Elem_Basis.bases_grads[i_point][i_poly] * d_form;
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
      {
        grad_Ni(i_poly, i_dim) = grad_Ni_at_Qpoint[i_dim];
      }
    }
    for (unsigned i_poly = 0; i_poly < n_polys; ++i_poly)
    {
      for (unsigned i_dim = 0; i_dim < dim; ++i_dim)
        Ni_vec(i_dim * n_polys + i_poly, i_dim) =
         the_elem_basis.bases[i_point][i_poly];
    }
    DM_star += cell_JxW[i_point] * grad_Ni * grad_Ni.transpose();
    Eigen::MatrixXd kappa_inv_ =
     kappa_inv.value(QPoints_Locs[i_point], QPoints_Locs[i_point]);
    DB2 += cell_JxW[i_point] * grad_Ni * kappa_inv_ * Ni_vec.transpose();
  }
}

template <int dim>
template <typename T1>
void Diffusion<dim>::PostProcess(Cell_Class<dim> &cell,
                                 const Poly_Basis<elem_basis_type, dim> &PostProcess_Elem_Basis,
                                 const T1 &u,
                                 const T1 &q,
                                 T1 &ustar,
                                 const T1 &PostProcess_Mode_to_Node,
                                 double &error_ustar)
{
  Eigen::MatrixXd LHS_mat_of_ustar, DB2;
  //  JacobiP<dim> Jacobi_P_plus1(poly_order + 1, 0, 0,
  // JacobiP<dim>::From_0_to_1);

  std::vector<dealii::Point<dim>> Q_Points_Loc =
   cell.pCell_FEValues->get_quadrature_points();
  std::vector<double> Q_JxWs = cell.pCell_FEValues->get_JxW_values();

  Calculate_Postprocess_Matrices(cell, PostProcess_Elem_Basis, LHS_mat_of_ustar, DB2);
  Eigen::MatrixXd RHS_vec_of_ustar = -DB2 * q;
  LHS_mat_of_ustar(0, 0) = 1;
  RHS_vec_of_ustar(0, 0) = u(0);
  ustar = LHS_mat_of_ustar.ldlt().solve(RHS_vec_of_ustar);
  double error_ustar_at_cell;
  Compute_Error(u_func, Q_Points_Loc, Q_JxWs, ustar, PostProcess_Mode_to_Node, error_ustar_at_cell);
  error_ustar += error_ustar_at_cell;
}

template <int dim>
void Diffusion<dim>::vtk_visualizer()
{
  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(DoF_H1_System);

  std::vector<std::string> solution_names(dim + 1);
  solution_names[0] = "head";
  for (unsigned i1 = 0; i1 < dim; ++i1)
    solution_names[i1 + 1] = "flow";
  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(
   1, dealii::DataComponentInterpretation::component_is_scalar);
  for (unsigned i1 = 0; i1 < dim; ++i1)
    data_component_interpretation.push_back(
     dealii::DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector(elem_solu,
                           solution_names,
                           dealii::DataOut<dim>::type_dof_data,
                           data_component_interpretation);

  dealii::Vector<float> subdomain(Grid1.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = comm_rank;
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  const std::string filename =
   ("solution-" + dealii::Utilities::int_to_string(refn_cycle, 2) + "." +
    dealii::Utilities::int_to_string(comm_rank, 4));
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

  if (comm_rank == 0)
  {
    std::vector<std::string> filenames;
    for (unsigned int i = 0; i < comm_size; ++i)
      filenames.push_back("solution-" +
                          dealii::Utilities::int_to_string(refn_cycle, 2) + "." +
                          dealii::Utilities::int_to_string(i, 4) + ".vtu");
    std::ofstream master_output((filename + ".pvtu").c_str());
    data_out.write_pvtu_record(master_output, filenames);
  }
}
