#include "diffusion.hpp"

template <int dim>
void Diffusion<dim>::Set_Boundary_Indicator()
{
  /* I am going to solve:
   *
   *              gN = 0.0
   *          ______________
   *          |            |
   *          |            |
   *  gD = 10 |            | gD = 0.0
   *          |            |
   *          |            |
   *          |  gN = 0.0  |
   *          ______________
   *
   * The following boundary condition loop should be applied on every active
   * face, either ghost or locally owned.
   */
  for (Cell_Type &&cell : Grid1.active_cell_iterators())
  {
    for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
    {
      auto &&face = cell->face(i_face);
      if (face->at_boundary())
      {
        face->set_boundary_id(Neumann_BC_Index);
        //        face->set_boundary_id(Dirichlet_BC_Index);
        const dealii::Point<dim> &face_center = face->center();
        if (face_center(0) < -1.0 + 1.0E-10 || face_center(0) > 1.0 - 1.0E-10)
        {
          face->set_boundary_id(Dirichlet_BC_Index);
        }
      }
    }
  }
}

template <int dim>
void Diffusion<dim>::Refine_Grid(int n)
{
  if (refn_cycle == 0)
  {
    Grid1.refine_global(n);
    refn_cycle += n;
  }
  else if (!Adaptive_ON)
  {
    Grid1.refine_global(1);
    ++refn_cycle;
  }
  else
  {
    dealii::Vector<float> estimated_error_per_cell(Grid1.n_active_cells());
    dealii::KellyErrorEstimator<dim>::estimate(DoF_H_Refine,
                                               dealii::QGauss<dim - 1>(quad_order),
                                               typename dealii::FunctionMap<dim>::type(),
                                               refn_solu,
                                               estimated_error_per_cell);

    dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
     Grid1, estimated_error_per_cell, 0.3, 0.03);
    Grid1.execute_coarsening_and_refinement();
    ++refn_cycle;
  }

  double penalty1 = 8.5;
  taus.assign(n_faces_per_cell, penalty1);
  Set_Boundary_Indicator();

  FreeUpContainers();
  Init_Mesh_Containers();
}

template <int dim>
void Diffusion<dim>::Init_Mesh_Containers()
{
  /* What comes next is just a try to make things more comfortable for myself.
   * I want to store iterators to active cells, to be able to mess up with
   * them later on. This means, I store the rvalue refernce to active cells in
   * a std::vector and use it to modify the mesh. After each refinement, this
   * std::vector will be reconstructed with the new cells ! At the time that I
   * did this, it looked smart !!
   */
  DoF_H_System.distribute_dofs(DG_System);
  DoF_H_Refine.distribute_dofs(DG_Elem);

  All_Owned_Cells.reserve(Grid1.n_locally_owned_active_cells());
  unsigned n_cell = 0;
  n_ghost_cell = 0;
  n_active_cell = 0;
  for (Cell_Type &&cell : DoF_H_System.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      All_Owned_Cells.push_back(std::move(Cell_Class<dim>(cell, n_active_cell)));
      cell_ID_to_num[All_Owned_Cells.back().cell_id] = n_active_cell;
      ++n_active_cell;
    }
    if (cell->is_ghost())
      ++n_ghost_cell;
    ++n_cell;
  }
}

template <int dim>
void Diffusion<dim>::Count_Globals()
{
  unsigned n_polyface = pow(poly_order + 1, dim - 1);
  std::vector<Cell_Class<dim>> All_Ghost_Cells;
  All_Ghost_Cells.reserve(n_ghost_cell);
  std::map<std::string, int> Ghost_ID_to_num;
  unsigned ghost_cell_counter = 0;
  for (Cell_Type &&cell : DoF_H_System.active_cell_iterators())
  {
    if (cell->is_ghost())
    {
      All_Ghost_Cells.push_back(std::move(Cell_Class<dim>(cell, n_active_cell)));
      std::stringstream ss_id;
      ss_id << cell->id();
      std::string str_id = ss_id.str();
      Ghost_ID_to_num[str_id] = ghost_cell_counter;
      ++ghost_cell_counter;
    }
    if (cell->is_locally_owned())
    {
      std::stringstream ss_id;
      ss_id << cell->id();
      std::string str_id = ss_id.str();
      assert(cell_ID_to_num.find(str_id) != cell_ID_to_num.end());
    }
  }

  unsigned local_face_id_on_this_rank = 0;
  unsigned global_face_id_on_this_rank = 0;
  int homogenous_dirichlet = -1;
  unsigned mpi_request_counter = 0;
  unsigned mpi_status_counter = 0;
  std::map<unsigned, bool> is_there_a_msg_from_rank;

  /* Here, we want to count the local and global faces of the mesh. By
   * local, we mean those faces counted in the subdomain of current rank.
   * By global, we mean those faces which are counted as parts of other
   * subdomains. We have two rules for this:
   *
   * rule 1: If a face is common between two subdomains, and one side is
   *         coarser than the other side, the face belongs to the coarser
   *         side; no matter which subdomain has smaller rank.
   *
   * rule 2: If a face is connected to two elements of the same refinement,
   *         and the elements are in two different subdomains, then the face
   *         belongs to the subdomain with smaller rank.
   */

  for (Cell_Class<dim> &cell : All_Owned_Cells)
  {
    for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
    {
      if (cell.Face_ID_in_this_rank[i_face] == -2)
      {
        const auto &face_i1 = cell.dealii_Cell->face(i_face);
        /* The basic case corresponds to face_i1 being on the boundary.
         * In this case we only need to set the number of current face,
         * and we do not bother to know what is going on, on the other
         * side of this face. Because (in the current version), that is
         * the way things are working ! */
        if (face_i1->at_boundary() && face_i1->boundary_id() == Dirichlet_BC_Index)
        {
          cell.Face_ID_in_this_rank[i_face] = homogenous_dirichlet;
          cell.Face_ID_in_all_ranks[i_face] = homogenous_dirichlet;
          cell.BCs[i_face] = Cell_Class<dim>::Dirichlet;
        }
        else if (face_i1->at_boundary() && face_i1->boundary_id() == Neumann_BC_Index)
        {
          cell.Face_ID_in_this_rank[i_face] = local_face_id_on_this_rank;
          cell.Face_ID_in_all_ranks[i_face] = global_face_id_on_this_rank;
          cell.BCs[i_face] = Cell_Class<dim>::Neumann;
          cell.face_owner_rank[i_face] = comm_rank;
          ++global_face_id_on_this_rank;
          ++local_face_id_on_this_rank;
        }
        else
        {
          /* At this point, we are sure that the cell has a neighbor. We will
           * have three cases:
           *
           * 1- The neighbor is coarser than the cell. This can only happen if
           *    the neighbor is a ghost cell, otherwise there is something
           *    wrong. So, when the neighbor is ghost, this subdomain does not
           *    own the face. Hence, we have to take the face number from the
           *    corresponding neighboer.
           *
           * 2- The neighbor is finer. In this case the face is owned by this
           *    subdomain, but we will have two subcases:
           *   2a- If the neighbor is in this subdomain, we act as if the domain
           *       was not decomposed.
           *   2b- If the neighbor is in some other subdomain, we have to also
           *       send the face number to all those finer neighbors, along with
           *       the corresponding subface id.
           *
           * 3- The face has neighbors of same refinement. This case is somehow
           *    trichier than what is looks. Because, you have to decide where
           *    face belongs to. As we said before, the face belongs to the
           *    domain which has smaller rank. So, we have to send the face
           *    number from the smaller rank to the higher rank.
           */
          if (cell.dealii_Cell->neighbor_is_coarser(i_face))
          {
            /*
             * The neighbor should be a ghost, because in each subdomain, the
             * elements are ordered from coarse to fine.
             */
            Cell_Type &&nb_i1 = cell.dealii_Cell->neighbor(i_face);
            assert(nb_i1->is_ghost());
            int face_nb_num = cell.dealii_Cell->neighbor_face_no(i_face);
            const auto &face_nb = nb_i1->face(face_nb_num);
            unsigned nb_face_of_nb_num = nb_i1->neighbor_face_no(face_nb_num);
            for (unsigned i_nb_subface = 0; i_nb_subface < face_nb->n_children();
                 ++i_nb_subface)
            {
              const Cell_Type &nb_of_nb_i1 =
               nb_i1->neighbor_child_on_subface(face_nb_num, i_nb_subface);
              if (nb_of_nb_i1->subdomain_id() == comm_rank)
              {
                std::stringstream nb_of_nb_ss_id;
                nb_of_nb_ss_id << nb_of_nb_i1->id();
                std::string nb_of_nb_str_id = nb_of_nb_ss_id.str();
                assert(cell_ID_to_num.find(nb_of_nb_str_id) != cell_ID_to_num.end());
                unsigned nb_of_nb_num = cell_ID_to_num[nb_of_nb_str_id];
                All_Owned_Cells[nb_of_nb_num].Face_ID_in_this_rank[nb_face_of_nb_num] =
                 local_face_id_on_this_rank;
                All_Owned_Cells[nb_of_nb_num].half_range_flag[nb_face_of_nb_num] =
                 i_nb_subface + 1;
                All_Owned_Cells[nb_of_nb_num].face_owner_rank[nb_face_of_nb_num] =
                 nb_i1->subdomain_id();
                face_to_rank_recver[nb_i1->subdomain_id()]++;
                if (!is_there_a_msg_from_rank[nb_i1->subdomain_id()])
                  is_there_a_msg_from_rank[nb_i1->subdomain_id()] = true;
                ++mpi_status_counter;
              }
            }
            ++local_face_id_on_this_rank;
          }
          else if (face_i1->has_children())
          {
            cell.Face_ID_in_this_rank[i_face] = local_face_id_on_this_rank;
            cell.half_range_flag[i_face] = 0;
            cell.Face_ID_in_all_ranks[i_face] = global_face_id_on_this_rank;
            cell.face_owner_rank[i_face] = comm_rank;
            for (unsigned i_subface = 0; i_subface < face_i1->number_of_children();
                 ++i_subface)
            {
              Cell_Type &&nb_i1 =
               cell.dealii_Cell->neighbor_child_on_subface(i_face, i_subface);
              int face_nb_i1 = cell.dealii_Cell->neighbor_face_no(i_face);
              std::stringstream nb_ss_id;
              nb_ss_id << nb_i1->id();
              std::string nb_str_id = nb_ss_id.str();
              if (nb_i1->subdomain_id() == comm_rank)
              {
                assert(cell_ID_to_num.find(nb_str_id) != cell_ID_to_num.end());
                int nb_i1_num = cell_ID_to_num[nb_str_id];
                All_Owned_Cells[nb_i1_num].Face_ID_in_this_rank[face_nb_i1] =
                 local_face_id_on_this_rank;
                All_Owned_Cells[nb_i1_num].half_range_flag[face_nb_i1] =
                 i_subface + 1;
                All_Owned_Cells[nb_i1_num].Face_ID_in_all_ranks[face_nb_i1] =
                 global_face_id_on_this_rank;
                All_Owned_Cells[nb_i1_num].face_owner_rank[face_nb_i1] = comm_rank;
              }
              else
              {
                /* Here, we are sure that the face is not owned by this rank.
                 * Hence, we do not bother to know if the beighbor subdomain is
                 * greater or smaller than the current rank.
                 */
                assert(nb_i1->subdomain_id() != comm_rank);
                assert(nb_i1->is_ghost());
                assert(Ghost_ID_to_num.find(nb_str_id) != Ghost_ID_to_num.end());
                unsigned nb_i1_num = Ghost_ID_to_num[nb_str_id];
                All_Ghost_Cells[nb_i1_num].Face_ID_in_this_rank[face_nb_i1] =
                 local_face_id_on_this_rank;
                All_Ghost_Cells[nb_i1_num].half_range_flag[face_nb_i1] =
                 i_subface + 1;
                All_Ghost_Cells[nb_i1_num].Face_ID_in_all_ranks[face_nb_i1] =
                 global_face_id_on_this_rank;
                All_Ghost_Cells[nb_i1_num].face_owner_rank[face_nb_i1] = comm_rank;
                /* Now we send id, face id, subface id, and neighbor face number
                 * to the corresponding rank. */
                char buffer[300];
                std::snprintf(buffer,
                              300,
                              "%s#%d#%d#%d",
                              nb_str_id.c_str(),
                              face_nb_i1,
                              i_subface + 1,
                              global_face_id_on_this_rank);
                face_to_rank_sender[nb_i1->subdomain_id()].push_back(buffer);
                ++mpi_request_counter;
              }
            }
            ++local_face_id_on_this_rank;
            ++global_face_id_on_this_rank;
          }
          else
          {
            cell.Face_ID_in_this_rank[i_face] = local_face_id_on_this_rank;
            cell.half_range_flag[i_face] = 0;
            Cell_Type &&nb_i1 = cell.dealii_Cell->neighbor(i_face);
            int face_nb_i1 = cell.dealii_Cell->neighbor_face_no(i_face);
            std::stringstream nb_ss_id;
            nb_ss_id << nb_i1->id();
            std::string nb_str_id = nb_ss_id.str();
            if (nb_i1->subdomain_id() == comm_rank)
            {
              assert(cell_ID_to_num.find(nb_str_id) != cell_ID_to_num.end());
              int nb_i1_num = cell_ID_to_num[nb_str_id];
              cell.Face_ID_in_all_ranks[i_face] = global_face_id_on_this_rank;
              cell.face_owner_rank[i_face] = comm_rank;
              All_Owned_Cells[nb_i1_num].Face_ID_in_this_rank[face_nb_i1] =
               local_face_id_on_this_rank;
              All_Owned_Cells[nb_i1_num].half_range_flag[face_nb_i1] = 0;
              All_Owned_Cells[nb_i1_num].Face_ID_in_all_ranks[face_nb_i1] =
               global_face_id_on_this_rank;
              All_Owned_Cells[nb_i1_num].face_owner_rank[face_nb_i1] = comm_rank;
              ++global_face_id_on_this_rank;
            }
            else
            {
              assert(nb_i1->subdomain_id() != comm_rank);
              assert(nb_i1->is_ghost());
              if (nb_i1->subdomain_id() > comm_rank)
              {
                cell.Face_ID_in_all_ranks[i_face] = global_face_id_on_this_rank;
                cell.face_owner_rank[i_face] = comm_rank;
                assert(Ghost_ID_to_num.find(nb_str_id) != Ghost_ID_to_num.end());
                unsigned nb_i1_num = Ghost_ID_to_num[nb_str_id];
                All_Ghost_Cells[nb_i1_num].Face_ID_in_this_rank[face_nb_i1] =
                 local_face_id_on_this_rank;
                All_Ghost_Cells[nb_i1_num].half_range_flag[face_nb_i1] = 0;
                All_Ghost_Cells[nb_i1_num].Face_ID_in_all_ranks[face_nb_i1] =
                 global_face_id_on_this_rank;
                All_Ghost_Cells[nb_i1_num].face_owner_rank[face_nb_i1] = comm_rank;
                /* Now we send id, face id, subface(=0), and neighbor face
                 * number to the corresponding rank. */
                char buffer[300];
                std::snprintf(buffer,
                              300,
                              "%s#%d#%d#%d",
                              nb_str_id.c_str(),
                              face_nb_i1,
                              0,
                              global_face_id_on_this_rank);
                face_to_rank_sender[nb_i1->subdomain_id()].push_back(buffer);
                ++global_face_id_on_this_rank;
                ++mpi_request_counter;
              }
              else
              {
                cell.face_owner_rank[i_face] = nb_i1->subdomain_id();
                face_to_rank_recver[nb_i1->subdomain_id()]++;
                if (!is_there_a_msg_from_rank[nb_i1->subdomain_id()])
                  is_there_a_msg_from_rank[nb_i1->subdomain_id()] = true;
                ++mpi_status_counter;
              }
            }
            ++local_face_id_on_this_rank;
          }
        }
      }
    }
  }

  /* We start ghost face numbers from -10, and go down. */
  int ghost_face_counter = -10;
  for (Cell_Class<dim> &ghost_cell : All_Ghost_Cells)
  {
    for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
    {
      if (ghost_cell.Face_ID_in_this_rank[i_face] == -2)
      {
        const auto &face_i1 = ghost_cell.dealii_Cell->face(i_face);
        /* The basic case corresponds to face_i1 being on the boundary.
         * In this case we only need to set the number of current face,
         * and we do not bother to know what is going on, on the other
         * side of this face. Because (in the current version), that is
         * the way things are working ! */
        if (face_i1->at_boundary() && face_i1->boundary_id() == Dirichlet_BC_Index)
        {
          ghost_cell.Face_ID_in_this_rank[i_face] = homogenous_dirichlet;
          ghost_cell.Face_ID_in_all_ranks[i_face] = homogenous_dirichlet;
          ghost_cell.BCs[i_face] = Cell_Class<dim>::Dirichlet;
        }
        else if (face_i1->at_boundary() && face_i1->boundary_id() == Neumann_BC_Index)
        {
          ghost_cell.Face_ID_in_this_rank[i_face] = ghost_face_counter;
          ghost_cell.Face_ID_in_all_ranks[i_face] = ghost_face_counter;
          ghost_cell.BCs[i_face] = Cell_Class<dim>::Neumann;
          ghost_cell.face_owner_rank[i_face] = ghost_cell.dealii_Cell->subdomain_id();
          --ghost_face_counter;
        }
        else
        {
          /*
           * We are sure that the face that we are on, is either on the coarser
           * side of an owned cell, or belongs to a lower rank than thr current
           * rank.
           */
          ghost_cell.Face_ID_in_this_rank[i_face] = ghost_face_counter;
          ghost_cell.Face_ID_in_all_ranks[i_face] = ghost_face_counter;
          ghost_cell.half_range_flag[i_face] = 0;
          ghost_cell.face_owner_rank[i_face] = ghost_cell.dealii_Cell->subdomain_id();
          if (face_i1->has_children())
          {
            int face_nb_subface = ghost_cell.dealii_Cell->neighbor_face_no(i_face);
            for (unsigned i_subface = 0; i_subface < face_i1->number_of_children();
                 ++i_subface)
            {
              Cell_Type &&nb_subface =
               ghost_cell.dealii_Cell->neighbor_child_on_subface(i_face, i_subface);
              if (nb_subface->is_ghost())
              {
                std::stringstream nb_ss_id;
                nb_ss_id << nb_subface->id();
                std::string nb_str_id = nb_ss_id.str();
                assert(Ghost_ID_to_num.find(nb_str_id) != Ghost_ID_to_num.end());
                int nb_subface_num = Ghost_ID_to_num[nb_str_id];
                assert(All_Ghost_Cells[nb_subface_num]
                        .Face_ID_in_this_rank[face_nb_subface] == -2);
                assert(All_Ghost_Cells[nb_subface_num]
                        .Face_ID_in_all_ranks[face_nb_subface] == -2);
                All_Ghost_Cells[nb_subface_num]
                 .Face_ID_in_this_rank[face_nb_subface] = ghost_face_counter;
                All_Ghost_Cells[nb_subface_num]
                 .Face_ID_in_all_ranks[face_nb_subface] = ghost_face_counter;
                All_Ghost_Cells[nb_subface_num].half_range_flag[face_nb_subface] =
                 i_subface + 1;
                All_Ghost_Cells[nb_subface_num].face_owner_rank[face_nb_subface] =
                 nb_subface->subdomain_id();
              }
            }
          }
          else if (ghost_cell.dealii_Cell->neighbor(i_face)->is_ghost())
          {
            Cell_Type &&nb_i1 = ghost_cell.dealii_Cell->neighbor(i_face);
            int face_nb_i1 = ghost_cell.dealii_Cell->neighbor_face_no(i_face);
            std::stringstream nb_ss_id;
            nb_ss_id << nb_i1->id();
            std::string nb_str_id = nb_ss_id.str();
            assert(Ghost_ID_to_num.find(nb_str_id) != Ghost_ID_to_num.end());
            int nb_i1_num = Ghost_ID_to_num[nb_str_id];
            assert(All_Ghost_Cells[nb_i1_num].Face_ID_in_this_rank[face_nb_i1] == -2);
            assert(All_Ghost_Cells[nb_i1_num].Face_ID_in_all_ranks[face_nb_i1] == -2);
            All_Ghost_Cells[nb_i1_num].Face_ID_in_this_rank[face_nb_i1] =
             ghost_face_counter;
            All_Ghost_Cells[nb_i1_num].Face_ID_in_all_ranks[face_nb_i1] =
             ghost_face_counter;
            All_Ghost_Cells[nb_i1_num].half_range_flag[face_nb_i1] = 0;
            All_Ghost_Cells[nb_i1_num].face_owner_rank[face_nb_i1] =
             nb_i1->subdomain_id();
          }
          --ghost_face_counter;
        }
      }
    }
  }

  face_count_up_to_rank.resize(comm_size, 0);
  face_count_before_rank.resize(comm_size, 0);
  unsigned number_of_faces_on_this_rank = global_face_id_on_this_rank;
  MPI_Allgather(&number_of_faces_on_this_rank,
                1,
                MPI_UNSIGNED,
                face_count_up_to_rank.data(),
                1,
                MPI_UNSIGNED,
                comm);

  for (unsigned i_num = 0; i_num < comm_size; ++i_num)
    for (unsigned j_num = 0; j_num < i_num; ++j_num)
      face_count_before_rank[i_num] += face_count_up_to_rank[j_num];

  for (Cell_Class<dim> &cell : All_Owned_Cells)
  {
    for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
    {
      if (cell.Face_ID_in_all_ranks[i_face] >= 0)
        cell.Face_ID_in_all_ranks[i_face] += face_count_before_rank[comm_rank];
    }
  }

  for (Cell_Class<dim> &ghost_cell : All_Ghost_Cells)
  {
    for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
    {
      if (ghost_cell.Face_ID_in_all_ranks[i_face] >= 0)
        ghost_cell.Face_ID_in_all_ranks[i_face] += face_count_before_rank[comm_rank];
    }
  }

  for (auto &&i_send = face_to_rank_sender.rbegin();
       i_send != face_to_rank_sender.rend();
       ++i_send)
  {
    assert(comm_rank != i_send->first);
    if (comm_rank < i_send->first)
    {
      /*
      std::cout << comm_rank
                << " sends: " << face_to_rank_sender[i_send->first].size()
                << " to " << i_send->first << std::endl;
      */
      unsigned num_sends = face_to_rank_sender[i_send->first].size();
      unsigned jth_rank_on_i_send = 0;
      std::vector<MPI_Request> all_mpi_reqs_of_rank(num_sends);
      for (auto &&msg_it : face_to_rank_sender[i_send->first])
      {
        MPI_Isend((char *)msg_it.c_str(),
                  msg_it.size() + 1,
                  MPI_CHAR,
                  i_send->first,
                  refn_cycle,
                  comm,
                  &all_mpi_reqs_of_rank[jth_rank_on_i_send]);
        ++jth_rank_on_i_send;
      }
      MPI_Waitall(num_sends, all_mpi_reqs_of_rank.data(), MPI_STATUSES_IGNORE);
    }
  }

  std::vector<MPI_Status> all_mpi_stats_of_rank(mpi_status_counter);
  unsigned recv_counter = 0;

  bool no_msg_left = (is_there_a_msg_from_rank.size() == 0);
  while (!no_msg_left)
  {
    auto i_recv = is_there_a_msg_from_rank.begin();
    no_msg_left = true;
    for (; i_recv != is_there_a_msg_from_rank.end(); ++i_recv)
    {
      if (i_recv->second && comm_rank > i_recv->first)
        no_msg_left = false;
      int flag = 0;
      if (comm_rank > i_recv->first)
        MPI_Iprobe(i_recv->first, refn_cycle, comm, &flag, MPI_STATUS_IGNORE);
      if (flag)
      {
        assert(i_recv->second);
        break;
      }
    }
    if (i_recv != is_there_a_msg_from_rank.end())
    {
      /*
      std::cout << comm_rank << " recvs: " << face_to_rank_recver[i_recv->first]
                << " from " << i_recv->first << std::endl;
      */
      for (unsigned jth_rank_on_i_recv = 0;
           jth_rank_on_i_recv < face_to_rank_recver[i_recv->first];
           ++jth_rank_on_i_recv)
      {
        char buffer[300];
        MPI_Recv(&buffer[0],
                 300,
                 MPI_CHAR,
                 i_recv->first,
                 refn_cycle,
                 comm,
                 &all_mpi_stats_of_rank[recv_counter]);
        std::vector<std::string> tokens;
        Tokenize(buffer, tokens, "#");
        assert(tokens.size() == 4);
        std::string cell_unique_id = tokens[0];
        assert(cell_ID_to_num.find(cell_unique_id) != cell_ID_to_num.end());
        int cell_number = cell_ID_to_num[cell_unique_id];
        unsigned face_num = std::stoi(tokens[1]);
        assert(All_Owned_Cells[cell_number].Face_ID_in_all_ranks[face_num] == -2);
        All_Owned_Cells[cell_number].Face_ID_in_all_ranks[face_num] =
         std::stoi(tokens[3]) + face_count_before_rank[i_recv->first];
        ++recv_counter;
      }
      i_recv->second = false;
    }
  }

  for (auto &&i_send = face_to_rank_sender.rbegin();
       i_send != face_to_rank_sender.rend();
       ++i_send)
  {
    assert(comm_rank != i_send->first);
    if (comm_rank > i_send->first)
    {
      /*
      std::cout << comm_rank
                << " sends: " << face_to_rank_sender[i_send->first].size()
                << " to " << i_send->first << std::endl;
      */
      unsigned num_sends = face_to_rank_sender[i_send->first].size();
      unsigned jth_rank_on_i_send = 0;
      std::vector<MPI_Request> all_mpi_reqs_of_rank(num_sends);
      for (auto &&msg_it : face_to_rank_sender[i_send->first])
      {
        MPI_Isend((char *)msg_it.c_str(),
                  msg_it.size() + 1,
                  MPI_CHAR,
                  i_send->first,
                  refn_cycle,
                  comm,
                  &all_mpi_reqs_of_rank[jth_rank_on_i_send]);
        ++jth_rank_on_i_send;
      }
      MPI_Waitall(num_sends, all_mpi_reqs_of_rank.data(), MPI_STATUSES_IGNORE);
    }
  }

  no_msg_left = (is_there_a_msg_from_rank.size() == 0);
  while (!no_msg_left)
  {
    auto i_recv = is_there_a_msg_from_rank.begin();
    no_msg_left = true;
    for (; i_recv != is_there_a_msg_from_rank.end(); ++i_recv)
    {
      if (i_recv->second && comm_rank < i_recv->first)
        no_msg_left = false;
      int flag = 0;
      if (comm_rank < i_recv->first)
        MPI_Iprobe(i_recv->first, refn_cycle, comm, &flag, MPI_STATUS_IGNORE);
      if (flag)
      {
        assert(i_recv->second);
        break;
      }
    }
    if (i_recv != is_there_a_msg_from_rank.end())
    {
      /*
      std::cout << comm_rank << " recvs: " << face_to_rank_recver[i_recv->first]
                << " from " << i_recv->first << std::endl;
      */
      for (unsigned jth_rank_on_i_recv = 0;
           jth_rank_on_i_recv < face_to_rank_recver[i_recv->first];
           ++jth_rank_on_i_recv)
      {
        char buffer[300];
        MPI_Recv(&buffer[0],
                 300,
                 MPI_CHAR,
                 i_recv->first,
                 refn_cycle,
                 comm,
                 &all_mpi_stats_of_rank[recv_counter]);
        std::vector<std::string> tokens;
        Tokenize(buffer, tokens, "#");
        assert(tokens.size() == 4);
        std::string cell_unique_id = tokens[0];
        assert(cell_ID_to_num.find(cell_unique_id) != cell_ID_to_num.end());
        int cell_number = cell_ID_to_num[cell_unique_id];
        unsigned face_num = std::stoi(tokens[1]);
        assert(All_Owned_Cells[cell_number].Face_ID_in_all_ranks[face_num] == -2);
        All_Owned_Cells[cell_number].Face_ID_in_all_ranks[face_num] =
         std::stoi(tokens[3]) + face_count_before_rank[i_recv->first];
        ++recv_counter;
      }
      i_recv->second = false;
    }
  }

  /*           THESE NEXT LOOPS ARE JUST FOR PETSc !!
   *
   * When you want to preallocate stiffness matrix in PETSc, it
   * accpet an argument which contains the number DOFs connected to
   * the DOF in each row. According to PETSc, if you let PETSc know
   * about this preallocation, you will get a noticeable performance
   * boost.
   */


  /* Let us build a vector containing each unique face which belongs
   * to this rank.
   */
  std::vector<Face_Class<dim>> All_Faces(global_face_id_on_this_rank);
  for (typename Cell_Class<dim>::vec_iterator_type cell_it = All_Owned_Cells.begin();
       cell_it != All_Owned_Cells.end();
       ++cell_it)
  {
    for (unsigned i1 = 0; i1 < cell_it->n_faces; ++i1)
    {
      int face_i1 =
       cell_it->Face_ID_in_all_ranks[i1] - face_count_before_rank[comm_rank];
      if (cell_it->face_owner_rank[i1] == comm_rank)
      {
        All_Faces[face_i1].Parent_Cells.push_back(cell_it);
        All_Faces[face_i1].connected_face_of_parent_cell.push_back(i1);
      }
    }
  }

  for (typename Cell_Class<dim>::vec_iterator_type ghost_cell_it =
        All_Ghost_Cells.begin();
       ghost_cell_it != All_Ghost_Cells.end();
       ++ghost_cell_it)
  {
    for (unsigned i1 = 0; i1 < ghost_cell_it->n_faces; ++i1)
    {
      int face_i1 = ghost_cell_it->Face_ID_in_all_ranks[i1] -
                    face_count_before_rank[comm_rank];
      if (ghost_cell_it->face_owner_rank[i1] == comm_rank)
      {
        All_Faces[face_i1].Parent_Ghosts.push_back(ghost_cell_it);
        All_Faces[face_i1].connected_face_of_parent_ghost.push_back(i1);
      }
    }
  }

  /* Now we count the number of global dofs in the rank, and also fill
   */
  num_global_DOFs_on_this_rank = 0;
  for (Face_Class<dim> &face : All_Faces)
  {
    face.n_local_connected_faces++;
    face.num_global_DOFs = n_polyface;
    num_global_DOFs_on_this_rank += face.num_global_DOFs;
  }

  for (Face_Class<dim> &face : All_Faces)
  {
    std::map<int, unsigned> local_face_num_map;
    std::map<int, unsigned> nonlocal_face_num_map;
    for (unsigned i_parent_cell = 0; i_parent_cell < face.Parent_Cells.size();
         ++i_parent_cell)
    {
      auto parent_cell = face.Parent_Cells[i_parent_cell];
      for (unsigned face_j_of_cell = 0; face_j_of_cell < n_faces_per_cell; ++face_j_of_cell)
        if (face_j_of_cell != face.connected_face_of_parent_cell[i_parent_cell])
        {
          if (parent_cell->face_owner_rank[face_j_of_cell] == comm_rank)
            local_face_num_map[parent_cell->Face_ID_in_all_ranks[face_j_of_cell]]++;
          else if (parent_cell->Face_ID_in_all_ranks[face_j_of_cell] >= 0)
            nonlocal_face_num_map[parent_cell->Face_ID_in_all_ranks[face_j_of_cell]]++;
        }
    }
    for (unsigned i_parent_ghost = 0; i_parent_ghost < face.Parent_Ghosts.size();
         ++i_parent_ghost)
    {
      auto parent_ghost = face.Parent_Ghosts[i_parent_ghost];
      for (unsigned face_j_of_ghost = 0; face_j_of_ghost < n_faces_per_cell;
           ++face_j_of_ghost)
        if (face_j_of_ghost != face.connected_face_of_parent_ghost[i_parent_ghost])
        {
          if (parent_ghost->face_owner_rank[face_j_of_ghost] == comm_rank)
            local_face_num_map[parent_ghost->Face_ID_in_all_ranks[face_j_of_ghost]]++;
          else if (parent_ghost->Face_ID_in_all_ranks[face_j_of_ghost] <= -10)
            nonlocal_face_num_map[parent_ghost->Face_ID_in_all_ranks[face_j_of_ghost]]++;
        }
    }
    face.n_local_connected_faces += local_face_num_map.size();
    face.n_nonlocal_connected_faces += nonlocal_face_num_map.size();
  }

  for (Face_Class<dim> &face : All_Faces)
  {
    face.n_local_connected_DOFs = face.n_local_connected_faces * n_polyface;
    face.n_nonlocal_connected_DOFs = face.n_nonlocal_connected_faces * n_polyface;
  }

  MPI_Allreduce(&num_global_DOFs_on_this_rank,
                &num_global_DOFs_on_all_ranks,
                1,
                MPI_UNSIGNED,
                MPI_SUM,
                comm);

  unsigned DOF_Counter1 = 0;
  n_local_DOFs_connected_to_DOF.resize(num_global_DOFs_on_this_rank);
  n_nonlocal_DOFs_connected_to_DOF.resize(num_global_DOFs_on_this_rank);
  for (Face_Class<dim> &face : All_Faces)
  {
    for (unsigned DOF_on_face = 0; DOF_on_face < face.num_global_DOFs; ++DOF_on_face)
    {
      n_local_DOFs_connected_to_DOF[DOF_Counter1 + DOF_on_face] +=
       face.n_local_connected_DOFs;
      n_nonlocal_DOFs_connected_to_DOF[DOF_Counter1 + DOF_on_face] +=
       face.n_nonlocal_connected_DOFs;
    }
    DOF_Counter1 += face.num_global_DOFs;
  }

  std::map<unsigned, unsigned> map_from_global_to_local;
  for (Cell_Class<dim> &cell : All_Owned_Cells)
  {
    for (unsigned i_face = 0; i_face < n_faces_per_cell; ++i_face)
    {
      int index1 = cell.Face_ID_in_this_rank[i_face];
      int index2 = cell.Face_ID_in_all_ranks[i_face];
      if (index1 != -1)
      {
        map_from_global_to_local[index2] = index1;
      }
    }
  }

  num_local_DOFs_on_this_rank = local_face_id_on_this_rank * n_polyface;
  scatter_from.reserve(num_local_DOFs_on_this_rank);
  scatter_to.reserve(num_local_DOFs_on_this_rank);
  for (const auto &map_it : map_from_global_to_local)
  {
    for (unsigned i_polyface = 0; i_polyface < n_polyface; ++i_polyface)
    {
      scatter_from.push_back(map_it.first * n_polyface + i_polyface);
      scatter_to.push_back(map_it.second * n_polyface + i_polyface);
    }
  }

  /*
  for (const Face_Class<dim> &face : All_Faces)
  {
    dealii::Point<dim> p_center =
     face.Parent_Cells[0]
      ->dealii_Cell->face(face.connected_face_of_parent_cell[0])
      ->center();
    printf(" rank ID : %3d, el num. %3d, face center %10.3e , %10.3e : local "
           "%3d ; nonlocal %3d \n",
           comm_rank,
           face.Parent_Cells[0]->dealii_Cell->index(),
           p_center[0],
           p_center[1],
           face.n_local_connected_faces,
           face.n_nonlocal_connected_faces);
  }
  */

  char buffer[100];
  std::snprintf(buffer,
                100,
                "Number of DOFs in this rank is: %d and number of dofs in all "
                "ranks is : %d",
                num_global_DOFs_on_this_rank,
                num_global_DOFs_on_all_ranks);
  OutLogger(Execution_Time, buffer, true);
  //  std::cout << buffer << std::endl;
}

template <int dim>
void Diffusion<dim>::Write_Grid_Out()
{
  dealii::GridOut Grid1_Out;
  dealii::GridOutFlags::Svg svg_flags(
   1,                                       // line_thickness = 2,
   2,                                       // boundary_line_thickness = 4,
   false,                                   // margin = true,
   dealii::GridOutFlags::Svg::transparent,  // background = white,
   0,                                       // azimuth_angle = 0,
   0,                                       // polar_angle = 0,
   dealii::GridOutFlags::Svg::subdomain_id, // coloring = level_number,
   false, // convert_level_number_to_height = false,
   false, // label_level_number = true,
   true,  // label_cell_index = true,
   false, // label_material_id = false,
   false, // label_subdomain_id = false,
   true,  // draw_colorbar = true,
   true); // draw_legend = true
  Grid1_Out.set_flags(svg_flags);
  if (dim == 2)
  {
    std::ofstream Grid1_OutFile("Grid1" + std::to_string(refn_cycle) +
                                std::to_string(comm_rank) + ".svg");
    Grid1_Out.write_svg(Grid1, Grid1_OutFile);
  }
  else
  {
    std::ofstream Grid1_OutFile("Grid1" + std::to_string(refn_cycle) +
                                std::to_string(comm_rank) + ".msh");
    Grid1_Out.write_msh(Grid1, Grid1_OutFile);
  }
}

template <typename T>
void Wreck_it_Ralph(T &Wreckee)
{
  T Wrecker;
  Wrecker.swap(Wreckee);
}

template <int dim>
void Diffusion<dim>::FreeUpContainers()
{
  Wreck_it_Ralph(All_Owned_Cells);
  Wreck_it_Ralph(n_local_DOFs_connected_to_DOF);
  Wreck_it_Ralph(n_nonlocal_DOFs_connected_to_DOF);
  Wreck_it_Ralph(scatter_from);
  Wreck_it_Ralph(scatter_to);
  Wreck_it_Ralph(cell_ID_to_num);
  Wreck_it_Ralph(face_to_rank_sender);
  Wreck_it_Ralph(face_to_rank_recver);
  Wreck_it_Ralph(face_count_before_rank);
  Wreck_it_Ralph(face_count_up_to_rank);
}
