#include "diffusion.hpp"

template <int dim>
void Diffusion_0<dim>::OutLogger(std::ostream &logger, const std::string &log, bool insert_eol)
{
  if (comm_rank == 0)
  {
    if (insert_eol)
      logger << log << std::endl;
    else
      logger << log;
  }
}

void Tokenize(const std::string &str_in,
              std::vector<std::string> &tokens,
              const std::string &delimiters = " ")
{
  auto lastPos = str_in.find_first_not_of(delimiters, 0);
  auto pos = str_in.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos)
  {
    tokens.push_back(str_in.substr(lastPos, pos - lastPos));
    lastPos = str_in.find_first_not_of(delimiters, pos);
    pos = str_in.find_first_of(delimiters, lastPos);
  }
}


/*
 * This is the main class in this program. I will elaborate, later !
 */

template <int dim>
PetscErrorCode Diffusion_0<dim>::Solve_Linear_Systam()
{
  int rows_owned_lo, rows_owned_hi;
  MatCreate(comm, &global_mat);
  MatSetType(global_mat, MATMPIAIJ);
  MatSetSizes(global_mat,
              num_global_DOFs_on_this_rank,
              num_global_DOFs_on_this_rank,
              num_global_DOFs_on_all_ranks,
              num_global_DOFs_on_all_ranks);

  MatMPIAIJSetPreallocation(global_mat,
                            0,
                            n_local_DOFs_connected_to_DOF.data(),
                            0,
                            n_nonlocal_DOFs_connected_to_DOF.data());

  /*
  std::cout << "rank ID : " << comm_rank << "   " << current_refinement_level
            << "   " << n_local_DOFs_connected_to_DOF.size() << std::endl;
  */

  MatGetOwnershipRange(global_mat, &rows_owned_lo, &rows_owned_hi);
  MatSetOption(global_mat, MAT_ROW_ORIENTED, PETSC_FALSE);
  MatSetOption(global_mat, MAT_SPD, PETSC_TRUE);

  VecCreateMPI(comm, num_global_DOFs_on_this_rank, num_global_DOFs_on_all_ranks, &RHS_vec);
  VecSetOption(RHS_vec, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
  VecDuplicate(RHS_vec, &solution_vec);
  VecDuplicate(RHS_vec, &exact_solution);

  double t11, t12, t13, t21, t22, t23;
  if (comm_rank == 0)
    Execution_Time << "Entering assembly : " << currentDateTime() << std::endl;
  t11 = MPI_Wtime();
  Assemble_Globals();
  t21 = MPI_Wtime();
  if (comm_rank == 0)
    Execution_Time << "Has finished assembly : " << currentDateTime() << std::endl;

  if (comm_rank == 0)
    Execution_Time << "Entering solver : " << currentDateTime() << std::endl;

  t12 = MPI_Wtime();
  PetscErrorCode assem_error = MatAssemblyBegin(global_mat, MAT_FINAL_ASSEMBLY);
  CHKERRQ(assem_error);
  assem_error = MatAssemblyEnd(global_mat, MAT_FINAL_ASSEMBLY);
  CHKERRQ(assem_error);

  VecAssemblyBegin(RHS_vec);
  VecAssemblyEnd(RHS_vec);
  double rhs_norm;
  VecNorm(RHS_vec, NORM_2, &rhs_norm);

  VecAssemblyBegin(exact_solution);
  VecAssemblyEnd(exact_solution);

  KSP TheSolver;
  KSPConvergedReason How_KSP_Stopped;
  KSPCreate(comm, &TheSolver);
  KSPSetTolerances(TheSolver, 1E-8, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
  KSPSetOperators(TheSolver, global_mat, global_mat);
  KSPSetType(TheSolver, KSPCG);
  KSPSetFromOptions(TheSolver);

  PC ThePreCond;
  KSPGetPC(TheSolver, &ThePreCond);
  PCSetFromOptions(ThePreCond);
  PCSetType(ThePreCond, PCGAMG);
  PCGAMGSetType(ThePreCond, PCGAMGAGG);
  PCGAMGSetNSmooths(ThePreCond, 1);

  int num_iter;
  KSPSolve(TheSolver, RHS_vec, solution_vec);
  KSPGetIterationNumber(TheSolver, &num_iter);
  KSPGetConvergedReason(TheSolver, &How_KSP_Stopped);
  if (comm_rank == 0)
  {
    Execution_Time << "Converged reason is: " << How_KSP_Stopped << std::endl;
    Execution_Time << "Number of iterations is: " << num_iter << std::endl;
  }
  double solution_norm;
  VecNorm(solution_vec, NORM_2, &solution_norm);
  if (comm_rank == 0)
    Execution_Time << "Finished solver : " << currentDateTime() << std::endl;

  double accuracy;
  VecAXPY(exact_solution, -1, solution_vec);
  VecNorm(exact_solution, NORM_2, &accuracy);

  IS from, to;
  Vec x;
  VecScatter scatter;
  VecCreateSeq(PETSC_COMM_SELF, num_local_DOFs_on_this_rank, &x);
  ISCreateGeneral(PETSC_COMM_SELF,
                  num_local_DOFs_on_this_rank,
                  &scatter_from[0],
                  PETSC_COPY_VALUES,
                  &from);
  ISCreateGeneral(PETSC_COMM_SELF,
                  num_local_DOFs_on_this_rank,
                  &scatter_to[0],
                  PETSC_COPY_VALUES,
                  &to);
  //  VecView(x, PETSC_VIEWER_STDOUT_SELF);
  VecScatterCreate(solution_vec, from, x, to, &scatter);
  VecScatterBegin(scatter, solution_vec, x, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(scatter, solution_vec, x, INSERT_VALUES, SCATTER_FORWARD);

  double *local_solution_vec_p;
  VecGetArray(x, &local_solution_vec_p);
  t22 = MPI_Wtime();
  /*
  std::vector<double> local_solution_vec(
   std::move_iterator<double *>(local_solution_vec_p),
   std::move_iterator<double *>(local_solution_vec_p +
  num_global_DOFs_on_this_rank));
  */

  if (comm_rank == 0)
    Execution_Time << "Entering local solver : " << currentDateTime() << std::endl;
  t13 = MPI_Wtime();
  Calculate_Internal_Unknowns(local_solution_vec_p);
  t23 = MPI_Wtime();
  if (comm_rank == 0)
    Execution_Time << "Finished local solver : " << currentDateTime() << std::endl;

  if (comm_rank == 0)
    std::cout << t21 - t11 << " " << t22 - t12 << " " << t23 - t13 << std::endl;

  MatDestroy(&global_mat);
  VecDestroy(&RHS_vec);
  VecDestroy(&exact_solution);
  VecDestroy(&solution_vec);
  VecDestroy(&x);

  return 0;
}

template <int dim>
void Diffusion_0<dim>::Setup_System(unsigned refinement)
{
  char buffer[300];
  std::snprintf(buffer,
                300,
                "Rank %5d is in cycle %5d and entering refinement: ",
                comm_rank,
                refn_cycle);
  if (comm_rank == comm_size)
    std::cout << buffer << currentDateTime() << std::endl;
  Refine_Grid(refinement);
  std::snprintf(buffer,
                300,
                "Rank %5d is in cycle %5d and has exited  refinement: ",
                comm_rank,
                refn_cycle);
  if (comm_rank == comm_size)
    std::cout << buffer << currentDateTime() << std::endl;

  //  Write_Grid_Out();

  std::snprintf(buffer,
                300,
                "Rank %5d is in cycle %5d and is entering counter: ",
                comm_rank,
                refn_cycle);
  if (comm_rank == 0)
    Execution_Time << buffer << currentDateTime() << std::endl;

  Count_Globals();
  std::snprintf(buffer,
                300,
                "Rank %5d is in cycle %5d and has exited  counter: ",
                comm_rank,
                refn_cycle);
  if (comm_rank == 0)
    Execution_Time << buffer << currentDateTime() << std::endl;
}

const std::string currentDateTime()
{
  time_t now = time(0);
  struct tm tstruct;
  char buf[80];
  tstruct = *localtime(&now);
  std::strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
  return buf;
}

void parse_my_options(const int &rank,
                      const int &argc2,
                      char *args2[],
                      bool &Adaptive,
                      unsigned &p_1,
                      unsigned &p_2,
                      unsigned &h_1,
                      unsigned &h_2)
{
  static struct option long_options[] = { { "adaptive", no_argument, 0, 'a' },
                                          { "p_0", required_argument, 0, 'p' },
                                          { "p_n", required_argument, 0, 'q' },
                                          { "h_0", required_argument, 0, 'h' },
                                          { "h_n", required_argument, 0, 'l' },
                                          { 0, 0, 0, 0 } };

  //  p_1 = 3;
  //  p_2 = 4;
  //  h_1 = 2;
  //  h_2 = 8;
  int long_index = 0;
  int opt = 0;
  opterr = 0;
  while ((opt = getopt_long_only(argc2, args2, "ap:q:h:l:", long_options, &long_index)) !=
         -1)
  {
    switch (opt)
    {
    case 'a':
      Adaptive = true;
      if (rank == 0)
        std::cout << "Used -adaptive option; adaptive is on." << std::endl;
      break;
    case 'p':
      p_1 = atoi(optarg);
      if (rank == 0)
        std::cout << "Used -p_0 option; starting order is: " << p_1 << std::endl;
      break;
    case 'q':
      p_2 = atoi(optarg);
      if (rank == 0)
        std::cout << "Used -p_n option; final order is: " << p_2 << std::endl;
      break;
    case 'h':
      h_1 = atoi(optarg);
      if (rank == 0)
        std::cout << "Used -h_0 option; starting refinement cycle is: " << h_1
                  << std::endl;
      break;
    case 'l':
      h_2 = atoi(optarg);
      if (rank == 0)
        std::cout << "Used -h_n option; final refinement cycle is: " << h_2
                  << std::endl;
      break;
    default:
      break;
    }
  }
}

int main(int argc, char *args[])
{
  SlepcInitialize(&argc, &args, (char *)0, NULL);
  PetscMPIInt rank, size;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  dealii::MultithreadInfo::set_thread_limit(1);

  int number_of_threads = 1;
#ifdef _OPENMP
  omp_set_num_threads(1);
  number_of_threads = omp_get_max_threads();
#endif

  char buffer[100];
  std::snprintf(buffer, 100, "There are %d threads available.\n", number_of_threads);

  if (rank == 0)
  {
    char help_line[300];
    std::snprintf(help_line,
                  300,
                  "\n"
                  "mpiexec -n 8 ./A1 -h_0 2 -h_n 12 -p_0 1 -p_n 2 -amr 1 "
                  "-pc_type hypre -pc_hypre_type boomeramg "
                  "-pc_hypre_boomeramg_strong_threshold 0.25 "
                  "-pc_hypre_boomeramg_coarsen_type CLJP "
                  "-pc_hypre_boomeramg_max_levels 100 "
                  "-pc_hypre_boomeramg_interp_type standard -ksp_type cg "
                  "-face_basis modal"
                  "\n");
    std::cout << help_line << std::endl;
    std::ofstream Convergence_Cleaner("Convergence_Result.txt");
    Convergence_Cleaner.close();
    std::ofstream ExecTime_Cleaner("Execution_Time.txt");
    ExecTime_Cleaner.close();
  }

  int p_1, p_2, h_1, h_2, found_options = 1;
  char face_basis_type[100];
  strcpy(face_basis_type, "legendre");
  PetscBool found_option;
  int Adaptive = 0;

  // parse_my_options(rank, argc2, args2, Adaptive, p_1, p_2, h_1, h_2);

  PetscOptionsGetInt(NULL, "-p_0", &p_1, &found_option);
  found_options = found_option * found_options;
  PetscOptionsGetInt(NULL, "-p_n", &p_2, &found_option);
  found_options = found_option && found_options;
  PetscOptionsGetInt(NULL, "-h_0", &h_1, &found_option);
  found_options = found_option && found_options;
  PetscOptionsGetInt(NULL, "-h_n", &h_2, &found_option);
  found_options = found_option && found_options;
  PetscOptionsGetInt(NULL, "-amr", &Adaptive, &found_option);
  found_options = found_option && found_options;
  PetscBool face_basis_option_flag;
  PetscOptionsGetString(NULL, "-face_basis", face_basis_type, 100, &face_basis_option_flag);

  if (face_basis_option_flag == PETSC_TRUE)
  {
    if (strcmp(face_basis_type, "lagrange") == 0)
    {
    }
    else if (strcmp(face_basis_type, "legendre") == 0)
    {
    }
    else
    {
      if (rank == 0)
      {
        std::cout << " HEY! : The face basis type should either be "
                     "<nodal> or <modal> "
                     "(default). \n" << std::endl;
        std::cout << buffer << std::endl;
      }
    }
  }

  p_1 = 3;
  p_2 = 4;
  h_1 = 2;
  h_2 = 8;
  Adaptive = 0;
  const int dim = 2;

  for (unsigned p1 = (unsigned)p_1; p1 < (unsigned)p_2; ++p1)
  {
    Diffusion_0<dim> diff0(p1, PETSC_COMM_WORLD, size, rank, number_of_threads, Adaptive);
    for (unsigned h1 = (unsigned)h_1; h1 < (unsigned)h_2; ++h1)
    {
      diff0.Setup_System(h1);
      diff0.Solve_Linear_Systam();
      diff0.vtk_visualizer();
    }
  }

  SlepcFinalize();
  return 0;
}
