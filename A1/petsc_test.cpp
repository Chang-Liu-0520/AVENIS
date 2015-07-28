#include <petscsys.h>
#include <petscis.h>
#include <petscvec.h>
#include <iostream>
#include <vector>
#include <array>
#include <memory>
#include <assert.h>
#include <cstdio>

int My_Petsc_Test(int argc, char **args)
{

  /*
   * Define some variables before starting the node communication.
   */

  //  PetscInt nlocal, nghost, iform[2], ierr, rstart, rend, ione;
  PetscInt ierr = PetscInitialize(&argc, &args, (char *)0, NULL);
  CHKERRQ(ierr);

  PetscInt NEQMTotal;

  /*
   * Now we initialize the processors communication.
   */

  PetscMPIInt rank, size;
  MPI_Comm_rank(PETSC_COMM_WORLD,
                &rank);// This is the number of current processor
  MPI_Comm_size(PETSC_COMM_WORLD, &size);// The total size of the communicator
  assert(size == 2);

  /*
   * problem definition
   * Global PETSc numbering
   *
   *              --- #0 ---           ---- #1 ----
   *
   *             1 . . . . 3        . . . . 5 . . . . 7
   *             .         .                .         .
   *             .         .                .         .
   *             .         .                .         .
   *             .         .                .         .
   *             0 . . . . 2        . . . . 4 . . . . 6
   *
   *
   * Local numbering
   *
   *              --- #0 ---           ---- #1 ----
   *
   *             1 . . . . 3      1 . . . . 3 . . . . 5
   *             .         .      .         .         .
   *             .         .      .         .         .
   *             .         .      .         .         .
   *             .         .      .         .         .
   *             0 . . . . 2      0 . . . . 2 . . . . 4
   *
   */

  NEQMTotal = 8;

  PetscInt NEqM_Mapping, NEQM;
  std::vector<int> PETSc_Numbers, Indices, ND, idx_from, idx_to;
  if (rank == 0)
  {
    NEqM_Mapping = 4;               /* Global equation numbers */
    NEQM = 4;                       /* Local number of total of equations */
    PETSc_Numbers = { 0, 1, 2, 3 }; /* Global numbers on CPU #0 */
    Indices = { 0, 1, 2, 3 };
    ND = { 0, 1, 2, 3 };
    idx_from = { 0, 1, 2, 3 };// This index is used for global Vec to
    // send data to local vector.
    idx_to = { 0, 1, 2, 3 };// This index is used for local vector to
                            // gather data from global Vec.
  }
  if (rank == 1)
  {
    NEqM_Mapping = 4;
    NEQM = 6;
    PETSc_Numbers = { 4, 5, 6, 7 };
    Indices = { 2, 3, 4, 5, 6, 7 };
    ND = { 0, 1, 2, 3, 4, 5 };
    idx_from = { 2, 3, 4, 5, 6, 7 };// This index is used for global Vec to
    // send data to local vector.
    idx_to = { 0, 1, 2, 3, 4, 5 };// This index is used for local vector to
                                  // gather data from global Vec.
  }

  /*
   * Now we define the mapping from local numbering to global numbering
   */
  Vec v1;
  ISLocalToGlobalMapping GMapping;
  ierr = ISLocalToGlobalMappingCreate(
   PETSC_COMM_WORLD, 1, NEQM, &Indices[0], PETSC_COPY_VALUES, &GMapping);
  //  ierr = ISLocalToGlobalMappingView(GMapping, PETSC_VIEWER_STDOUT_WORLD);
  //  MPI_Barrier(PETSC_COMM_WORLD);
  //  return 0;
  ierr = VecCreateMPI(PETSC_COMM_WORLD, NEqM_Mapping, NEQMTotal, &v1);
  ierr = VecSetLocalToGlobalMapping(v1, GMapping);
  /*
   * Setting values on nodes, as below:
   *
   *              --- #0 ---           ---- #1 ----
   *
   *            11 . . . . 13    21 . . . . 23 . . . . 25
   *             .         .      .         .          .
   *             .         .      .         .          .
   *             .         .      .         .          .
   *             .         .      .         .          .
   *            10 . . . . 12    20 . . . . 22 . . . . 24
   *
   */
  ierr = VecSet(v1, 0.0);
  //  ierr = VecView(v1, PETSC_VIEWER_STDOUT_WORLD);
  std::vector<PetscScalar> values;
  if (rank == 0)
  {
    values = { 10, 11, 12, 13 };
  }
  else
  {
    values = { 20, 21, 22, 23, 24, 25 };
  }

  /*
   * Assembling values on nodes, to get:
   *
   *              --- #0 ---           ---- #1 ----
   *
   *            11 . . . . 34       . . . . 23 . . . . 25
   *             .         .                .          .
   *             .         .                .          .
   *             .         .                .          .
   *             .         .                .          .
   *            10 . . . . 32       . . . . 22 . . . . 24
   *
   */
  ierr = VecSetValuesLocal(v1, NEQM, ND.data(), values.data(), ADD_VALUES);
  VecAssemblyBegin(v1);
  VecAssemblyEnd(v1);
  //  ierr = VecView(v1, PETSC_VIEWER_STDOUT_WORLD);
  //  return 0;
  int IStart, IEnd;
  PetscScalar *x0;
  ierr = VecGetOwnershipRange(v1, &IStart, &IEnd);
  ierr = VecGetArray(v1, &x0);
  //  printf("Rank %d starts at %d and ends in %d\n", rank, IStart, IEnd);
  //  if (rank == 0) {
  //    for (int i1 = 0; i1 < IEnd - IStart; i1++) {
  //      std::cout << "Process " << rank << "value " << i1 << " " << x0[i1]
  //                << "\n";
  //    }
  //  }
  //  return 0;

  /*
   *
   * N O W,   I T   I S   T I M E    T O    O B T A I N
   *       W H A T   I S   O N   V E C T O R S
   *
   * So, the first and the most basic method is to scatter:
   */
  IS from, to;
  Vec x;
  PetscScalar *x_local;
  VecScatter scatter;
  VecCreateSeq(PETSC_COMM_SELF, NEQM, &x);
  ISCreateGeneral(PETSC_COMM_SELF, NEQM, &idx_from[0], PETSC_COPY_VALUES, &from);
  ISCreateGeneral(PETSC_COMM_SELF, NEQM, &idx_to[0], PETSC_COPY_VALUES, &to);
  //  VecView(x, PETSC_VIEWER_STDOUT_SELF);
  ierr = VecScatterCreate(v1, from, x, to, &scatter);
  VecScatterBegin(scatter, v1, x, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(scatter, v1, x, INSERT_VALUES, SCATTER_FORWARD);
  if (rank == 1)
  {
    ierr = VecView(x, PETSC_VIEWER_STDOUT_SELF);
  }
  ierr = VecGetArray(x, &x_local);

  /*
   * Finally We Clean The Mess
   */

  ierr = PetscFinalize();
  return 0;
}
