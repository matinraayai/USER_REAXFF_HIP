#ifndef __CUDA_ALLOCATE_H_
#define __CUDA_ALLOCATE_H_

#if defined(PURE_REAX)
    #include "../reax_types.h"
#elif defined(LAMMPS_REAX)
    #include "reaxff_types.h"
#endif

void Hip_Allocate_Atoms(reax_system *system);


void Hip_Allocate_System( reax_system * );

void Hip_Allocate_Grid( reax_system * );

void Hip_Allocate_Simulation_Data( simulation_data * );

void Hip_Allocate_Control( control_params * );

void Hip_Allocate_Workspace_Part1( reax_system *, control_params *, storage *, int );

void Hip_Allocate_Workspace_Part2( reax_system *, control_params *, storage *, int );

void Hip_Allocate_Matrix( sparse_matrix * const, int, int, int, int );

void Hip_Deallocate_Grid_Cell_Atoms( reax_system * );

void Hip_Allocate_Grid_Cell_Atoms( reax_system *, int );

void Hip_Deallocate_Workspace_Part1( control_params *, storage * );

void Hip_Deallocate_Workspace_Part2( control_params *, storage * );

void Hip_Deallocate_Matrix( sparse_matrix * );

void Hip_Reallocate_Part1( reax_system *, control_params *, simulation_data *, storage *,
        reax_list **, mpi_datatypes * );

void Hip_Reallocate_Part2( reax_system *, control_params *, simulation_data *, storage *,
        reax_list **, mpi_datatypes * );


#endif