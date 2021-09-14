
#ifndef __CUDA_SYSTEM_PROPS_H__
#define __CUDA_SYSTEM_PROPS_H__

#include "reaxff_types.h"


void Hip_Generate_Initial_Velocities( reax_system *, control_params *, real );

void Hip_Compute_Total_Mass( reax_system *, control_params *,
        storage *, simulation_data *, MPI_Comm );

void Hip_Compute_Pressure( reax_system *, control_params *,
        storage *, simulation_data *, mpi_datatypes * );

#ifdef __cplusplus
extern "C"  {
#endif

void Hip_Compute_Kinetic_Energy( reax_system *, control_params *,
        storage *, simulation_data *, MPI_Comm );

void Hip_Compute_Center_of_Mass( reax_system *, control_params *,
        storage *, simulation_data *, mpi_datatypes *, MPI_Comm );

#ifdef __cplusplus
}
#endif


#endif
