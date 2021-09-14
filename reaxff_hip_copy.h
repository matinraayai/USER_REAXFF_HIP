#ifndef __CUDA_COPY_H_
#define __CUDA_COPY_H_

#if defined(PURE_REAX)
    #include "../reax_types.h"
#elif defined(LAMMPS_REAX)
    #include "reaxff_types.h"
#endif


#ifdef __cplusplus
extern "C"  {
#endif

void Output_Sync_Forces(storage *, int );


void Hip_Copy_Atoms_Host_to_Device( reax_system * );

void Hip_Copy_Grid_Host_to_Device( grid *, grid * );

void Hip_Copy_System_Host_to_Device( reax_system * );

void Hip_Copy_List_Device_to_Host( reax_list *, reax_list *, int );

void Hip_Copy_Atoms_Device_to_Host( reax_system * );

void Hip_Copy_Simulation_Data_Device_to_Host( simulation_data *, simulation_data * );

void Hip_Copy_MPI_Data_Host_to_Device( mpi_datatypes * );

#ifdef __cplusplus
}
#endif


#endif
