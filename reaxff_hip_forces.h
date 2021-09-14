
#ifndef __CUDA_FORCES_H__
#define __CUDA_FORCES_H__

#if defined(PURE_REAX)
    #include "reax_types.h"
#elif defined(LAMMPS_REAX)
    #include "reaxff_types.h"
#endif


void Hip_Init_Neighbor_Indices( reax_system *, reax_list * );

void Hip_Init_HBond_Indices( reax_system *, storage *,
        reax_list * );

void Hip_Init_Bond_Indices( reax_system *, reax_list * );

void Hip_Init_Sparse_Matrix_Indices( reax_system *, sparse_matrix * );

void Hip_Init_Three_Body_Indices( int *, int, reax_list ** );

void Hip_Estimate_Storages( reax_system *, control_params *, storage *,
        reax_list **, int, int, int, int );

int Hip_Init_Forces( reax_system *, control_params *, simulation_data *,
        storage *, reax_list **, output_controls * );

int Hip_Init_Forces_No_Charges( reax_system *, control_params *, simulation_data *,
        storage *, reax_list **, output_controls * );

int Hip_Compute_Bonded_Forces( reax_system *, control_params *, simulation_data *,
        storage *, reax_list **, output_controls * );

void Hip_Compute_NonBonded_Forces( reax_system *, control_params *,
        simulation_data *, storage *, reax_list **, output_controls *,
        mpi_datatypes * );

#ifdef __cplusplus
extern "C" {
#endif

int Hip_Compute_Forces( reax_system*, control_params*, simulation_data*,
        storage*, reax_list**, output_controls*, mpi_datatypes* );

HIP_GLOBAL void k_init_dist( reax_atom *my_atoms, reax_list far_nbr_list, int N );

HIP_GLOBAL void k_estimate_storages_cm_full( control_params *control,
                                             reax_list far_nbr_list, int cm_n, int cm_n_max,
                                             int *cm_entries, int *max_cm_entries );

#ifdef __cplusplus
}
#endif


#endif
