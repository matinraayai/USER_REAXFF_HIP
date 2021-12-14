
#ifndef __HIP_INIT_MD_H__
#define __HIP_INIT_MD_H__

#if defined(LAMMPS_REAX)
    #include "reaxff_types.h"
#else
    #include "../reax_types.h"
#endif


#ifdef __cplusplus
extern "C" {
#endif

void Hip_Initialize( reax_system*, control_params*, simulation_data*,
        storage*, reax_list**, output_controls*, mpi_datatypes* );

#ifdef __cplusplus
}
#endif


#endif
