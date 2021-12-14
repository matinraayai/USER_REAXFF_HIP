
#ifndef __HIP_POST_EVOLVE_H__
#define __HIP_POST_EVOLVE_H__

#if defined(LAMMPS_REAX)
    #include "reaxff_types.h"
#else
    #include "../reax_types.h"
#endif


#ifdef __cplusplus
extern "C" {
#endif

void Hip_Remove_CoM_Velocities( reax_system *, control_params *,
        simulation_data * );

#ifdef __cplusplus
}
#endif


#endif
