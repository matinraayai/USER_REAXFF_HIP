
#ifndef __HIP_LOOKUP_H__
#define __HIP_LOOKUP_H__

#if defined(LAMMPS_REAX)
    #include "reaxff_types.h"
#else
    #include "../reax_types.h"
#endif


#ifdef __cplusplus
extern "C" {
#endif

void Hip_Copy_LR_Lookup_Table_Host_to_Device( reax_system *, control_params *,
        storage *, int * );

#ifdef __cplusplus
}
#endif


#endif
