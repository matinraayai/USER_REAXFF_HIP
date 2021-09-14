
#ifndef __CUDA_ENVIRONMENT_H__
#define __CUDA_ENVIRONMENT_H__

#if defined(PURE_REAX)
    #include "../reax_types.h"
#elif defined(LAMMPS_REAX)
    #include "reaxff_types.h"
#endif

#ifdef __cplusplus
extern "C"  {
#endif

void Hip_Setup_Environment( int, int, int );

void Hip_Init_Block_Sizes( reax_system *, control_params * );

void Hip_Cleanup_Environment( );

#ifdef __cplusplus
}
#endif


#endif
