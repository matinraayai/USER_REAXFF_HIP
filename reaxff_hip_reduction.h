
#ifndef __HIP_REDUCTION_H__
#define __HIP_REDUCTION_H__

#if defined(LAMMPS_REAX)
    #include "reaxff_types.h"
#else
    #include "../reax_types.h"
#endif


void Hip_Reduction_Sum( int *, int *, size_t, int, hipStream_t );

void Hip_Reduction_Sum( real *, real *, size_t, int, hipStream_t );

//void Hip_Reduction_Sum( rvec *, rvec *, size_t, hipStream_t );

void Hip_Scan_Excl_Sum( int *, int *, size_t, int, hipStream_t );

HIP_GLOBAL void k_reduction_rvec( rvec *, rvec *, size_t );

HIP_GLOBAL void k_reduction_rvec2( rvec2 *, rvec2 *, size_t );


#endif
