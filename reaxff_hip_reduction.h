
#ifndef __CUDA_REDUCTION_H__
#define __CUDA_REDUCTION_H__

#include "reaxff_types.h"


void Hip_Reduction_Sum( int *, int *, size_t );

void Hip_Reduction_Sum( real *, real *, size_t );

//void Hip_Reduction_Sum( rvec *, rvec *, size_t );

void Hip_Reduction_Max( int *, int *, size_t );

void Hip_Scan_Excl_Sum( int *, int *, size_t );

HIP_GLOBAL void k_reduction_rvec( rvec *, rvec *, size_t );

HIP_GLOBAL void k_reduction_rvec2( rvec2 *, rvec2 *, size_t );


#endif
