#ifndef __CUDA_DENSE_LIN_ALG_H_
#define __CUDA_DENSE_LIN_ALG_H_

#if defined(PURE_REAX)
    #include "../reax_types.h"
#elif defined(LAMMPS_REAX)
    #include "reaxff_types.h"
#endif


int Vector_isZero( real const * const, unsigned int );

void Vector_MakeZero( real * const, unsigned int );

void Vector_Copy( real * const, real const * const,
        unsigned int );

void Vector_Copy_rvec2( rvec2 * const, rvec2 const * const,
        unsigned int );

void Vector_Copy_From_rvec2( real * const, rvec2 const * const,
        int, int );

void Vector_Copy_To_rvec2( rvec2 * const, real const * const,
        int, int );

void Vector_Scale( real * const, real, real const * const,
        unsigned int );

void Vector_Sum( real * const, real, real const * const,
        real, real const * const, unsigned int );

void Vector_Sum_rvec2( rvec2 * const, real, real, rvec2 const * const,
        real, real, rvec2 const * const, unsigned int );

void Vector_Add( real * const, real, real const * const,
        unsigned int );

void Vector_Add_rvec2( rvec2 * const, real, real, rvec2 const * const,
        unsigned int );

void Vector_Mult( real * const, real const * const,
        real const * const, unsigned int );

void Vector_Mult_rvec2( rvec2 * const, rvec2 const * const,
        rvec2 const * const, unsigned int );

real Norm( storage * const,
        real const * const, unsigned int, MPI_Comm );

real Dot( storage * const,
        real const * const, real const * const,
        unsigned int, MPI_Comm );

real Dot_local( storage * const,
        real const * const, real const * const,
        unsigned int );

void Dot_local_rvec2( control_params const * const,
        storage * const,
        rvec2 const * const, rvec2 const * const,
        unsigned int, real *, real * );

HIP_GLOBAL void k_vector_sum( real * const dest, real c, real const * const v,
                              real d, real const * const y, unsigned int k );

HIP_GLOBAL void k_vector_mult( real * const dest, real const * const v1,
                               real const * const v2, unsigned k );

#endif
