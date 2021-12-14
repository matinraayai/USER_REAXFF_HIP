#ifndef __HIP_DENSE_LIN_ALG_H_
#define __HIP_DENSE_LIN_ALG_H_

#if defined(LAMMPS_REAX)
    #include "reaxff_types.h"
#else
    #include "../reax_types.h"
#endif


void Vector_MakeZero( real * const, unsigned int,
        hipStream_t );

void Vector_Copy( real * const, real const * const,
        unsigned int, hipStream_t );

void Vector_Copy_rvec2( rvec2 * const, rvec2 const * const,
        unsigned int, hipStream_t );

void Vector_Copy_From_rvec2( real * const, rvec2 const * const,
        int, int, hipStream_t );

void Vector_Copy_To_rvec2( rvec2 * const, real const * const,
        int, int, hipStream_t );

void Vector_Scale( real * const, real, real const * const,
        unsigned int, hipStream_t );

void Vector_Sum( real * const, real, real const * const,
        real, real const * const, unsigned int, hipStream_t );

void Vector_Sum_rvec2( rvec2 * const, real, real, rvec2 const * const,
        real, real, rvec2 const * const, unsigned int, hipStream_t );

void Vector_Add( real * const, real, real const * const,
        unsigned int, hipStream_t );

void Vector_Add_rvec2( rvec2 * const, real, real, rvec2 const * const,
        unsigned int, hipStream_t );

void Vector_Mult( real * const, real const * const,
        real const * const, unsigned int, hipStream_t );

void Vector_Mult_rvec2( rvec2 * const, rvec2 const * const,
        rvec2 const * const, unsigned int, hipStream_t );

real Norm( storage * const, real const * const, unsigned int, MPI_Comm, hipStream_t );

real Dot( storage * const, real const * const, real const * const,
        unsigned int, MPI_Comm, hipStream_t );

real Dot_local( storage * const, real const * const, real const * const,
        unsigned int, hipStream_t );

void Dot_local_rvec2( storage * const, rvec2 const * const, rvec2 const * const,
        unsigned int, real *, real *, hipStream_t );


#endif
