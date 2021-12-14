#include "hip/hip_runtime.h"
#if defined(LAMMPS_REAX)
    #include "reaxff_hip_dense_lin_alg.h"
    
    #include "reaxff_hip_reduction.h"
    #include "reaxff_hip_utils.h"
    
    #include "reaxff_comm_tools.h"
#else
    #include "hip_dense_lin_alg.h"
    
    #include "hip_reduction.h"
    #include "hip_utils.h"
    
    #include "../comm_tools.h"
#endif

#include <hipcub/hipcub.hpp>

/* sets all entries of a dense vector to zero
 *
 * inputs:
 *  v: dense vector
 *  k: number of entries in v
 * output: v with entries set to zero
 */
HIP_GLOBAL void k_vector_makezero( real * const v, unsigned int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    v[i] = ZERO;
}


/* copy the entries from one vector to another
 *
 * inputs:
 *  v: dense vector to copy
 *  k: number of entries in v
 * output:
 *  dest: vector copied into
 */
HIP_GLOBAL void k_vector_copy( real * const dest, real const * const v,
        unsigned int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    dest[i] = v[i];
}


/* copy the entries from one vector to another
 *
 * inputs:
 *  v: dense vector to copy
 *  k: number of entries in v
 * output:
 *  dest: vector copied into
 */
HIP_GLOBAL void k_vector_copy_rvec2( rvec2 * const dest, rvec2 const * const v,
        unsigned int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    dest[i][0] = v[i][0];
    dest[i][1] = v[i][1];
}


HIP_GLOBAL void k_vector_copy_from_rvec2( real * const dst, rvec2 const * const src,
        int index, int n )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    dst[i] = src[i][index];
}


HIP_GLOBAL void k_vector_copy_to_rvec2( rvec2 * const dst, real const * const src,
        int index, int n )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    dst[i][index] = src[i];
}


/* scales the entries of a dense vector by a constant
 *
 * inputs:
 *  c: scaling constant
 *  v: dense vector whose entries to scale
 *  k: number of entries in v
 * output:
 *  dest: with entries scaled
 */
HIP_GLOBAL void k_vector_scale( real * const dest, real c, real const * const v,
        unsigned int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    dest[i] = c * v[i];
}


/* computed the scaled sum of two dense vector and store
 * the result in a third vector (SAXPY operation in BLAS)
 *
 * inputs:
 *  c, d: scaling constants
 *  v, y: dense vector whose entries to scale
 *  k: number of entries in the vectors
 * output:
 *  dest: vector containing the scaled sum
 */
HIP_GLOBAL void k_vector_sum( real * const dest, real c, real const * const v,
        real d, real const * const y, unsigned int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    dest[i] = c * v[i] + d * y[i];
}


HIP_GLOBAL void k_vector_sum_rvec2( rvec2 * const dest, real c0, real c1, rvec2 const * const v,
        real d0, real d1, rvec2 const * const y, unsigned int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    dest[i][0] = c0 * v[i][0] + d0 * y[i][0];
    dest[i][1] = c1 * v[i][1] + d1 * y[i][1];
}


/* add the scaled sum of a dense vector to another vector
 * and store in-place
 *
 * inputs:
 *  c: scaling constant
 *  v: dense vector whose entries to scale
 *  k: number of entries in the vectors
 * output:
 *  dest: vector to accumulate with the scaled sum
 */
HIP_GLOBAL void k_vector_add( real * const dest, real c, real const * const v,
        unsigned int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    dest[i] += c * v[i];
}


/* add the scaled sum of a dense vector to another vector
 * and store in-place
 *
 * inputs:
 *  c: scaling constant
 *  v: dense vector whose entries to scale
 *  k: number of entries in the vectors
 * output:
 *  dest: vector to accumulate with the scaled sum
 */
HIP_GLOBAL void k_vector_add_rvec2( rvec2 * const dest, real c0, real c1, rvec2 const * const v,
        unsigned int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    dest[i][0] += c0 * v[i][0];
    dest[i][1] += c1 * v[i][1];
}


/* element-wise multiplication of a dense vector to another vector
 *
 * inputs:
 *  v1, v2: dense vectors whose entries to multiply
 *  k: number of entries in the vectors
 * output:
 *  dest: vector with the result of the multiplication
 */
HIP_GLOBAL void k_vector_mult( real * const dest, real const * const v1,
        real const * const v2, unsigned k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    dest[i] = v1[i] * v2[i];
}


/* element-wise multiplication of a dense vector to another vector
 *
 * inputs:
 *  v1, v2: dense vectors whose entries to multiply
 *  k: number of entries in the vectors
 * output:
 *  dest: vector with the result of the multiplication
 */
HIP_GLOBAL void k_vector_mult_rvec2( rvec2 * const dest, rvec2 const * const v1,
        rvec2 const * const v2, unsigned k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    dest[i][0] = v1[i][0] * v2[i][0];
    dest[i][1] = v1[i][1] * v2[i][1];
}


/* sets all entries of a dense vector to zero
 *
 * inputs:
 *  v: dense vector
 *  k: number of entries in v
 * output: v with entries set to zero
 */
void Vector_MakeZero( real * const v, unsigned int k, hipStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_makezero <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( v, k );
    hipCheckError( );
}


/* copy the entries from one vector to another
 *
 * inputs:
 *  v: dense vector to copy
 *  k: number of entries in v
 * output:
 *  dest: vector copied into
 */
void Vector_Copy( real * const dest, real const * const v,
        unsigned int k, hipStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_copy <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dest, v, k );
    hipCheckError( );
}


/* copy the entries from one vector to another
 *
 * inputs:
 *  v: dense vector to copy
 *  k: number of entries in v
 * output:
 *  dest: vector copied into
 */
void Vector_Copy_rvec2( rvec2 * const dest, rvec2 const * const v,
        unsigned int k, hipStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_copy_rvec2 <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dest, v, k );
    hipCheckError( );
}


void Vector_Copy_From_rvec2( real * const dst, rvec2 const * const src,
        int index, int k, hipStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_copy_from_rvec2 <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dst, src, index, k );
    hipCheckError( );
}


void Vector_Copy_To_rvec2( rvec2 * const dst, real const * const src,
        int index, int k, hipStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_copy_to_rvec2 <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dst, src, index, k );
    hipCheckError( );
}


/* scales the entries of a dense vector by a constant
 *
 * inputs:
 *  c: scaling constant
 *  v: dense vector whose entries to scale
 *  k: number of entries in v
 * output:
 *  dest: with entries scaled
 */
void Vector_Scale( real * const dest, real c, real const * const v,
        unsigned int k, hipStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_scale <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dest, c, v, k );
    hipCheckError( );
}


/* computed the scaled sum of two dense vector and store
 * the result in a third vector (SAXPY operation in BLAS)
 *
 * inputs:
 *  c, d: scaling constants
 *  v, y: dense vector whose entries to scale
 *  k: number of entries in the vectors
 * output:
 *  dest: vector containing the scaled sum
 */
void Vector_Sum( real * const dest, real c, real const * const v,
        real d, real const * const y, unsigned int k, hipStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_sum <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dest, c, v, d, y, k );
    hipCheckError( );
}


void Vector_Sum_rvec2( rvec2 * const dest, real c0, real c1, rvec2 const * const v,
        real d0, real d1, rvec2 const * const y, unsigned int k, hipStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_sum_rvec2 <<< blocks, DEF_BLOCK_SIZE, 0, s >>> 
        ( dest, c0, c1, v, d0, d1, y, k );
    hipCheckError( );
}


/* add the scaled sum of a dense vector to another vector
 * and store in-place
 *
 * inputs:
 *  c: scaling constant
 *  v: dense vector whose entries to scale
 *  k: number of entries in the vectors
 * output:
 *  dest: vector to accumulate with the scaled sum
 */
void Vector_Add( real * const dest, real c, real const * const v,
        unsigned int k, hipStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_add <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dest, c, v, k );
    hipCheckError( );
}


/* add the scaled sum of a dense vector to another vector
 * and store in-place
 *
 * inputs:
 *  c: scaling constant
 *  v: dense vector whose entries to scale
 *  k: number of entries in the vectors
 * output:
 *  dest: vector to accumulate with the scaled sum
 */
void Vector_Add_rvec2( rvec2 * const dest, real c0, real c1, rvec2 const * const v,
        unsigned int k, hipStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_add_rvec2 <<< blocks, DEF_BLOCK_SIZE, 0, s >>> 
        ( dest, c0, c1, v, k );
    hipCheckError( );
}


/* element-wise multiplication of a dense vector to another vector
 *
 * inputs:
 *  v1, v2: dense vectors whose entries to multiply
 *  k: number of entries in the vectors
 * output:
 *  dest: vector with the result of the multiplication
 */
void Vector_Mult( real * const dest, real const * const v1,
        real const * const v2, unsigned int k, hipStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_mult <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dest, v1, v2, k );
    hipCheckError( );
}


/* element-wise multiplication of a dense vector to another vector
 *
 * inputs:
 *  v1, v2: dense vectors whose entries to multiply
 *  k: number of entries in the vectors
 * output:
 *  dest: vector with the result of the multiplication
 */
void Vector_Mult_rvec2( rvec2 * const dest, rvec2 const * const v1,
        rvec2 const * const v2, unsigned int k, hipStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_mult_rvec2 <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dest, v1, v2, k );
    hipCheckError( );
}


/* compute the 2-norm (Euclidean) of a dense vector
 *
 * inputs:
 *  workspace: storage container for workspace structures
 *  v1: dense vector
 *  k: number of entries in the vector
 *  comm: MPI communicator
 *  s: HIP stream
 * output:
 *  norm: 2-norm
 */
real Norm( storage * const workspace,
        real const * const v1, unsigned int k, MPI_Comm comm, hipStream_t s )
{
    return SQRT( Dot( workspace, v1, v1, k, comm, s ) );
}


/* compute the inner product of two dense vectors
 *
 * inputs:
 *  workspace: storage container for workspace structures
 *  v1, v2: dense vectors
 *  k: number of entries in the vectors
 *  comm: MPI communicator
 * output:
 *  dot: inner product of the two vector
 */
real Dot( storage * const workspace,
        real const * const v1, real const * const v2,
        unsigned int k, MPI_Comm comm, hipStream_t s )
{
    int ret;
    real sum, *spad;
#if !defined(MPIX_HIP_AWARE_SUPPORT) || !MPIX_HIP_AWARE_SUPPORT
    real temp;
#endif

    sHipCheckMalloc( &workspace->scratch[4], &workspace->scratch_size[4],
            sizeof(real) * (k + 1), __FILE__, __LINE__ );
    spad = (real *) workspace->scratch[4];

    Vector_Mult( spad, v1, v2, k, s );

    /* local reduction (sum) on device */
    Hip_Reduction_Sum( spad, &spad[k], k, 4, s );

    /* global reduction (sum) of local device sums and store on host */
#if defined(MPIX_HIP_AWARE_SUPPORT) && MPIX_HIP_AWARE_SUPPORT
    hipStreamSynchronize( s );
    ret = MPI_Allreduce( &spad[k], &sum, 1, MPI_DOUBLE, MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
#else
    sHipMemcpyAsync( &temp, &spad[k], sizeof(real),
            hipMemcpyDeviceToHost, s, __FILE__, __LINE__ );

    hipStreamSynchronize( s );

    ret = MPI_Allreduce( &temp, &sum, 1, MPI_DOUBLE, MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
#endif

    return sum;
}


/* compute the local portions of the inner product of two dense vectors
 *
 * inputs:
 *  workspace: storage container for workspace structures
 *  v1, v2: dense vectors
 *  k: number of entries in the vectors
 * output:
 *  dot: inner product of the two vector
 */
real Dot_local( storage * const workspace,
        real const * const v1, real const * const v2,
        unsigned int k, hipStream_t s )
{
    real sum, *spad;

    sHipCheckMalloc( &workspace->scratch[4], &workspace->scratch_size[4],
            sizeof(real) * (k + 1), __FILE__, __LINE__ );
    spad = (real *) workspace->scratch[4];

    Vector_Mult( spad, v1, v2, k, s );

    /* local reduction (sum) on device */
    Hip_Reduction_Sum( spad, &spad[k], k, 4, s );

    //TODO: keep result of reduction on devie and pass directly to HIP-aware MPI
    sHipMemcpyAsync( &sum, &spad[k], sizeof(real),
            hipMemcpyDeviceToHost, s, __FILE__, __LINE__ );

    hipStreamSynchronize( s );

    return sum;
}


/* compute the local portions of the inner product of two dense vectors
 *
 * inputs:
 *  workspace: storage container for workspace structures
 *  v1, v2: dense vectors
 *  k: number of entries in the vectors
 * output:
 *  dot: inner product of the two vector
 */
void Dot_local_rvec2( storage * const workspace,
        rvec2 const * const v1, rvec2 const * const v2,
        unsigned int k, real * sum1, real * sum2, hipStream_t s )
{
    int blocks;
    size_t sz;
    rvec2 sum, *spad;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

#if !defined(HIP_ACCUM_ATOMIC)
    sz = sizeof(rvec2) * (k + blocks + 1);
#else
    sz = sizeof(rvec2) * (k + 1);
#endif

    sHipCheckMalloc( &workspace->scratch[4], &workspace->scratch_size[4],
            sz, __FILE__, __LINE__ );
    spad = (rvec2 *) workspace->scratch[4];

    Vector_Mult_rvec2( spad, v1, v2, k, s );

    /* local reduction (sum) on device */
//    Hip_Reduction_Sum( spad, &spad[k], k, 4, s );

#if defined(HIP_ACCUM_ATOMIC)
    sHipMemsetAsync( &spad[k], 0, sizeof(rvec2), s, __FILE__, __LINE__ );
#endif

    k_reduction_rvec2 <<< blocks, DEF_BLOCK_SIZE,
                      sizeof(hipcub::BlockReduce<double, DEF_BLOCK_SIZE>::TempStorage),
                      s >>>
        ( spad, &spad[k], k );
    hipCheckError( );

#if !defined(HIP_ACCUM_ATOMIC)
    k_reduction_rvec2 <<< 1, ((blocks + warpSize - 1) / warpSize) * warpSize,
                      sizeof(hipcub::BlockReduce<double, DEF_BLOCK_SIZE>::TempStorage),
                      s >>>
        ( &spad[k], &spad[k + blocks], blocks );
    hipCheckError( );
#endif

    //TODO: keep result of reduction on devie and pass directly to HIP-aware MPI
    sHipMemcpyAsync( &sum,
#if !defined(HIP_ACCUM_ATOMIC)
            &spad[k + blocks],
#else
            &spad[k],
#endif
            sizeof(rvec2), hipMemcpyDeviceToHost, s, __FILE__, __LINE__ );

    hipStreamSynchronize( s );

    *sum1 = sum[0];
    *sum2 = sum[1];
}
