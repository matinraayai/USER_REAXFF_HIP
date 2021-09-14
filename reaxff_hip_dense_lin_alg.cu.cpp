#include "hip/hip_runtime.h"
#if defined(PURE_REAX)
    #include "hip_dense_lin_alg.h"

    #include "hip_reduction.h"
    #include "hip_utils.h"

    #include "../comm_tools.h"
#elif defined(LAMMPS_REAX)
    #include "reaxff_hip_dense_lin_alg.h"

    #include "reaxff_hip_reduction.h"
    #include "reaxff_hip_utils.h"

    #include "reaxff_comm_tools.h"
#endif


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


/* check if all entries of a dense vector are sufficiently close to zero
 *
 * inputs:
 *  v: dense vector
 *  k: number of entries in v
 * output: TRUE if all entries are sufficiently close to zero, FALSE otherwise
 */
int Vector_isZero( real const * const v, unsigned int k )
{
    unsigned int i, ret;

    ret = TRUE;

    for ( i = 0; i < k; ++i )
    {
        if ( FABS( v[i] ) > ALMOST_ZERO )
        {
            ret = FALSE;
        }
    }

    return ret;
}


/* sets all entries of a dense vector to zero
 *
 * inputs:
 *  v: dense vector
 *  k: number of entries in v
 * output: v with entries set to zero
 */
void Vector_MakeZero( real * const v, unsigned int k )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    hipLaunchKernelGGL(k_vector_makezero, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  v, k );
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
        unsigned int k )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    hipLaunchKernelGGL(k_vector_copy, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  dest, v, k );
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
        unsigned int k )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    hipLaunchKernelGGL(k_vector_copy_rvec2, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  dest, v, k );
    hipCheckError( );
}


void Vector_Copy_From_rvec2( real * const dst, rvec2 const * const src,
        int index, int k )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    hipLaunchKernelGGL(k_vector_copy_from_rvec2, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  dst, src, index, k );
    hipCheckError( );
}


void Vector_Copy_To_rvec2( rvec2 * const dst, real const * const src,
        int index, int k )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    hipLaunchKernelGGL(k_vector_copy_to_rvec2, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  dst, src, index, k );
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
        unsigned int k )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    hipLaunchKernelGGL(k_vector_scale, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  dest, c, v, k );
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
        real d, real const * const y, unsigned int k )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    hipLaunchKernelGGL(k_vector_sum, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  dest, c, v, d, y, k );
    hipCheckError( );
}


void Vector_Sum_rvec2( rvec2 * const dest, real c0, real c1, rvec2 const * const v,
        real d0, real d1, rvec2 const * const y, unsigned int k )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    hipLaunchKernelGGL(k_vector_sum_rvec2, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  dest, c0, c1, v, d0, d1, y, k );
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
        unsigned int k )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    hipLaunchKernelGGL(k_vector_add, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  dest, c, v, k );
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
        unsigned int k )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    hipLaunchKernelGGL(k_vector_add_rvec2, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  dest, c0, c1, v, k );
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
        real const * const v2, unsigned int k )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    hipLaunchKernelGGL(k_vector_mult, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  dest, v1, v2, k );
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
        rvec2 const * const v2, unsigned int k )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    hipLaunchKernelGGL(k_vector_mult_rvec2, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  dest, v1, v2, k );
    hipCheckError( );
}


/* compute the 2-norm (Euclidean) of a dense vector
 *
 * inputs:
 *  workspace: storage container for workspace structures
 *  v1: dense vector
 *  k: number of entries in the vector
 *  comm: MPI communicator
 * output:
 *  norm: 2-norm
 */
real Norm( storage * const workspace,
        real const * const v1, unsigned int k, MPI_Comm comm )
{
    return SQRT( Dot( workspace, v1, v1, k, comm ) );
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
        unsigned int k, MPI_Comm comm )
{
    int ret;
    real sum, *spad;
//#if !defined(MPIX_CUDA_AWARE_SUPPORT) || !MPIX_CUDA_AWARE_SUPPORT
    real temp;
//#endif

    hip_check_malloc( &workspace->scratch, &workspace->scratch_size,
            sizeof(real) * (k + 1), "Dot::workspace->scratch" );
    spad = (real *) workspace->scratch;

    Vector_Mult( spad, v1, v2, k );

    /* local reduction (sum) on device */
    Hip_Reduction_Sum( spad, &spad[k], k );

    /* global reduction (sum) of local device sums and store on host */
//#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
//    ret = MPI_Allreduce( &spad[k], &sum, 1, MPI_DOUBLE, MPI_SUM, comm );
//    Check_MPI_Error( ret, __FILE__, __LINE__ );
//#else
    sHipMemcpy( &temp, &spad[k], sizeof(real),
            hipMemcpyDeviceToHost, __FILE__, __LINE__ );

    ret = MPI_Allreduce( &temp, &sum, 1, MPI_DOUBLE, MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
//#endif

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
        unsigned int k )
{
    real sum, *spad;

    hip_check_malloc( &workspace->scratch, &workspace->scratch_size,
            sizeof(real) * (k + 1), "Dot_local::workspace->scratch" );
    spad = (real *) workspace->scratch;

    Vector_Mult( spad, v1, v2, k );

    /* local reduction (sum) on device */
    Hip_Reduction_Sum( spad, &spad[k], k );

    //TODO: keep result of reduction on devie and pass directly to CUDA-aware MPI
    sHipMemcpy( &sum, &spad[k], sizeof(real),
            hipMemcpyDeviceToHost, __FILE__, __LINE__ );

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
void Dot_local_rvec2( control_params const * const control,
        storage * const workspace,
        rvec2 const * const v1, rvec2 const * const v2,
        unsigned int k, real * sum1, real * sum2 )
{
    int blocks;
    rvec2 sum, *spad;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    hip_check_malloc( &workspace->scratch, &workspace->scratch_size,
            sizeof(rvec2) * (k + blocks + 1), "Dot_local_rvec2::workspace->scratch" );
    spad = (rvec2 *) workspace->scratch;

    Vector_Mult_rvec2( spad, v1, v2, k );

    /* local reduction (sum) on device */
//    Hip_Reduction_Sum( spad, &spad[k], k );
    hipLaunchKernelGGL(k_reduction_rvec2, dim3(blocks), dim3(DEF_BLOCK_SIZE), sizeof(rvec2) * (DEF_BLOCK_SIZE / warpSize) , 0,
                        spad, &spad[k], k );
    hipCheckError( );

    hipLaunchKernelGGL(k_reduction_rvec2, dim3(1), dim3(((blocks + warpSize - 1) / warpSize) * warpSize),
                       sizeof(rvec2) * ((blocks + warpSize - 1) / warpSize) , 0,  &spad[k], &spad[k + blocks], blocks );
    hipCheckError( );



   //TODO: keep result of reduction on devie and pass directly to CUDA-aware MPI
    sHipMemcpy( &sum, &spad[k + blocks], sizeof(rvec2),
            hipMemcpyDeviceToHost, __FILE__, __LINE__ );

    *sum1 = sum[0];
    *sum2 = sum[1];


}
