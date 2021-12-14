
#if defined(LAMMPS_REAX)
    #include "reaxff_hip_reduction.h"

    #include "reaxff_hip_utils.h"

    #include "reaxff_vector.h"
#else
    #include "hip_reduction.h"

    #include "hip_utils.h"

    #include "../vector.h"
#endif

#include "hip/hip_runtime.h"
#include <hipcub/hipcub.hpp>



//struct RvecSum
//{
//    template <typename T>
//    __device__ __forceinline__
//    T operator()(const T &a, const T &b) const
//    {
//        T c;
//        return c {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
//    }
//};


/* Perform a device-wide reduction (sum operation)
 *
 * d_array: device array to reduce
 * d_dest: device pointer to hold result of reduction */
void Hip_Reduction_Sum( int *d_array, int *d_dest, size_t n, int s_index, hipStream_t s )
{
    static void *d_temp_storage[MAX_HIP_STREAMS] = { NULL };
    static size_t temp_storage_bytes[MAX_HIP_STREAMS] = { 0 };
    void *temp = NULL;
    size_t temp_bytes = 0;

    /* determine temporary device storage requirements */
    hipcub::DeviceReduce::Sum( temp, temp_bytes, d_array, d_dest, n, s );
    hipCheckError( );

    /* allocate temporary storage */
    sHipCheckMalloc( &d_temp_storage[s_index], &temp_storage_bytes[s_index],
            temp_bytes, __FILE__, __LINE__ );

    /* run sum-reduction */
    hipcub::DeviceReduce::Sum( d_temp_storage[s_index], temp_storage_bytes[s_index],
            d_array, d_dest, n, s );
    hipCheckError( );
}


/* Perform a device-wide reduction (sum operation)
 *
 * d_array: device array to reduce
 * d_dest: device pointer to hold result of reduction */
void Hip_Reduction_Sum( real *d_array, real *d_dest, size_t n, int s_index, hipStream_t s )
{
    static void *d_temp_storage[MAX_HIP_STREAMS] = { NULL };
    static size_t temp_storage_bytes[MAX_HIP_STREAMS] = { 0 };
    void *temp = NULL;
    size_t temp_bytes = 0;

    /* determine temporary device storage requirements */
    hipcub::DeviceReduce::Sum( temp, temp_bytes, d_array, d_dest, n, s );
    hipCheckError( );

    /* allocate temporary storage */
    sHipCheckMalloc( &d_temp_storage[s_index], &temp_storage_bytes[s_index],
            temp_bytes, __FILE__, __LINE__ );

    /* run sum-reduction */
    hipcub::DeviceReduce::Sum( d_temp_storage[s_index], temp_storage_bytes[s_index],
            d_array, d_dest, n, s );
    hipCheckError( );
}


///* Perform a device-wide reduction (sum operation)
// *
// * d_array: device array to reduce
// * d_dest: device pointer to hold result of reduction */
//void Hip_Reduction_Sum( rvec *d_array, rvec *d_dest, size_t n, hipStream_t s )
//{
//    void *d_temp_storage = NULL;
//    size_t temp_storage_bytes = 0;
//    RvecSum sum_op;
//    rvec init = {0.0, 0.0, 0.0};
//
//    /* determine temporary device storage requirements */
//    cub::DeviceReduce::Reduce( d_temp_storage, temp_storage_bytes,
//            d_array, d_dest, n, sum_op, init, s );
//    hipCheckError( );
//
//    /* allocate temporary storage */
//    sHipMalloc( &d_temp_storage, temp_storage_bytes, __FILE__, __LINE__ );
//
//    /* run sum-reduction */
//    cub::DeviceReduce::Reduce( d_temp_storage, temp_storage_bytes,
//            d_array, d_dest, n, sum_op, init, s );
//    hipCheckError( );
//
//    /* deallocate temporary storage */
//    sHipFree( d_temp_storage, __FILE__, __LINE__ );
//}


/* Perform a device-wide scan (partial sum operation)
 *
 * d_src: device array to scan
 * d_dest: device array to hold result of scan */
void Hip_Scan_Excl_Sum( int *d_src, int *d_dest, size_t n, int s_index, hipStream_t s )
{
    static void *d_temp_storage[MAX_HIP_STREAMS] = { NULL };
    static size_t temp_storage_bytes[MAX_HIP_STREAMS] = { 0 };
    void *temp = NULL;
    size_t temp_bytes = 0;

    /* determine temporary device storage requirements */
    hipcub::DeviceScan::ExclusiveSum( temp, temp_bytes, d_src, d_dest, n, s );
    hipCheckError( );

    /* allocate temporary storage */
    sHipCheckMalloc( &d_temp_storage[s_index], &temp_storage_bytes[s_index],
            temp_bytes, __FILE__, __LINE__ );

    /* run exclusive prefix sum */
    hipcub::DeviceScan::ExclusiveSum( d_temp_storage[s_index], temp_storage_bytes[s_index],
            d_src, d_dest, n, s );
    hipCheckError( );
}


/* Performs a device-wide partial reduction (sum) on input in 2 stages:
 *  1) Perform a warp-level sum of parts of input assigned to warps
 *  2) Perform an block-level sum of the warp-local partial sums
 * The block-level sums are written to global memory pointed to by results
 *  in accordance to their block IDs.
 */
HIP_GLOBAL void k_reduction_rvec( rvec *input, rvec *results, size_t n )
{
    extern __shared__ rvec data_s[];
    rvec data;
    unsigned int i;
    int offset;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) {
        rvec_Copy( data, input[i] );
    }
    else {
        data[0] = 0.f;
        data[1] = 0.f;
        data[2] = 0.f;
    }

    /* warp-level sum using registers within a warp */
    for ( offset = warpSize >> 1; offset > 0; offset >>= 1 )
    {
        data[0] += __shfl_down(data[0], offset );
        data[1] += __shfl_down(data[1], offset );
        data[2] += __shfl_down(data[2], offset );
    }

    /* first thread within a warp writes warp-level sum to shared memory */
    if ( threadIdx.x % warpSize == 0 )
    {
        rvec_Copy( data_s[threadIdx.x / warpSize], data );
    }
    __syncthreads( );

    /* block-level sum using shared memory */
    for ( offset = blockDim.x / (warpSize << 1); offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            rvec_Add( data_s[threadIdx.x], data_s[threadIdx.x + offset] );
        }

        __syncthreads( );
    }

    /* one thread writes the block-level partial sum
     * of the reduction back to global memory */
    if ( threadIdx.x == 0 )
    {
        rvec_Copy( results[blockIdx.x], data_s[0] );
    }
}


HIP_GLOBAL void k_reduction_rvec2( rvec2 *input, rvec2 *results, size_t n )
{
    extern __shared__ hipcub::BlockReduce<double, DEF_BLOCK_SIZE>::TempStorage temp_block[];
    unsigned int i;
    rvec2 data;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < n )
    {
        data[0] = input[i][0];
        data[1] = input[i][1];
    }
    else
    {
        data[0] = 0.0;
        data[1] = 0.0;
    }

    data[0] = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(data[0]);
    __syncthreads( );
    data[1] = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(data[1]);

    /* one thread writes the block-level partial sum
     * of the reduction back to global memory */
    if ( threadIdx.x == 0 )
    {
#if !defined(HIP_ACCUM_ATOMIC)
        results[blockIdx.x][0] = data[0];
        results[blockIdx.x][1] = data[1];
#else
        atomicAdd( (double *) &results[0][0], (double) data[0] );
        atomicAdd( (double *) &results[0][1], (double) data[1] );
#endif
    }

}
