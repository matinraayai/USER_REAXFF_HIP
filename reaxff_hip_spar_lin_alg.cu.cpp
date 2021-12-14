/*----------------------------------------------------------------------
  PuReMD - Purdue ReaxFF Molecular Dynamics Program

  Copyright (2010) Purdue University
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
  Joseph Fogarty, jcfogart@mail.usf.edu
  Sagar Pandit, pandit@usf.edu
  Ananth Y Grama, ayg@cs.purdue.edu

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License as
  published by the Free Software Foundation; either version 2 of 
  the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
  See the GNU General Public License for more details:
  <http://www.gnu.org/licenses/>.
  ----------------------------------------------------------------------*/
#include "hip/hip_runtime.h"
#if defined(LAMMPS_REAX)
    #include "reaxff_hip_spar_lin_alg.h"

    #if defined(HIP_DEVICE_PACK)
      #include "reaxff_hip_basic_comm.h"
    #endif
    #include "reaxff_hip_dense_lin_alg.h"
    #include "reaxff_hip_helpers.h"
    #include "reaxff_hip_utils.h"
    #include "reaxff_hip_reduction.h"

    #if !defined(HIP_DEVICE_PACK)
      #include "reaxff_basic_comm.h"
    #endif
    #include "reaxff_comm_tools.h"
    #include "reaxff_tool_box.h"
#else
    #include "hip_spar_lin_alg.h"

    #if defined(HIP_DEVICE_PACK)
      #include "hip_basic_comm.h"
    #endif
    #include "hip_dense_lin_alg.h"
    #include "hip_helpers.h"
    #include "hip_utils.h"
    #include "hip_reduction.h"

    #if !defined(HIP_DEVICE_PACK)
      #include "../basic_comm.h"
    #endif
    #include "../comm_tools.h"
    #include "../tool_box.h"
#endif


enum preconditioner_type
{
    LEFT = 0,
    RIGHT = 1,
};


/* Jacobi preconditioner computation */
HIP_GLOBAL void k_jacobi_cm_half( int *row_ptr_start,
        int *row_ptr_end, int *col_ind, real *vals,
        real * const Hdia_inv, int N )
{
    int i;
    real diag;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    if ( FABS( vals[row_ptr_end[i]] ) >= 1.0e-12 )
    {
        diag = 1.0 / vals[row_ptr_end[i]];
    }
    else
    {
        diag = 1.0;
    }

    Hdia_inv[i] = diag;
}


/* Jacobi preconditioner computation */
HIP_GLOBAL void k_jacobi_cm_full( int *row_ptr_start,
        int *row_ptr_end, int *col_ind, real *vals,
        real * const Hdia_inv, int N )
{
    int i, pj;
    real diag;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    for ( pj = row_ptr_start[i]; pj < row_ptr_end[i]; ++pj )
    {
        if ( col_ind[pj] == i )
        {
            if ( FABS( vals[pj] ) >= 1.0e-12 )
            {
                diag = 1.0 / vals[pj];
            }
            else
            {
                diag = 1.0;
            }

            break;
        }
    }

    __syncthreads( );

    Hdia_inv[i] = diag;
}


HIP_GLOBAL void k_dual_jacobi_apply( real const * const Hdia_inv, rvec2 const * const y,
        rvec2 * const x, int n )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    x[i][0] = Hdia_inv[i] * y[i][0];
    x[i][1] = Hdia_inv[i] * y[i][1];
}


HIP_GLOBAL void k_jacobi_apply( real const * const Hdia_inv, real const * const y,
        real * const x, int n )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    x[i] = Hdia_inv[i] * y[i];
}


/* sparse matrix, dense vector multiplication Ax = b,
 * where one GPU thread multiplies a row
 *
 * A: symmetric (upper triangular portion only stored) matrix,
 *    stored in CSR format
 * x: dense vector, size equal to num. columns in A
 * b (output): dense vector, size equal to num. columns in A
 * N: number of rows in A */
HIP_GLOBAL void k_sparse_matvec_half_csr( int *row_ptr_start,
        int *row_ptr_end, int *col_ind, real *vals,
        const real * const x, real * const b, int N )
{
    int i, pj, si, ei;
    real sum;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    si = row_ptr_start[i];
    ei = row_ptr_end[i];

    /* A symmetric, upper triangular portion stored
     * => diagonal only contributes once */
    sum = vals[si] * x[i];

    for ( pj = si + 1; pj < ei; ++pj )
    {
        sum += vals[pj] * x[col_ind[pj]];
        /* symmetric contribution to row j */
        atomicAdd( (double *) &b[col_ind[pj]], (double) (vals[pj] * x[i]) );
    }

    /* local contribution to row i for this thread */
    atomicAdd( (double *) &b[i], (double) sum );
}


/* sparse matrix, dense vector multiplication Ax = b,
 * where warps collaborate to multiply each row
 *
 * A: symmetric (upper triangular portion only stored) matrix,
 *    stored in CSR format
 * x: dense vector, size equal to num. columns in A
 * b (output): dense vector, size equal to num. columns in A
 * N: number of rows in A */
HIP_GLOBAL void k_sparse_matvec_half_opt_csr( int *row_ptr_start,
        int *row_ptr_end, int *col_ind, real *vals,
        const real * const x, real * const b, int N )
{
    int pj, si, ei, thread_id, warp_id, lane_id, offset, itr, col_ind_l;
    real vals_l, sum;

    thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    warp_id = thread_id / warpSize;

    if ( warp_id >= N )
    {
        return;
    }

    lane_id = thread_id % warpSize;
    si = row_ptr_start[warp_id];
    ei = row_ptr_end[warp_id];
    sum = 0.0;

    /* partial sums per thread */
    for ( itr = 0, pj = si + lane_id; itr < (ei - si + warpSize - 1) / warpSize; ++itr )
    {
        /* coaleseced 128-bit aligned reads from global memory */
        vals_l = vals[pj];
        col_ind_l = col_ind[pj];

        /* only threads with value non-zero positions accumulate the result */
        if ( pj < ei )
        {
            /* gather on x from global memory and compute partial sum for this non-zero entry */
            sum += vals_l * x[col_ind_l];

            /* A symmetric, upper triangular portion stored
             * => diagonal only contributes once */
            if ( pj > si )
            {
                /* symmetric contribution to row j */
                atomicAdd( (double *) &b[col_ind[pj]], (double) (vals_l * x[warp_id]) );
            }
        }

        pj += warpSize;
    }

    /* warp-level reduction of partial sums
     * using registers within a warp */
    for ( offset = warpSize >> 1; offset > 0; offset >>= 1 )
    {
        sum += __shfl_down(sum, offset );
    }

    /* local contribution to row i for this warp */
    if ( lane_id == 0 )
    {
        atomicAdd( (double *) &b[warp_id], (double) sum );
    }
}


/* sparse matrix, dense vector multiplication Ax = b,
 * where one GPU thread multiplies a row
 *
 * A: symmetric matrix,
 *    stored in CSR format
 * x: dense vector, size equal to num. columns in A
 * b (output): dense vector, size equal to num. columns in A
 * N: number of rows in A */
HIP_GLOBAL void k_sparse_matvec_full_csr( int *row_ptr_start,
        int *row_ptr_end, int *col_ind, real *vals,
        const real * const x, real * const b, int n )
{
    int i, pj, si, ei;
    real sum;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    sum = 0.0;
    si = row_ptr_start[i];
    ei = row_ptr_end[i];

    for ( pj = si; pj < ei; ++pj )
    {
        sum += vals[pj] * x[col_ind[pj]];
    }

    __syncthreads( );

    b[i] = sum;
}


/* sparse matrix, dense vector multiplication Ax = b,
 * where warps collaborate to multiply each row
 *
 * A: symmetric matrix,
 *    stored in CSR format
 * x: dense vector, size equal to num. columns in A
 * b (output): dense vector, size equal to num. columns in A
 * N: number of rows in A */
HIP_GLOBAL void k_sparse_matvec_full_opt_csr( int *row_ptr_start,
        int *row_ptr_end, int *col_ind, real *vals,
        const real * const x, real * const b, int n )
{
    int pj, si, ei, thread_id, warp_id, lane_id, offset, itr, col_ind_l;
    real vals_l, sum;

    thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    warp_id = thread_id / warpSize;

    if ( warp_id >= n )
    {
        return;
    }

    lane_id = thread_id % warpSize;
    si = row_ptr_start[warp_id];
    ei = row_ptr_end[warp_id];
    sum = 0.0;

    /* partial sums per thread */
    for ( itr = 0, pj = si + lane_id; itr < (ei - si + warpSize - 1) / warpSize; ++itr )
    {
        /* coalesced 128-bit aligned reads from global memory */
        vals_l = vals[pj];
        col_ind_l = col_ind[pj];

        /* only threads with value non-zero positions accumulate the result */
        if ( pj < ei )
        {
            /* gather on x from global memory and compute partial sum for this non-zero entry */
            sum += vals_l * x[col_ind_l];
        }

        pj += warpSize;
    }

    /* warp-level reduction of partial sums
     * using registers within a warp */
    for ( offset = warpSize >> 1; offset > 0; offset >>= 1 )
    {
        sum += __shfl_down(sum, offset );
    }

    __syncthreads( );

    /* first thread within a warp writes sum to global memory */
    if ( lane_id == 0 )
    {
        b[warp_id] = sum;
    }
}


/* sparse matrix, dense vector multiplication Ax = b,
 * where one GPU thread multiplies a row
 *
 * A: symmetric (upper triangular portion only stored) matrix,
 *    stored in CSR format
 * X: 2 dense vectors, size equal to num. columns in A
 * B (output): 2 dense vectors, size equal to num. columns in A
 * N: number of rows in A */
HIP_GLOBAL void k_dual_sparse_matvec_half_csr( int *row_ptr_start,
        int *row_ptr_end, int *col_ind, real *vals,
        const rvec2 * const x, rvec2 * const b, int N )
{
    int i, pj, si, ei;
    rvec2 sum;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    si = row_ptr_start[i];
    ei = row_ptr_end[i];

    /* A symmetric, upper triangular portion stored
     * => diagonal only contributes once */
    sum[0] = vals[si] * x[i][0];
    sum[1] = vals[si] * x[i][1];

    for ( pj = si + 1; pj < ei; ++pj )
    {
        sum[0] += vals[pj] * x[col_ind[pj]][0];
        sum[1] += vals[pj] * x[col_ind[pj]][1];
        /* symmetric contribution to row j */
        atomicAdd( (double *) &b[col_ind[pj]][0], (double) (vals[pj] * x[i][0]) );
        atomicAdd( (double *) &b[col_ind[pj]][1], (double) (vals[pj] * x[i][1]) );
    }

    /* local contribution to row i for this thread */
    atomicAdd( (double *) &b[i][0], (double) sum[0] );
    atomicAdd( (double *) &b[i][1], (double) sum[1] );
}


/* sparse matrix, dense vector multiplication Ax = b,
 * where warps collaborate to multiply each row
 *
 * A: symmetric (upper triangular portion only stored) matrix,
 *    stored in CSR format
 * X: 2 dense vectors, size equal to num. columns in A
 * B (output): 2 dense vectors, size equal to num. columns in A
 * N: number of rows in A */
HIP_GLOBAL void k_dual_sparse_matvec_half_opt_csr( int *row_ptr_start,
        int *row_ptr_end, int *col_ind, real *vals,
        const rvec2 * const x, rvec2 * const b, int N )
{
    int pj, si, ei, thread_id, warp_id, lane_id, offset;
    rvec2 sum;

    thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    warp_id = thread_id / warpSize;

    if ( warp_id >= N )
    {
        return;
    }

    lane_id = thread_id % warpSize;
    si = row_ptr_start[warp_id];
    ei = row_ptr_end[warp_id];

    /* A symmetric, upper triangular portion stored
     * => diagonal only contributes once */
    if ( lane_id == 0 )
    {
        sum[0] = vals[si] * x[warp_id][0];
        sum[1] = vals[si] * x[warp_id][1];
    }
    else
    {
        sum[0] = 0.0;
        sum[1] = 0.0;
    }

    /* partial sums per thread */
    for ( pj = si + lane_id + 1; pj < ei; pj += warpSize )
    {
        sum[0] += vals[pj] * x[col_ind[pj]][0];
        sum[1] += vals[pj] * x[col_ind[pj]][1];
        /* symmetric contribution to row j */
        atomicAdd( (double *) &b[col_ind[pj]][0], (double) (vals[pj] * x[warp_id][0]) );
        atomicAdd( (double *) &b[col_ind[pj]][1], (double) (vals[pj] * x[warp_id][1]) );
    }

    /* warp-level reduction of partial sums
     * using registers within a warp */
    for ( offset = warpSize >> 1; offset > 0; offset >>= 1 )
    {
        sum[0] += __shfl_down(sum[0], offset );
        sum[1] += __shfl_down(sum[1], offset );
    }

    /* local contribution to row i for this warp */
    if ( lane_id == 0 )
    {
        atomicAdd( (double *) &b[warp_id][0], (double) sum[0] );
        atomicAdd( (double *) &b[warp_id][1], (double) sum[1] );
    }
}


/* 1 thread per row implementation */
HIP_GLOBAL void k_dual_sparse_matvec_full_csr( sparse_matrix A,
        rvec2 const * const x, rvec2 * const b, int n )
{
    int i, pj, si, ei;
    rvec2 sum;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    sum[0] = 0.0;
    sum[1] = 0.0;
    si = A.start[i];
    ei = A.end[i];

    for ( pj = si; pj < ei; ++pj )
    {
        sum[0] += A.val[pj] * x[A.j[pj]][0];
        sum[1] += A.val[pj] * x[A.j[pj]][1];
    }

    b[i][0] = sum[0];
    b[i][1] = sum[1];
}


/* sparse matrix, dense vector multiplication AX = B,
 * where warps
 * collaborate to multiply each row
 *
 * A: symmetric matrix,
 *    stored in CSR format
 * X: 2 dense vectors, size equal to num. columns in A
 * B (output): 2 dense vectors, size equal to num. columns in A
 * n: number of rows in A */
HIP_GLOBAL void k_dual_sparse_matvec_full_opt_csr( int *row_ptr_start,
        int *row_ptr_end, int *col_ind, real *vals,
        rvec2 const * const x, rvec2 * const b, int n )
{
    int pj, si, ei, thread_id, warp_id, lane_id, offset;
    rvec2 sum;

    thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    warp_id = thread_id / warpSize;

    if ( warp_id >= n )
    {
        return;
    }

    lane_id = thread_id % warpSize;
    si = row_ptr_start[warp_id];
    ei = row_ptr_end[warp_id];
    sum[0] = 0.0;
    sum[1] = 0.0;

    /* partial sums per thread */
    for ( pj = si + lane_id; pj < ei; pj += warpSize )
    {
        sum[0] += vals[pj] * x[col_ind[pj]][0];
        sum[1] += vals[pj] * x[col_ind[pj]][1];
    }

    /* warp-level reduction of partial sums
     * using registers within a warp */
    for ( offset = warpSize >> 1; offset > 0; offset >>= 1 )
    {
        sum[0] += __shfl_down(sum[0], offset );
        sum[1] += __shfl_down(sum[1], offset );
    }

    __syncthreads( );

    /* first thread within a warp writes sum to global memory */
    if ( lane_id == 0 )
    {
        b[warp_id][0] = sum[0];
        b[warp_id][1] = sum[1];
    }
}


void dual_jacobi_apply( real const * const Hdia_inv, rvec2 const * const y,
        rvec2 * const x, int n, hipStream_t s )
{
    int blocks;

    blocks = (n / DEF_BLOCK_SIZE)
        + ((n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_dual_jacobi_apply <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( Hdia_inv, y, x, n );
    hipCheckError( );
}


void jacobi_apply( real const * const Hdia_inv, real const * const y,
        real * const x, int n, hipStream_t s )
{
    int blocks;

    blocks = (n / DEF_BLOCK_SIZE)
        + ((n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_jacobi_apply <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( Hdia_inv, y, x, n );
    hipCheckError( );
}


/* Communications for sparse matrix-dense vector multiplication AX = B
 *
 * system:
 * control: 
 * mpi_data:
 * x: dense vector (device)
 * n: number of entries in x
 * buf_type: data structure type for x
 * mpi_type: MPI_Datatype struct for communications
 *
 * returns: communication time
 */
static void Dual_Sparse_MatVec_Comm_Part1( const reax_system * const system,
        const control_params * const control, storage * const workspace,
        mpi_datatypes * const mpi_data, void const * const x, int n,
        int buf_type, MPI_Datatype mpi_type )
{
#if !defined(HIP_DEVICE_PACK)
    rvec2 *spad;

    smalloc_check( &workspace->host_scratch, &workspace->host_scratch_size,
            sizeof(rvec2) * n, TRUE, SAFE_ZONE, __FILE__, __LINE__ );
    spad = (rvec2 *) workspace->host_scratch;

    sHipMemcpyAsync( spad, (void *) x, sizeof(rvec2) * n,
            hipMemcpyDeviceToHost, control->streams[4], __FILE__, __LINE__ );

    hipStreamSynchronize( control->streams[4] );

    /* exploit 3D domain decomposition of simulation space with 3-stage communication pattern */
    Dist( system, mpi_data, spad, buf_type, mpi_type );

    sHipMemcpyAsync( (void *) x, spad, sizeof(rvec2) * n,
            hipMemcpyHostToDevice, control->streams[4], __FILE__, __LINE__ );
#else
    /* exploit 3D domain decomposition of simulation space with 3-stage communication pattern */
    Hip_Dist( system, workspace, mpi_data, x, buf_type, mpi_type, control->streams[4] );
#endif
}


/* Local arithmetic portion of sparse matrix-dense vector multiplication AX = B
 *
 * control:
 * A: sparse matrix, 1D partitioned row-wise
 * x: dense vector
 * b (output): dense vector
 * n: number of entries in b
 */
static void Dual_Sparse_MatVec_local( control_params const * const control,
        sparse_matrix const * const A, rvec2 const * const x,
        rvec2 * const b, int n, hipStream_t s )
{
    int blocks;

    if ( A->format == SYM_HALF_MATRIX )
    {
        /* half-format requires entries of b be initialized to zero */
        sHipMemsetAsync( b, 0, sizeof(rvec2) * n, s, __FILE__, __LINE__ );

        /* 1 thread per row implementation */
//        k_dual_sparse_matvec_half_csr <<< control->blocks, control->block_size, 0, s >>>
//            ( A->start, A->end, A->j, A->val, x, b, A->n );

        blocks = A->n * warpSize / DEF_BLOCK_SIZE
            + (A->n * warpSize % DEF_BLOCK_SIZE == 0 ? 0 : 1);
        
        /* 32 threads per row implementation
         * using registers to accumulate partial row sums */
        k_dual_sparse_matvec_half_opt_csr <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
             ( A->start, A->end, A->j, A->val, x, b, A->n );
    }
    else if ( A->format == SYM_FULL_MATRIX )
    {
        /* 1 thread per row implementation */
//        k_dual_sparse_matvec_full_csr <<< control->blocks_n, control->blocks_size_n, 0, s >>>
//             ( *A, x, b, A->n );

        blocks = ((A->n * warpSize) / DEF_BLOCK_SIZE)
            + (((A->n * warpSize) % DEF_BLOCK_SIZE) == 0 ? 0 : 1);
        
        /* 32 threads per row implementation
         * using registers to accumulate partial row sums */
        k_dual_sparse_matvec_full_opt_csr <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
                ( A->start, A->end, A->j, A->val, x, b, A->n );
    }
    hipCheckError( );
}


/* Communications for collecting the distributed partial sums
 * in the sparse matrix-dense vector multiplication AX = B.
 * Specifically, B contains the distributed partial sums
 * (and hence has the same number of entries as X).
 *
 * system:
 * control:
 * mpi_data:
 * mat_format: storage type of sparse matrix A
 * b: dense vector (device)
 * n1: number of entries in x
 * n2: number of entries in b (at output)
 * buf_type: data structure type for b
 * mpi_type: MPI_Datatype struct for communications
 *
 * returns: communication time
 */
static void Dual_Sparse_MatVec_Comm_Part2( const reax_system * const system,
        const control_params * const control, storage * const workspace,
        mpi_datatypes * const mpi_data, int mat_format,
        void * const b, int n1, int n2, int buf_type, MPI_Datatype mpi_type )
{
#if !defined(HIP_DEVICE_PACK)
    rvec2 *spad;
#endif

    /* reduction required for symmetric half matrix */
    if ( mat_format == SYM_HALF_MATRIX )
    {
#if !defined(HIP_DEVICE_PACK)
        smalloc_check( &workspace->host_scratch, &workspace->host_scratch_size,
                sizeof(rvec2) * n1, TRUE, SAFE_ZONE, __FILE__, __LINE__ );
        spad = (rvec2 *) workspace->host_scratch;

        sHipMemcpyAsync( spad, b, sizeof(rvec2) * n1,
                hipMemcpyDeviceToHost, control->streams[4], __FILE__, __LINE__ );

        hipStreamSynchronize( control->streams[4] );

        Coll( system, mpi_data, spad, buf_type, mpi_type );

        sHipMemcpyAsync( b, spad, sizeof(rvec2) * n2,
                hipMemcpyHostToDevice, control->streams[4], __FILE__, __LINE__ );
#else
        Hip_Coll( system, mpi_data, b, buf_type, mpi_type, control->streams[4] );
#endif
    }
}


/* sparse matrix, dense vector multiplication AX = B
 *
 * system:
 * control:
 * data:
 * workspace: storage container for workspace structures
 * A: symmetric matrix,
 *    stored in CSR format
 * X: dense vector, size equal to num. columns in A
 * n: number of rows in X
 * B (output): dense vector */
static void Dual_Sparse_MatVec( const reax_system * const system,
        control_params const * const control, simulation_data * const data,
        storage * const workspace, mpi_datatypes * const mpi_data,
        sparse_matrix const * const A, rvec2 const * const x,
        int n, rvec2 * const b )
{
#if defined(LOG_PERFORMANCE)
    real time;

    time = Get_Time( );
#endif

    Dual_Sparse_MatVec_Comm_Part1( system, control, workspace, mpi_data,
            x, n, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_comm );
#endif

    Dual_Sparse_MatVec_local( control, A, x, b, n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_spmv );
#endif

    Dual_Sparse_MatVec_Comm_Part2( system, control, workspace, mpi_data,
            A->format, b, n, A->n, RVEC2_PTR_TYPE, mpi_data->mpi_rvec2 );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_comm );
#endif
}


/* Communications for sparse matrix-dense vector multiplication Ax = b
 *
 * system:
 * control: 
 * mpi_data:
 * x: dense vector (device)
 * n: number of entries in x
 * buf_type: data structure type for x
 * mpi_type: MPI_Datatype struct for communications
 */
static void Sparse_MatVec_Comm_Part1( const reax_system * const system,
        const control_params * const control, storage * const workspace,
        mpi_datatypes * const mpi_data, void const * const x, int n,
        int buf_type, MPI_Datatype mpi_type )
{
#if !defined(HIP_DEVICE_PACK)
    real *spad;

    smalloc_check( &workspace->host_scratch, &workspace->host_scratch_size,
            sizeof(real) * n, TRUE, SAFE_ZONE, __FILE__, __LINE__ );
    spad = (real *) workspace->host_scratch;

    sHipMemcpyAsync( spad, (void *) x, sizeof(real) * n,
            hipMemcpyDeviceToHost, control->streams[4], __FILE__, __LINE__ );

    hipStreamSynchronize( control->streams[4] );

    /* exploit 3D domain decomposition of simulation space with 3-stage communication pattern */
    Dist( system, mpi_data, spad, buf_type, mpi_type );

    sHipMemcpyAsync( (void *) x, spad, sizeof(real) * n,
            hipMemcpyHostToDevice, control->streams[4], __FILE__, __LINE__ );
#else
    /* exploit 3D domain decomposition of simulation space with 3-stage communication pattern */
    Hip_Dist( system, workspace, mpi_data, x, buf_type, mpi_type, control->streams[4] );
#endif
}


/* Local arithmetic portion of sparse matrix-dense vector multiplication Ax = b
 *
 * control:
 * A: sparse matrix, 1D partitioned row-wise
 * x: dense vector
 * b (output): dense vector
 * n: number of entries in b
 */
static void Sparse_MatVec_local( control_params const * const control,
        sparse_matrix const * const A, real const * const x,
        real * const b, int n, hipStream_t s )
{
    int blocks;

    if ( A->format == SYM_HALF_MATRIX )
    {
        /* half-format requires entries of b be initialized to zero */
        sHipMemsetAsync( b, 0, sizeof(real) * n, s, __FILE__, __LINE__ );

        /* 1 thread per row implementation */
//        k_sparse_matvec_half_csr <<< control->blocks, control->block_size, 0, s >>>
//            ( A->start, A->end, A->j, A->val, x, b, A->n );

        blocks = (A->n * warpSize / DEF_BLOCK_SIZE)
            + (A->n * warpSize % DEF_BLOCK_SIZE == 0 ? 0 : 1);

        /* 32 threads per row implementation
         * using registers to accumulate partial row sums */
        k_sparse_matvec_half_opt_csr <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
             ( A->start, A->end, A->j, A->val, x, b, A->n );
    }
    else if ( A->format == SYM_FULL_MATRIX )
    {
        /* 1 thread per row implementation */
//        k_sparse_matvec_full_csr <<< control->blocks, control->blocks_size, 0, s >>>
//             ( A->start, A->end, A->j, A->val, x, b, A->n );

        blocks = ((A->n * warpSize) / DEF_BLOCK_SIZE)
            + (((A->n * warpSize) % DEF_BLOCK_SIZE) == 0 ? 0 : 1);

        /*
         * using registers to accumulate partial row sums */
        k_sparse_matvec_full_opt_csr <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
             ( A->start, A->end, A->j, A->val, x, b, A->n );
    }
    hipCheckError( );
}


/* Communications for collecting the distributed partial sums
 * in the sparse matrix-dense vector multiplication Ax = b.
 * Specifically, b contains the distributed partial sums
 * (and hence has the same number of entries as x).
 *
 * system:
 * control:
 * mpi_data:
 * mat_format: storage type of sparse matrix A
 * b: dense vector (device)
 * n1: number of entries in x
 * n2: number of entries in b (at output)
 * buf_type: data structure type for b
 * mpi_type: MPI_Datatype struct for communications
 */
static void Sparse_MatVec_Comm_Part2( const reax_system * const system,
        const control_params * const control, storage * const workspace,
        mpi_datatypes * const mpi_data, int mat_format,
        void * const b, int n1, int n2, int buf_type, MPI_Datatype mpi_type )
{
#if !defined(HIP_DEVICE_PACK)
    real *spad;
#endif

    /* reduction required for symmetric half matrix */
    if ( mat_format == SYM_HALF_MATRIX )
    {
#if !defined(HIP_DEVICE_PACK)
        smalloc_check( &workspace->host_scratch, &workspace->host_scratch_size,
                sizeof(real) * n1, TRUE, SAFE_ZONE, __FILE__, __LINE__ );
        spad = (real *) workspace->host_scratch;

        sHipMemcpyAsync( spad, b, sizeof(real) * n1,
                hipMemcpyDeviceToHost, control->streams[4], __FILE__, __LINE__ );

        hipStreamSynchronize( control->streams[4] );

        Coll( system, mpi_data, spad, buf_type, mpi_type );

        sHipMemcpyAsync( b, spad, sizeof(real) * n2,
                hipMemcpyHostToDevice, control->streams[4], __FILE__, __LINE__ );
#else
        Hip_Coll( system, mpi_data, b, buf_type, mpi_type, control->streams[4] );
#endif
    }
}


/* sparse matrix, dense vector multiplication Ax = b
 *
 * system:
 * control:
 * data:
 * workspace: storage container for workspace structures
 * A: symmetric matrix,
 *    stored in CSR format
 * x: dense vector
 * n: number of entries in x
 * b (output): dense vector */
static void Sparse_MatVec( reax_system const * const system,
        control_params const * const control, simulation_data * const data,
        storage * const workspace, mpi_datatypes * const mpi_data,
        sparse_matrix const * const A, real const * const x,
        int n, real * const b )
{
#if defined(LOG_PERFORMANCE)
    real time;

    time = Get_Time( );
#endif

    Sparse_MatVec_Comm_Part1( system, control, workspace, mpi_data,
            x, n, REAL_PTR_TYPE, MPI_DOUBLE );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_comm );
#endif

    Sparse_MatVec_local( control, A, x, b, n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_spmv );
#endif

    Sparse_MatVec_Comm_Part2( system, control, workspace, mpi_data,
            A->format, b, n, A->n, REAL_PTR_TYPE, MPI_DOUBLE );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_comm );
#endif
}


/* Apply left-sided preconditioning while solving M^{-1}AX = M^{-1}B
 *
 * system:
 * workspace: data struct containing matrices and vectors, stored in CSR
 * control: data struct containing parameters
 * data: struct containing timing simulation data (including performance data)
 * y: vector to which to apply preconditioning,
 *  specific to internals of iterative solver being used
 * x (output): preconditioned vector
 * fresh_pre: parameter indicating if this is a newly computed (fresh) preconditioner
 * side: used in determining how to apply preconditioner if the preconditioner is
 *  factorized as M = M_{1}M_{2} (e.g., incomplete LU, A \approx LU)
 *
 * Assumptions:
 *   Matrices have non-zero diagonals
 *   Each row of a matrix has at least one non-zero (i.e., no rows with all zeros) */
static void dual_apply_preconditioner( reax_system const * const system,
        storage * const workspace, control_params const * const control,
        simulation_data * const data, mpi_datatypes * const  mpi_data,
        rvec2 const * const y, rvec2 * const x, int fresh_pre, int side )
{
//    int i, si;

    /* no preconditioning */
    if ( control->cm_solver_pre_comp_type == NONE_PC )
    {
        if ( x != y )
        {
            Vector_Copy_rvec2( x, y, system->n, control->streams[4] );
        }
    }
    else
    {
        switch ( side )
        {
            case LEFT:
                switch ( control->cm_solver_pre_app_type )
                {
                    case TRI_SOLVE_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                                dual_jacobi_apply( workspace->d_workspace->Hdia_inv,
                                        y, x, system->n, control->streams[4] );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve( workspace->L, y, x, workspace->L->n, LOWER );
//                                  break;
                            case SAI_PC:
#if defined(NEUTRAL_TERRITORY)
                                Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
                                        &workspace->d_workspace->H_app_inv,
                                        y, H->NT, x );
#else
                                Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
                                        &workspace->d_workspace->H_app_inv,
                                        y, system->n, x );
#endif
                                break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case TRI_SOLVE_LEVEL_SCHED_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                                dual_jacobi_apply( workspace->d_workspace->Hdia_inv,
                                        y, x, system->n, control->streams[4] );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve_level_sched( (static_storage *) workspace,
//                                          workspace->L, y, x, workspace->L->n, LOWER, fresh_pre );
//                                  break;
                            case SAI_PC:
#if defined(NEUTRAL_TERRITORY)
                                Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
                                        &workspace->d_workspace->H_app_inv,
                                        y, H->NT, x );
#else
                                Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
                                        &workspace->d_workspace->H_app_inv,
                                        y, system->n, x );
#endif
                                break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case TRI_SOLVE_GC_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  for ( i = 0; i < workspace->H->n; ++i )
//                                  {
//                                      workspace->y_p[i] = y[i];
//                                  }
//
//                                  permute_vector( workspace, workspace->y_p, workspace->H->n, FALSE, LOWER );
//                                  tri_solve_level_sched( (static_storage *) workspace,
//                                  workspace->L, workspace->y_p, x, workspace->L->n, LOWER, fresh_pre );
//                                  break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case JACOBI_ITER_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                // construct D^{-1}_L
//                                if ( fresh_pre == TRUE )
//                                {
//                                    for ( i = 0; i < workspace->L->n; ++i )
//                                    {
//                                        si = workspace->L->start[i + 1] - 1;
//                                        workspace->Dinv_L[i] = 1.0 / workspace->L->val[si];
//                                    }
//                                }
//
//                                jacobi_iter( workspace, workspace->L, workspace->Dinv_L,
//                                        y, x, LOWER, control->cm_solver_pre_app_jacobi_iters );
//                                break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    default:
                        fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                        exit( INVALID_INPUT );
                        break;

                }
                break;

            case RIGHT:
                switch ( control->cm_solver_pre_app_type )
                {
                    case TRI_SOLVE_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                if ( x != y )
                                {
                                    Vector_Copy_rvec2( x, y, system->n, control->streams[4] );
                                }
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve( workspace->U, y, x, workspace->U->n, UPPER );
//                                  break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case TRI_SOLVE_LEVEL_SCHED_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                if ( x != y )
                                {
                                    Vector_Copy_rvec2( x, y, system->n, control->streams[4] );
                                }
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve_level_sched( (static_storage *) workspace,
//                                          workspace->U, y, x, workspace->U->n, UPPER, fresh_pre );
//                                  break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case TRI_SOLVE_GC_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve_level_sched( (static_storage *) workspace,
//                                  workspace->U, y, x, workspace->U->n, UPPER, fresh_pre );
//                                  permute_vector( workspace, x, workspace->H->n, TRUE, UPPER );
//                                  break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case JACOBI_ITER_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  if ( fresh_pre == TRUE )
//                                  {
//                                      for ( i = 0; i < workspace->U->n; ++i )
//                                      {
//                                          si = workspace->U->start[i];
//                                          workspace->Dinv_U[i] = 1.0 / workspace->U->val[si];
//                                      }
//                                  }
//
//                                  jacobi_iter( workspace, workspace->U, workspace->Dinv_U,
//                                          y, x, UPPER, control->cm_solver_pre_app_jacobi_iters );
//                                  break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    default:
                        fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                        exit( INVALID_INPUT );
                        break;

                }
                break;
        }
    }
}


/* Apply left-sided preconditioning while solving M^{-1}Ax = M^{-1}b
 *
 * system:
 * workspace: data struct containing matrices and vectors, stored in CSR
 * control: data struct containing parameters
 * data: struct containing timing simulation data (including performance data)
 * y: vector to which to apply preconditioning,
 *  specific to internals of iterative solver being used
 * x (output): preconditioned vector
 * fresh_pre: parameter indicating if this is a newly computed (fresh) preconditioner
 * side: used in determining how to apply preconditioner if the preconditioner is
 *  factorized as M = M_{1}M_{2} (e.g., incomplete LU, A \approx LU)
 *
 * Assumptions:
 *   Matrices have non-zero diagonals
 *   Each row of a matrix has at least one non-zero (i.e., no rows with all zeros) */
static void apply_preconditioner( reax_system const * const system,
        storage * const workspace, control_params const * const control,
        simulation_data * const data, mpi_datatypes * const  mpi_data,
        real const * const y, real * const x, int fresh_pre, int side )
{
//    int i, si;

    /* no preconditioning */
    if ( control->cm_solver_pre_comp_type == NONE_PC )
    {
        if ( x != y )
        {
            Vector_Copy( x, y, system->n, control->streams[4] );
        }
    }
    else
    {
        switch ( side )
        {
            case LEFT:
                switch ( control->cm_solver_pre_app_type )
                {
                    case TRI_SOLVE_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                                jacobi_apply( workspace->d_workspace->Hdia_inv,
                                        y, x, system->n, control->streams[4] );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve( workspace->L, y, x, workspace->L->n, LOWER );
//                                  break;
                            case SAI_PC:
#if defined(NEUTRAL_TERRITORY)
                                Sparse_MatVec( system, control, data, workspace, mpi_data,
                                        &workspace->d_workspace->H_app_inv,
                                        y, H->NT, x );
#else
                                Sparse_MatVec( system, control, data, workspace, mpi_data,
                                        &workspace->d_workspace->H_app_inv,
                                        y, system->n, x );
#endif
                                break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case TRI_SOLVE_LEVEL_SCHED_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                                jacobi_apply( workspace->d_workspace->Hdia_inv,
                                        y, x, system->n, control->streams[4] );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve_level_sched( (static_storage *) workspace,
//                                          workspace->L, y, x, workspace->L->n, LOWER, fresh_pre );
//                                  break;
                            case SAI_PC:
#if defined(NEUTRAL_TERRITORY)
                                Sparse_MatVec( system, control, data, workspace, mpi_data,
                                        &workspace->d_workspace->H_app_inv,
                                        y, H->NT, x );
#else
                                Sparse_MatVec( system, control, data, workspace, mpi_data,
                                        &workspace->d_workspace->H_app_inv,
                                        y, system->n, x );
#endif
                                break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case TRI_SOLVE_GC_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  for ( i = 0; i < workspace->H->n; ++i )
//                                  {
//                                      workspace->y_p[i] = y[i];
//                                  }
//
//                                  permute_vector( workspace, workspace->y_p, workspace->H->n, FALSE, LOWER );
//                                  tri_solve_level_sched( (static_storage *) workspace,
//                                  workspace->L, workspace->y_p, x, workspace->L->n, LOWER, fresh_pre );
//                                  break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case JACOBI_ITER_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                // construct D^{-1}_L
//                                if ( fresh_pre == TRUE )
//                                {
//                                    for ( i = 0; i < workspace->L->n; ++i )
//                                    {
//                                        si = workspace->L->start[i + 1] - 1;
//                                        workspace->Dinv_L[i] = 1.0 / workspace->L->val[si];
//                                    }
//                                }
//
//                                jacobi_iter( workspace, workspace->L, workspace->Dinv_L,
//                                        y, x, LOWER, control->cm_solver_pre_app_jacobi_iters );
//                                break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    default:
                        fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                        exit( INVALID_INPUT );
                        break;

                }
                break;

            case RIGHT:
                switch ( control->cm_solver_pre_app_type )
                {
                    case TRI_SOLVE_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                if ( x != y )
                                {
                                    Vector_Copy( x, y, system->n, control->streams[4] );
                                }
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve( workspace->U, y, x, workspace->U->n, UPPER );
//                                  break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case TRI_SOLVE_LEVEL_SCHED_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                if ( x != y )
                                {
                                    Vector_Copy( x, y, system->n, control->streams[4] );
                                }
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve_level_sched( (static_storage *) workspace,
//                                          workspace->U, y, x, workspace->U->n, UPPER, fresh_pre );
//                                  break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case TRI_SOLVE_GC_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  tri_solve_level_sched( (static_storage *) workspace,
//                                  workspace->U, y, x, workspace->U->n, UPPER, fresh_pre );
//                                  permute_vector( workspace, x, workspace->H->n, TRUE, UPPER );
//                                  break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    case JACOBI_ITER_PA:
                        switch ( control->cm_solver_pre_comp_type )
                        {
                            case JACOBI_PC:
                            case SAI_PC:
                                fprintf( stderr, "Unsupported preconditioner computation/application method combination. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
//                            case ICHOLT_PC:
//                            case ILUT_PC:
//                            case ILUTP_PC:
//                                  if ( fresh_pre == TRUE )
//                                  {
//                                      for ( i = 0; i < workspace->U->n; ++i )
//                                      {
//                                          si = workspace->U->start[i];
//                                          workspace->Dinv_U[i] = 1.0 / workspace->U->val[si];
//                                      }
//                                  }
//
//                                  jacobi_iter( workspace, workspace->U, workspace->Dinv_U,
//                                          y, x, UPPER, control->cm_solver_pre_app_jacobi_iters );
//                                  break;
                            default:
                                fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                                exit( INVALID_INPUT );
                                break;
                        }
                        break;
                    default:
                        fprintf( stderr, "Unrecognized preconditioner application method. Terminating...\n" );
                        exit( INVALID_INPUT );
                        break;

                }
                break;
        }
    }
}


int Hip_dual_SDM( reax_system const * const system,
        control_params const * const control,
        simulation_data * const data, storage * const workspace,
        sparse_matrix const * const H, rvec2 const * const b, real tol,
        rvec2 * const x, mpi_datatypes * const mpi_data, int fresh_pre )
{
    unsigned int i, matvecs;
    int ret;
    rvec2 tmp, alpha, sig, b_norm;
    real redux[4];
#if defined(LOG_PERFORMANCE)
    real time;
#endif

    Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, x, system->N, workspace->d_workspace->q2 );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum_rvec2( workspace->d_workspace->r2, 1.0, 1.0, b,
            -1.0, -1.0, workspace->d_workspace->q2, system->n,
	   control->streams[4] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    dual_apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->r2, workspace->d_workspace->q2, fresh_pre, LEFT );
    dual_apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->q2, workspace->d_workspace->d2, fresh_pre, RIGHT );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

    Dot_local_rvec2( workspace, b, b, system->n, &redux[0], &redux[1], control->streams[4] );
    Dot_local_rvec2( workspace, workspace->d_workspace->r2,
            workspace->d_workspace->d2, system->n, &redux[2], &redux[3], control->streams[4] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Allreduce( MPI_IN_PLACE, redux, 4, MPI_DOUBLE,
            MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    b_norm[0] = SQRT( redux[0] );
    b_norm[1] = SQRT( redux[1] );
    sig[0] = redux[2];
    sig[1] = redux[3];

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    for ( i = 0; i < control->cm_solver_max_iters; ++i )
    {
        if ( SQRT(sig[0]) / b_norm[0] <= tol || SQRT(sig[1]) / b_norm[1] <= tol )
        {
            break;
        }

        Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
                H, workspace->d_workspace->d2, system->N, workspace->d_workspace->q2 );

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        Dot_local_rvec2( workspace, workspace->d_workspace->r2,
                workspace->d_workspace->d2, system->n, &redux[0], &redux[1], control->streams[4] );
        Dot_local_rvec2( workspace, workspace->d_workspace->d2,
                workspace->d_workspace->q2, system->n, &redux[2], &redux[3], control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 4, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        sig[0] = redux[0];
        sig[1] = redux[1];
        tmp[0] = redux[2];
        tmp[1] = redux[3];
        alpha[0] = sig[0] / tmp[0];
        alpha[1] = sig[1] / tmp[1];
        Vector_Add_rvec2( x, alpha[0], alpha[1], workspace->d_workspace->d2, system->n, control->streams[4] );
        Vector_Add_rvec2( workspace->d_workspace->r2, -1.0 * alpha[0], -1.0 * alpha[1],
                workspace->d_workspace->q2, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        dual_apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->r2, workspace->d_workspace->q2, FALSE, LEFT );
        dual_apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->q2, workspace->d_workspace->d2, FALSE, RIGHT );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif
    }

    if ( SQRT(sig[0]) / b_norm[0] <= tol
            && SQRT(sig[1]) / b_norm[1] > tol )
    {
        Vector_Copy_From_rvec2( workspace->d_workspace->t,
                workspace->d_workspace->x, 1, system->n, control->streams[4] );

        matvecs = Hip_SDM( system, control, data, workspace, H,
                workspace->d_workspace->b_t, tol,
                workspace->d_workspace->t, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->d_workspace->x,
                workspace->d_workspace->t, 1, system->n, control->streams[4] );
    }
    else if ( SQRT(sig[1]) / b_norm[1] <= tol
            && SQRT(sig[0]) / b_norm[0] > tol )
    {
        Vector_Copy_From_rvec2( workspace->d_workspace->s,
                workspace->d_workspace->x, 0, system->n, control->streams[4] );

        matvecs = Hip_SDM( system, control, data, workspace, H,
                workspace->d_workspace->b_s, tol,
                workspace->d_workspace->s, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->d_workspace->x,
                workspace->d_workspace->s, 0, system->n, control->streams[4] );
    }
    else
    {
        matvecs = 0;
    }

    if ( i >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] p%d: dual SDM convergence failed (%d iters)\n",
                system->my_rank, i );
        fprintf( stderr, "    [INFO] Rel. residual error for s solve: %e\n", SQRT(sig[0]) / b_norm[0] );
        fprintf( stderr, "    [INFO] Rel. residual error for t solve: %e\n", SQRT(sig[1]) / b_norm[1] );
    }

    return (i + 1) + matvecs;
}


/* Steepest Descent */
int Hip_SDM( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        sparse_matrix const * const H, real const * const b, real tol,
        real * const x, mpi_datatypes * const mpi_data, int fresh_pre )
{
    unsigned int i;
    int ret;
    real tmp, alpha, sig, b_norm;
    real redux[2];
#if defined(LOG_PERFORMANCE)
    real time;
#endif

    Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, x, system->N, workspace->d_workspace->q );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum( workspace->d_workspace->r, 1.0, b,
            -1.0, workspace->d_workspace->q, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->r, workspace->d_workspace->q, fresh_pre, LEFT );
    apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->q, workspace->d_workspace->d, fresh_pre, RIGHT );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

    redux[0] = Dot_local( workspace, b, b, system->n, control->streams[4] );
    redux[1] = Dot_local( workspace, workspace->d_workspace->r,
            workspace->d_workspace->d, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
            MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    b_norm = SQRT( redux[0] );
    sig = redux[1];

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    for ( i = 0; i < control->cm_solver_max_iters && SQRT(sig) / b_norm > tol; ++i )
    {
        Sparse_MatVec( system, control, data, workspace, mpi_data,
                H, workspace->d_workspace->d, system->N, workspace->d_workspace->q );

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        redux[0] = Dot_local( workspace, workspace->d_workspace->r,
                workspace->d_workspace->d, system->n, control->streams[4] );
        redux[1] = Dot_local( workspace, workspace->d_workspace->d,
                workspace->d_workspace->q, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        sig = redux[0];
        tmp = redux[1];
        alpha = sig / tmp;
        Vector_Add( x, alpha, workspace->d_workspace->d, system->n, control->streams[4] );
        Vector_Add( workspace->d_workspace->r, -1.0 * alpha,
                workspace->d_workspace->q, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->r, workspace->d_workspace->q, FALSE, LEFT );
        apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->q, workspace->d_workspace->d, FALSE, RIGHT );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif
    }

    if ( i >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] p%d: SDM convergence failed (%d iters)\n",
                system->my_rank, i );
        fprintf( stderr, "    [INFO] Rel. residual error: %e\n", SQRT(sig) / b_norm );
    }

    return i;
}


/* Dual iteration for the Preconditioned Conjugate Gradient Method
 * for QEq (2 simultaneous solves) */
int Hip_dual_CG( reax_system const * const system,
        control_params const * const control,
        simulation_data * const data, storage * const workspace,
        sparse_matrix const * const H, rvec2 const * const b, real tol,
        rvec2 * const x, mpi_datatypes * const mpi_data, int fresh_pre )
{
    unsigned int i, matvecs;
    int ret;
    rvec2 tmp, alpha, beta, r_norm, b_norm, sig_old, sig_new;
    real redux[6];
#if defined(LOG_PERFORMANCE)
    real time;
#endif

    Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, x, system->N, workspace->d_workspace->q2 );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum_rvec2( workspace->d_workspace->r2, 1.0, 1.0, b,
            -1.0, -1.0, workspace->d_workspace->q2, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    dual_apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->r2, workspace->d_workspace->q2, fresh_pre, LEFT );
    dual_apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->q2, workspace->d_workspace->d2, fresh_pre, RIGHT );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

    Dot_local_rvec2( workspace, workspace->d_workspace->r2,
            workspace->d_workspace->d2, system->n, &redux[0], &redux[1], control->streams[4] );
    Dot_local_rvec2( workspace, workspace->d_workspace->d2,
            workspace->d_workspace->d2, system->n, &redux[2], &redux[3], control->streams[4] );
    Dot_local_rvec2( workspace, b, b, system->n, &redux[4], &redux[5], control->streams[4] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Allreduce( MPI_IN_PLACE, redux, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    sig_new[0] = redux[0];
    sig_new[1] = redux[1];
    r_norm[0] = SQRT( redux[2] );
    r_norm[1] = SQRT( redux[3] );
    b_norm[0] = SQRT( redux[4] );
    b_norm[1] = SQRT( redux[5] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    for ( i = 0; i < control->cm_solver_max_iters; ++i )
    {
        if ( r_norm[0] / b_norm[0] <= tol || r_norm[1] / b_norm[1] <= tol )
        {
            break;
        }

        Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
                H, workspace->d_workspace->d2, system->N, workspace->d_workspace->q2 );

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        Dot_local_rvec2( workspace, workspace->d_workspace->d2,
                workspace->d_workspace->q2, system->n, &redux[0], &redux[1], control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        tmp[0] = redux[0];
        tmp[1] = redux[1];

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        alpha[0] = sig_new[0] / tmp[0];
        alpha[1] = sig_new[1] / tmp[1];
        Vector_Add_rvec2( x, alpha[0], alpha[1],
                workspace->d_workspace->d2, system->n, control->streams[4] );
        Vector_Add_rvec2( workspace->d_workspace->r2, -1.0 * alpha[0], -1.0 * alpha[1],
                workspace->d_workspace->q2, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        dual_apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->r2, workspace->d_workspace->q2, FALSE, LEFT );
        dual_apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->q2, workspace->d_workspace->p2, FALSE, RIGHT );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

        Dot_local_rvec2( workspace, workspace->d_workspace->r2,
                workspace->d_workspace->p2, system->n, &redux[0], &redux[1], control->streams[4] );
        Dot_local_rvec2( workspace, workspace->d_workspace->p2,
                workspace->d_workspace->p2, system->n, &redux[2], &redux[3], control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        sig_old[0] = sig_new[0];
        sig_old[1] = sig_new[1];
        sig_new[0] = redux[0];
        sig_new[1] = redux[1];
        r_norm[0] = SQRT( redux[2] );
        r_norm[1] = SQRT( redux[3] );
        beta[0] = sig_new[0] / sig_old[0];
        beta[1] = sig_new[1] / sig_old[1];
        /* d = p + beta * d */
        Vector_Sum_rvec2( workspace->d_workspace->d2,
                1.0, 1.0, workspace->d_workspace->p2,
                beta[0], beta[1], workspace->d_workspace->d2, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif
    }

    if ( r_norm[0] / b_norm[0] <= tol
            && r_norm[1] / b_norm[1] > tol )
    {
        Vector_Copy_From_rvec2( workspace->d_workspace->t,
                workspace->d_workspace->x, 1, system->n, control->streams[4] );

        matvecs = Hip_CG( system, control, data, workspace, H,
                workspace->d_workspace->b_t, tol,
                workspace->d_workspace->t, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->d_workspace->x,
                workspace->d_workspace->t, 1, system->n, control->streams[4] );
    }
    else if ( r_norm[1] / b_norm[1] <= tol
            && r_norm[0] / b_norm[0] > tol )
    {
        Vector_Copy_From_rvec2( workspace->d_workspace->s,
                workspace->d_workspace->x, 0, system->n, control->streams[4] );

        matvecs = Hip_CG( system, control, data, workspace, H,
                workspace->d_workspace->b_s, tol,
                workspace->d_workspace->s, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->d_workspace->x,
                workspace->d_workspace->s, 0, system->n, control->streams[4] );
    }
    else
    {
        matvecs = 0;
    }

    if ( i >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] p%d: dual CG convergence failed (%d iters)\n",
                system->my_rank, i );
        fprintf( stderr, "    [INFO] Rel. residual error for s solve: %e\n", r_norm[0] / b_norm[0] );
        fprintf( stderr, "    [INFO] Rel. residual error for t solve: %e\n", r_norm[1] / b_norm[1] );
    }

    return (i + 1) + matvecs;
}


/* Preconditioned Conjugate Gradient Method */
int Hip_CG( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        sparse_matrix const * const H, real const * const b, real tol,
        real * const x, mpi_datatypes * const mpi_data, int fresh_pre )
{
    unsigned int i;
    int ret;
    real tmp, alpha, beta, r_norm, b_norm;
    real sig_old, sig_new;
    real redux[3];
#if defined(LOG_PERFORMANCE)
    real time;
#endif

    Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, x, system->N, workspace->d_workspace->q );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum( workspace->d_workspace->r, 1.0, b,
            -1.0, workspace->d_workspace->q, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->r, workspace->d_workspace->q, fresh_pre, LEFT );
    apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->q, workspace->d_workspace->d, fresh_pre, RIGHT );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

    redux[0] = Dot_local( workspace, workspace->d_workspace->r,
            workspace->d_workspace->d, system->n, control->streams[4] );
    redux[1] = Dot_local( workspace, workspace->d_workspace->d,
            workspace->d_workspace->d, system->n, control->streams[4] );
    redux[2] = Dot_local( workspace, b, b, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Allreduce( MPI_IN_PLACE, redux, 3, MPI_DOUBLE,
            MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    sig_new = redux[0];
    r_norm = SQRT( redux[1] );
    b_norm = SQRT( redux[2] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    for ( i = 0; i < control->cm_solver_max_iters && r_norm / b_norm > tol; ++i )
    {
        Sparse_MatVec( system, control, data, workspace, mpi_data,
                H, workspace->d_workspace->d, system->N, workspace->d_workspace->q );

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        tmp = Dot( workspace, workspace->d_workspace->d, workspace->d_workspace->q,
                system->n, MPI_COMM_WORLD, control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        alpha = sig_new / tmp;
        Vector_Add( x, alpha, workspace->d_workspace->d, system->n, control->streams[4] );
        Vector_Add( workspace->d_workspace->r, -1.0 * alpha,
                workspace->d_workspace->q, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->r, workspace->d_workspace->q, FALSE, LEFT );
        apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->q, workspace->d_workspace->p, FALSE, RIGHT );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

        redux[0] = Dot_local( workspace, workspace->d_workspace->r,
                workspace->d_workspace->p, system->n, control->streams[4] );
        redux[1] = Dot_local( workspace, workspace->d_workspace->p,
                workspace->d_workspace->p, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        sig_old = sig_new;
        sig_new = redux[0];
        r_norm = SQRT( redux[1] );
        beta = sig_new / sig_old;
        Vector_Sum( workspace->d_workspace->d, 1.0, workspace->d_workspace->p,
                beta, workspace->d_workspace->d, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif
    }

    if ( i >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] p%d: CG convergence failed (%d iters)\n",
                system->my_rank, i );
        fprintf( stderr, "    [INFO] Rel. residual error: %e\n", r_norm / b_norm );
    }

    return i;
}


/* Bi-conjugate gradient stabalized method with left preconditioning for
 * solving nonsymmetric linear systems
 * Note: this version is for the dual QEq solver
 *
 * system: 
 * workspace: struct containing storage for workspace for the linear solver
 * control: struct containing parameters governing the simulation and numeric methods
 * data: struct containing simulation data (e.g., atom info)
 * H: sparse, symmetric matrix in CSR format
 * b: right-hand side of the linear system
 * tol: tolerence compared against the relative residual for determining convergence
 * x: inital guess
 * mpi_data: 
 *
 * Reference: Netlib (in MATLAB)
 *  http://www.netlib.org/templates/matlab/bicgstab.m
 * */
int Hip_dual_BiCGStab( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        sparse_matrix const * const H, rvec2 const * const b, real tol,
        rvec2 * const x, mpi_datatypes * const mpi_data, int fresh_pre )
{
    unsigned int i, matvecs;
    int ret;
    rvec2 tmp, alpha, beta, omega, sigma, rho, rho_old, r_norm, b_norm;
    real redux[4];
#if defined(LOG_PERFORMANCE)
    real time;
#endif

    Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, x, system->N, workspace->d_workspace->d2 );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum_rvec2( workspace->d_workspace->r2, 1.0, 1.0, b,
            -1.0, -1.0, workspace->d_workspace->d2, system->n, control->streams[4] );
    Dot_local_rvec2( workspace, b,
            b, system->n, &redux[0], &redux[1], control->streams[4] );
    Dot_local_rvec2( workspace, workspace->d_workspace->r2,
            workspace->d_workspace->r2, system->n, &redux[2], &redux[3], control->streams[4] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Allreduce( MPI_IN_PLACE, redux, 4, MPI_DOUBLE,
            MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    b_norm[0] = SQRT( redux[0] );
    b_norm[1] = SQRT( redux[1] );
    r_norm[0] = SQRT( redux[2] );
    r_norm[1] = SQRT( redux[3] );
    if ( b_norm[0] == 0.0 )
    {
        b_norm[0] = 1.0;
    }
    if ( b_norm[1] == 0.0 )
    {
        b_norm[1] = 1.0;
    }
    Vector_Copy_rvec2( workspace->d_workspace->r_hat2,
            workspace->d_workspace->r2, system->n, control->streams[4] );
    omega[0] = 1.0;
    omega[1] = 1.0;
    rho[0] = 1.0;
    rho[1] = 1.0;

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    for ( i = 0; i < control->cm_solver_max_iters; ++i )
    {
        if ( r_norm[0] / b_norm[0] <= tol || r_norm[1] / b_norm[1] <= tol )
        {
            break;
        }

        Dot_local_rvec2( workspace, workspace->d_workspace->r_hat2,
                workspace->d_workspace->r2, system->n, &redux[0], &redux[1], control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        rho[0] = redux[0];
        rho[1] = redux[1];
        if ( rho[0] == 0.0 || rho[1] == 0.0 )
        {
            break;
        }
        if ( i > 0 )
        {
            beta[0] = (rho[0] / rho_old[0]) * (alpha[0] / omega[0]);
            beta[1] = (rho[1] / rho_old[1]) * (alpha[1] / omega[1]);
            Vector_Sum_rvec2( workspace->d_workspace->q2,
                    1.0, 1.0, workspace->d_workspace->p2,
                    -1.0 * omega[0], -1.0 * omega[1], workspace->d_workspace->z2, system->n, control->streams[4] );
            Vector_Sum_rvec2( workspace->d_workspace->p2,
                    1.0, 1.0, workspace->d_workspace->r2,
                    beta[0], beta[1], workspace->d_workspace->q2, system->n, control->streams[4] );
        }
        else
        {
            Vector_Copy_rvec2( workspace->d_workspace->p2,
                    workspace->d_workspace->r2, system->n, control->streams[4] );
        }

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        dual_apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->p2, workspace->d_workspace->y2,
                i == 0 ? fresh_pre : FALSE, LEFT );
        dual_apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->y2, workspace->d_workspace->d2,
                i == 0 ? fresh_pre : FALSE, RIGHT );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

        Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
                H, workspace->d_workspace->d2, system->N, workspace->d_workspace->z2 );

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        Dot_local_rvec2( workspace, workspace->d_workspace->r_hat2,
                workspace->d_workspace->z2, system->n, &redux[0], &redux[1], control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        tmp[0] = redux[0];
        tmp[1] = redux[1];
        alpha[0] = rho[0] / tmp[0];
        alpha[1] = rho[1] / tmp[1];
        Vector_Sum_rvec2( workspace->d_workspace->q2,
                1.0, 1.0, workspace->d_workspace->r2,
                -1.0 * alpha[0], -1.0 * alpha[1], workspace->d_workspace->z2, system->n, control->streams[4] );
        Dot_local_rvec2( workspace, workspace->d_workspace->q2,
                workspace->d_workspace->q2, system->n, &redux[0], &redux[1], control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        tmp[0] = redux[0];
        tmp[1] = redux[1];
        /* early convergence check */
        if ( tmp[0] < tol || tmp[1] < tol )
        {
            Vector_Add_rvec2( x, alpha[0], alpha[1], workspace->d_workspace->d2, system->n, control->streams[4] );
            break;
        }

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        dual_apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->q2, workspace->d_workspace->y2, fresh_pre, LEFT );
        dual_apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->y2, workspace->d_workspace->q_hat2, fresh_pre, RIGHT );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

        Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
                H, workspace->d_workspace->q_hat2, system->N, workspace->d_workspace->y2 );

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        Dot_local_rvec2( workspace, workspace->d_workspace->y2,
                workspace->d_workspace->q2, system->n, &redux[0], &redux[1], control->streams[4] );
        Dot_local_rvec2( workspace, workspace->d_workspace->y2,
                workspace->d_workspace->y2, system->n, &redux[2], &redux[3], control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 4, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        sigma[0] = redux[0];
        sigma[1] = redux[1];
        tmp[0] = redux[2];
        tmp[1] = redux[3];
        omega[0] = sigma[0] / tmp[0];
        omega[1] = sigma[1] / tmp[1];
        Vector_Sum_rvec2( workspace->d_workspace->g2,
                alpha[0], alpha[1], workspace->d_workspace->d2,
                omega[0], omega[1], workspace->d_workspace->q_hat2, system->n, control->streams[4] );
        Vector_Add_rvec2( x, 1.0, 1.0, workspace->d_workspace->g2, system->n, control->streams[4] );
        Vector_Sum_rvec2( workspace->d_workspace->r2,
                1.0, 1.0, workspace->d_workspace->q2,
                -1.0 * omega[0], -1.0 * omega[1], workspace->d_workspace->y2, system->n, control->streams[4] );
        Dot_local_rvec2( workspace, workspace->d_workspace->r2,
                workspace->d_workspace->r2, system->n, &redux[0], &redux[1], control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        r_norm[0] = SQRT( redux[0] );
        r_norm[1] = SQRT( redux[1] );
        if ( omega[0] == 0.0 || omega[1] == 0.0 )
        {
            break;
        }
        rho_old[0] = rho[0];
        rho_old[1] = rho[1];

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif
    }

    if ( r_norm[0] / b_norm[0] <= tol
            && r_norm[1] / b_norm[1] > tol )
    {
        Vector_Copy_From_rvec2( workspace->d_workspace->t,
                workspace->d_workspace->x, 1, system->n, control->streams[4] );

        matvecs = Hip_BiCGStab( system, control, data, workspace, H,
                workspace->d_workspace->b_t, tol,
                workspace->d_workspace->t, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->d_workspace->x,
                workspace->d_workspace->t, 1, system->n, control->streams[4] );
    }
    else if ( r_norm[1] / b_norm[1] <= tol
            && r_norm[0] / b_norm[0] > tol )
    {
        Vector_Copy_From_rvec2( workspace->d_workspace->s,
                workspace->d_workspace->x, 0, system->n, control->streams[4] );

        matvecs = Hip_BiCGStab( system, control, data, workspace, H,
                workspace->d_workspace->b_s, tol,
                workspace->d_workspace->s, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->d_workspace->x,
                workspace->d_workspace->s, 0, system->n, control->streams[4] );
    }
    else
    {
        matvecs = 0;
    }

    if ( i >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] p%d: dual BiCGStab convergence failed (%d iters)\n",
                system->my_rank, i );
        fprintf( stderr, "    [INFO] Rel. residual error for s solve: %e\n", r_norm[0] / b_norm[0] );
        fprintf( stderr, "    [INFO] Rel. residual error for t solve: %e\n", r_norm[1] / b_norm[1] );
    }

    return (i + 1) + matvecs;
}


/* Bi-conjugate gradient stabalized method with left preconditioning for
 * solving nonsymmetric linear systems
 *
 * system: 
 * workspace: struct containing storage for workspace for the linear solver
 * control: struct containing parameters governing the simulation and numeric methods
 * data: struct containing simulation data (e.g., atom info)
 * H: sparse, symmetric matrix in CSR format
 * b: right-hand side of the linear system
 * tol: tolerence compared against the relative residual for determining convergence
 * x: inital guess
 * mpi_data: 
 *
 * Reference: Netlib (in MATLAB)
 *  http://www.netlib.org/templates/matlab/bicgstab.m
 * */
int Hip_BiCGStab( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        sparse_matrix const * const H, real const * const b, real tol,
        real * const x, mpi_datatypes * const mpi_data, int fresh_pre )
{
    unsigned int i;
    int ret;
    real tmp, alpha, beta, omega, sigma, rho, rho_old, r_norm, b_norm;
    real redux[2];
#if defined(LOG_PERFORMANCE)
    real time;
#endif

    Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, x, system->N, workspace->d_workspace->d );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum( workspace->d_workspace->r, 1.0, b,
            -1.0, workspace->d_workspace->d, system->n, control->streams[4] );
    redux[0] = Dot_local( workspace, b, b, system->n, control->streams[4] );
    redux[1] = Dot_local( workspace, workspace->d_workspace->r,
            workspace->d_workspace->r, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
            MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    b_norm = SQRT( redux[0] );
    r_norm = SQRT( redux[1] );
    if ( b_norm == 0.0 )
    {
        b_norm = 1.0;
    }
    Vector_Copy( workspace->d_workspace->r_hat,
            workspace->d_workspace->r, system->n, control->streams[4] );
    omega = 1.0;
    rho = 1.0;

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    for ( i = 0; i < control->cm_solver_max_iters && r_norm / b_norm > tol; ++i )
    {
        redux[0] = Dot_local( workspace, workspace->d_workspace->r_hat,
                workspace->d_workspace->r, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 1, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        rho = redux[0];
        if ( rho == 0.0 )
        {
            break;
        }
        if ( i > 0 )
        {
            beta = (rho / rho_old) * (alpha / omega);
            Vector_Sum( workspace->d_workspace->q,
                    1.0, workspace->d_workspace->p,
                    -1.0 * omega, workspace->d_workspace->z, system->n, control->streams[4] );
            Vector_Sum( workspace->d_workspace->p,
                    1.0, workspace->d_workspace->r,
                    beta, workspace->d_workspace->q, system->n, control->streams[4] );
        }
        else
        {
            Vector_Copy( workspace->d_workspace->p,
                    workspace->d_workspace->r, system->n, control->streams[4] );
        }

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->p, workspace->d_workspace->y,
                i == 0 ? fresh_pre : FALSE, LEFT );
        apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->y, workspace->d_workspace->d,
                i == 0 ? fresh_pre : FALSE, RIGHT );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

        Sparse_MatVec( system, control, data, workspace, mpi_data,
                H, workspace->d_workspace->d, system->N, workspace->d_workspace->z );

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        redux[0] = Dot_local( workspace, workspace->d_workspace->r_hat,
                workspace->d_workspace->z, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 1, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        tmp = redux[0];
        alpha = rho / tmp;
        Vector_Sum( workspace->d_workspace->q,
                1.0, workspace->d_workspace->r,
                -1.0 * alpha, workspace->d_workspace->z, system->n, control->streams[4] );
        redux[0] = Dot_local( workspace, workspace->d_workspace->q,
                workspace->d_workspace->q, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 1, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        tmp = redux[0];
        /* early convergence check */
        if ( tmp < tol )
        {
            Vector_Add( x, alpha, workspace->d_workspace->d, system->n, control->streams[4] );
            break;
        }

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->q, workspace->d_workspace->y, fresh_pre, LEFT );
        apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->y, workspace->d_workspace->q_hat, fresh_pre, RIGHT );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

        Sparse_MatVec( system, control, data, workspace, mpi_data,
                H, workspace->d_workspace->q_hat, system->N, workspace->d_workspace->y );

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        redux[0] = Dot_local( workspace, workspace->d_workspace->y,
                workspace->d_workspace->q, system->n, control->streams[4] );
        redux[1] = Dot_local( workspace, workspace->d_workspace->y,
                workspace->d_workspace->y, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        sigma = redux[0];
        tmp = redux[1];
        omega = sigma / tmp;
        Vector_Sum( workspace->d_workspace->g,
                alpha, workspace->d_workspace->d,
                omega, workspace->d_workspace->q_hat, system->n, control->streams[4] );
        Vector_Add( x, 1.0, workspace->d_workspace->g, system->n, control->streams[4] );
        Vector_Sum( workspace->d_workspace->r,
                1.0, workspace->d_workspace->q,
                -1.0 * omega, workspace->d_workspace->y, system->n, control->streams[4] );
        redux[0] = Dot_local( workspace, workspace->d_workspace->r,
                workspace->d_workspace->r, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Allreduce( MPI_IN_PLACE, redux, 1, MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        r_norm = SQRT( redux[0] );
        if ( omega == 0.0 )
        {
            break;
        }
        rho_old = rho;

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif
    }

    if ( i >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] p%d: BiCGStab convergence failed (%d iters)\n",
                system->my_rank, i );
        fprintf( stderr, "    [INFO] Rel. residual error: %e\n", r_norm / b_norm );
    }

    return i;
}


/* Dual iteration for the Pipelined Preconditioned Conjugate Gradient Method
 * for QEq (2 simultaneous solves)
 *
 * References:
 * 1) Hiding global synchronization latency in the preconditioned Conjugate Gradient algorithm,
 *  P. Ghysels and W. Vanroose, Parallel Computing, 2014.
 * 2) Scalable Non-blocking Preconditioned Conjugate Gradient Methods,
 *  Paul R. Eller and William Gropp, SC '16 Proceedings of the International Conference
 *  for High Performance Computing, Networking, Storage and Analysis, 2016.
 *  */
int Hip_dual_PIPECG( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        sparse_matrix const * const H, rvec2 const * const b, real tol,
        rvec2 * const x, mpi_datatypes * const mpi_data, int fresh_pre )
{
    unsigned int i, matvecs;
    int ret;
    rvec2 alpha, beta, delta, gamma_old, gamma_new, r_norm, b_norm;
    real redux[8];
    MPI_Request req;
#if defined(LOG_PERFORMANCE)
    real time;
#endif

    Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, x, system->N, workspace->d_workspace->u2 );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum_rvec2( workspace->d_workspace->r2, 1.0, 1.0, b,
            -1.0, -1.0, workspace->d_workspace->u2, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    dual_apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->r2, workspace->d_workspace->m2, fresh_pre, LEFT );
    dual_apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->m2, workspace->d_workspace->u2, fresh_pre, RIGHT );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

    Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, workspace->d_workspace->u2, system->N, workspace->d_workspace->w2 );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Dot_local_rvec2( workspace, workspace->d_workspace->w2,
            workspace->d_workspace->u2, system->n, &redux[0], &redux[1], control->streams[4] );
    Dot_local_rvec2( workspace, workspace->d_workspace->r2,
            workspace->d_workspace->u2, system->n, &redux[2], &redux[3], control->streams[4] );
    Dot_local_rvec2( workspace, workspace->d_workspace->u2,
            workspace->d_workspace->u2, system->n, &redux[4], &redux[5], control->streams[4] );
    Dot_local_rvec2( workspace, b, b, system->n, &redux[6], &redux[7], control->streams[4] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Iallreduce( MPI_IN_PLACE, redux, 8, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD, &req );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    dual_apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->w2, workspace->d_workspace->n2, FALSE, LEFT );
    dual_apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->n2, workspace->d_workspace->m2, FALSE, RIGHT );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

    Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, workspace->d_workspace->m2, system->N, workspace->d_workspace->n2 );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    ret = MPI_Wait( &req, MPI_STATUS_IGNORE );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    delta[0] = redux[0];
    delta[1] = redux[1];
    gamma_new[0] = redux[2];
    gamma_new[1] = redux[3];
    r_norm[0] = SQRT( redux[4] );
    r_norm[1] = SQRT( redux[5] );
    b_norm[0] = SQRT( redux[6] );
    b_norm[1] = SQRT( redux[7] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    for ( i = 0; i < control->cm_solver_max_iters; ++i )
    {
        if ( r_norm[0] / b_norm[0] <= tol || r_norm[1] / b_norm[1] <= tol )
        {
            break;
        }

        if ( i > 0 )
        {
            beta[0] = gamma_new[0] / gamma_old[0];
            beta[1] = gamma_new[1] / gamma_old[1];
            alpha[0] = gamma_new[0] / (delta[0] - beta[0] / alpha[0] * gamma_new[0]);
            alpha[1] = gamma_new[1] / (delta[1] - beta[1] / alpha[1] * gamma_new[1]);
        }
        else
        {
            beta[0] = 0.0;
            beta[1] = 0.0;
            alpha[0] = gamma_new[0] / delta[0];
            alpha[1] = gamma_new[1] / delta[1];
        }

        Vector_Sum_rvec2( workspace->d_workspace->z2, 1.0, 1.0, workspace->d_workspace->n2,
                beta[0], beta[1], workspace->d_workspace->z2, system->n, control->streams[4] );
        Vector_Sum_rvec2( workspace->d_workspace->q2, 1.0, 1.0, workspace->d_workspace->m2,
                beta[0], beta[1], workspace->d_workspace->q2, system->n, control->streams[4] );
        Vector_Sum_rvec2( workspace->d_workspace->p2, 1.0, 1.0, workspace->d_workspace->u2,
                beta[0], beta[1], workspace->d_workspace->p2, system->n, control->streams[4] );
        Vector_Sum_rvec2( workspace->d_workspace->d2, 1.0, 1.0, workspace->d_workspace->w2,
                beta[0], beta[1], workspace->d_workspace->d2, system->n, control->streams[4] );
        Vector_Sum_rvec2( x, 1.0, 1.0, x,
                alpha[0], alpha[1], workspace->d_workspace->p2, system->n, control->streams[4] );
        Vector_Sum_rvec2( workspace->d_workspace->u2, 1.0, 1.0, workspace->d_workspace->u2,
                -1.0 * alpha[0], -1.0 * alpha[1], workspace->d_workspace->q2, system->n, control->streams[4] );
        Vector_Sum_rvec2( workspace->d_workspace->w2, 1.0, 1.0, workspace->d_workspace->w2,
                -1.0 * alpha[0], -1.0 * alpha[1], workspace->d_workspace->z2, system->n, control->streams[4] );
        Vector_Sum_rvec2( workspace->d_workspace->r2, 1.0, 1.0, workspace->d_workspace->r2,
                -1.0 * alpha[0], -1.0 * alpha[1], workspace->d_workspace->d2, system->n, control->streams[4] );
        Dot_local_rvec2( workspace, workspace->d_workspace->w2,
                workspace->d_workspace->u2, system->n, &redux[0], &redux[1], control->streams[4] );
        Dot_local_rvec2( workspace, workspace->d_workspace->r2,
                workspace->d_workspace->u2, system->n, &redux[2], &redux[3], control->streams[4] );
        Dot_local_rvec2( workspace, workspace->d_workspace->u2,
                workspace->d_workspace->u2, system->n, &redux[4], &redux[5], control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Iallreduce( MPI_IN_PLACE, redux, 6, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD, &req );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        dual_apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->w2, workspace->d_workspace->n2, FALSE, LEFT );
        dual_apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->n2, workspace->d_workspace->m2, FALSE, RIGHT );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

        Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
                H, workspace->d_workspace->m2, system->N, workspace->d_workspace->n2 );

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        gamma_old[0] = gamma_new[0];
        gamma_old[1] = gamma_new[1];

        ret = MPI_Wait( &req, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        delta[0] = redux[0];
        delta[1] = redux[1];
        gamma_new[0] = redux[2];
        gamma_new[1] = redux[3];
        r_norm[0] = SQRT( redux[4] );
        r_norm[1] = SQRT( redux[5] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif
    }

    if ( r_norm[0] / b_norm[0] <= tol
            && r_norm[1] / b_norm[1] > tol )
    {
        Vector_Copy_From_rvec2( workspace->d_workspace->t,
                workspace->d_workspace->x, 1, system->n, control->streams[4] );

        matvecs = Hip_PIPECG( system, control, data, workspace, H,
                workspace->d_workspace->b_t, tol,
                workspace->d_workspace->t, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->d_workspace->x,
                workspace->d_workspace->t, 1, system->n, control->streams[4] );
    }
    else if ( r_norm[1] / b_norm[1] <= tol
            && r_norm[0] / b_norm[0] > tol )
    {
        Vector_Copy_From_rvec2( workspace->d_workspace->s,
                workspace->d_workspace->x, 0, system->n, control->streams[4] );

        matvecs = Hip_PIPECG( system, control, data, workspace, H,
                workspace->d_workspace->b_s, tol,
                workspace->d_workspace->s, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->d_workspace->x,
                workspace->d_workspace->s, 0, system->n, control->streams[4] );
    }
    else
    {
        matvecs = 0;
    }

    if ( i >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] p%d: dual PIPECG convergence failed (%d iters)\n",
                system->my_rank, i );
        fprintf( stderr, "    [INFO] Rel. residual error for s solve: %e\n", r_norm[0] / b_norm[0] );
        fprintf( stderr, "    [INFO] Rel. residual error for t solve: %e\n", r_norm[1] / b_norm[1] );
    }

    return (i + 1) + matvecs;
}


/* Pipelined Preconditioned Conjugate Gradient Method
 *
 * References:
 * 1) Hiding global synchronization latency in the preconditioned Conjugate Gradient algorithm,
 *  P. Ghysels and W. Vanroose, Parallel Computing, 2014.
 * 2) Scalable Non-blocking Preconditioned Conjugate Gradient Methods,
 *  Paul R. Eller and William Gropp, SC '16 Proceedings of the International Conference
 *  for High Performance Computing, Networking, Storage and Analysis, 2016.
 *  */
int Hip_PIPECG( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        sparse_matrix const * const H, real const * const b, real tol,
        real * const x, mpi_datatypes * const mpi_data, int fresh_pre )
{
    unsigned int i;
    int ret;
    real alpha, beta, delta, gamma_old, gamma_new, r_norm, b_norm;
    real redux[4];
    MPI_Request req;
#if defined(LOG_PERFORMANCE)
    real time;
#endif

    Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, x, system->N, workspace->d_workspace->u );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum( workspace->d_workspace->r, 1.0, b,
            -1.0, workspace->d_workspace->u, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->r, workspace->d_workspace->m, fresh_pre, LEFT );
    apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->m, workspace->d_workspace->u, fresh_pre, RIGHT );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

    Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, workspace->d_workspace->u, system->N, workspace->d_workspace->w );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    redux[0] = Dot_local( workspace, workspace->d_workspace->w,
            workspace->d_workspace->u, system->n, control->streams[4] );
    redux[1] = Dot_local( workspace, workspace->d_workspace->r,
            workspace->d_workspace->u, system->n, control->streams[4] );
    redux[2] = Dot_local( workspace, workspace->d_workspace->u,
            workspace->d_workspace->u, system->n, control->streams[4] );
    redux[3] = Dot_local( workspace, b, b, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    ret = MPI_Iallreduce( MPI_IN_PLACE, redux, 4, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD, &req );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->w, workspace->d_workspace->n, FALSE, LEFT );
    apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->n, workspace->d_workspace->m, FALSE, RIGHT );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

    Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, workspace->d_workspace->m, system->N, workspace->d_workspace->n );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    ret = MPI_Wait( &req, MPI_STATUS_IGNORE );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    delta = redux[0];
    gamma_new = redux[1];
    r_norm = SQRT( redux[2] );
    b_norm = SQRT( redux[3] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    for ( i = 0; i < control->cm_solver_max_iters && r_norm / b_norm > tol; ++i )
    {
        if ( i > 0 )
        {
            beta = gamma_new / gamma_old;
            alpha = gamma_new / (delta - beta / alpha * gamma_new);
        }
        else
        {
            beta = 0.0;
            alpha = gamma_new / delta;
        }

        Vector_Sum( workspace->d_workspace->z, 1.0, workspace->d_workspace->n,
                beta, workspace->d_workspace->z, system->n, control->streams[4] );
        Vector_Sum( workspace->d_workspace->q, 1.0, workspace->d_workspace->m,
                beta, workspace->d_workspace->q, system->n, control->streams[4] );
        Vector_Sum( workspace->d_workspace->p, 1.0, workspace->d_workspace->u,
                beta, workspace->d_workspace->p, system->n, control->streams[4] );
        Vector_Sum( workspace->d_workspace->d, 1.0, workspace->d_workspace->w,
                beta, workspace->d_workspace->d, system->n, control->streams[4] );
        Vector_Sum( x, 1.0, x,
                alpha, workspace->d_workspace->p, system->n, control->streams[4] );
        Vector_Sum( workspace->d_workspace->u, 1.0, workspace->d_workspace->u,
                -1.0 * alpha, workspace->d_workspace->q, system->n, control->streams[4] );
        Vector_Sum( workspace->d_workspace->w, 1.0, workspace->d_workspace->w,
                -1.0 * alpha, workspace->d_workspace->z, system->n, control->streams[4] );
        Vector_Sum( workspace->d_workspace->r, 1.0, workspace->d_workspace->r,
                -1.0 * alpha, workspace->d_workspace->d, system->n, control->streams[4] );
        redux[0] = Dot_local( workspace, workspace->d_workspace->w,
                workspace->d_workspace->u, system->n, control->streams[4] );
        redux[1] = Dot_local( workspace, workspace->d_workspace->r,
                workspace->d_workspace->u, system->n, control->streams[4] );
        redux[2] = Dot_local( workspace, workspace->d_workspace->u,
                workspace->d_workspace->u, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Iallreduce( MPI_IN_PLACE, redux, 3, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD, &req );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->w, workspace->d_workspace->n, FALSE, LEFT );
        apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->n, workspace->d_workspace->m, FALSE, RIGHT );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

        Sparse_MatVec( system, control, data, workspace, mpi_data,
                H, workspace->d_workspace->m, system->N, workspace->d_workspace->n );

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        gamma_old = gamma_new;

        ret = MPI_Wait( &req, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        delta = redux[0];
        gamma_new = redux[1];
        r_norm = SQRT( redux[2] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif
    }

    if ( i >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] p%d: PIPECG convergence failed (%d iters)\n",
                system->my_rank, i );
        fprintf( stderr, "    [INFO] Rel. residual error: %e\n", r_norm / b_norm );
    }

    return i;
}


/* Dual iteration for the Pipelined Preconditioned Conjugate Residual Method
 * for QEq (2 simultaneous solves)
 *
 * References:
 * 1) Hiding global synchronization latency in the preconditioned Conjugate Gradient algorithm,
 *  P. Ghysels and W. Vanroose, Parallel Computing, 2014.
 *  */
int Hip_dual_PIPECR( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        sparse_matrix const * const H, rvec2 const * const b, real tol,
        rvec2 * const x, mpi_datatypes * const mpi_data, int fresh_pre )
{
    unsigned int i, matvecs;
    int ret;
    rvec2 alpha, beta, delta, gamma_old, gamma_new, r_norm, b_norm;
    real redux[6];
    MPI_Request req;
#if defined(LOG_PERFORMANCE)
    real time;
#endif

    Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, x, system->N, workspace->d_workspace->u2 );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum_rvec2( workspace->d_workspace->r2, 1.0, 1.0, b,
            -1.0, -1.0, workspace->d_workspace->u2, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    dual_apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->r2, workspace->d_workspace->n2, fresh_pre, LEFT );
    dual_apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->n2, workspace->d_workspace->u2, fresh_pre, RIGHT );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

    Dot_local_rvec2( workspace, b, b, system->n, &redux[0], &redux[1], control->streams[4] );
    Dot_local_rvec2( workspace, workspace->d_workspace->u2,
            workspace->d_workspace->u2, system->n, &redux[2], &redux[3], control->streams[4] );

    ret = MPI_Iallreduce( MPI_IN_PLACE, redux, 4, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD, &req );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, workspace->d_workspace->u2, system->N, workspace->d_workspace->w2 );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    ret = MPI_Wait( &req, MPI_STATUS_IGNORE );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    b_norm[0] = SQRT( redux[0] );
    b_norm[1] = SQRT( redux[1] );
    r_norm[0] = SQRT( redux[2] );
    r_norm[1] = SQRT( redux[3] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    for ( i = 0; i < control->cm_solver_max_iters; ++i )
    {
        if ( r_norm[0] / b_norm[0] <= tol || r_norm[1] / b_norm[1] <= tol )
        {
            break;
        }

        dual_apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->w2, workspace->d_workspace->n2, FALSE, LEFT );
        dual_apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->n2, workspace->d_workspace->m2, FALSE, RIGHT );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

        Dot_local_rvec2( workspace, workspace->d_workspace->w2,
                workspace->d_workspace->u2, system->n, &redux[0], &redux[1], control->streams[4] );
        Dot_local_rvec2( workspace, workspace->d_workspace->m2,
                workspace->d_workspace->w2, system->n, &redux[2], &redux[3], control->streams[4] );
        Dot_local_rvec2( workspace, workspace->d_workspace->u2,
                workspace->d_workspace->u2, system->n, &redux[4], &redux[5], control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Iallreduce( MPI_IN_PLACE, redux, 6, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD, &req );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        Dual_Sparse_MatVec( system, control, data, workspace, mpi_data,
                H, workspace->d_workspace->m2, system->N, workspace->d_workspace->n2 );

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        ret = MPI_Wait( &req, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        gamma_new[0] = redux[0];
        gamma_new[1] = redux[1];
        delta[0] = redux[2];
        delta[1] = redux[3];
        r_norm[0] = SQRT( redux[4] );
        r_norm[1] = SQRT( redux[5] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        if ( i > 0 )
        {
            beta[0] = gamma_new[0] / gamma_old[0];
            beta[1] = gamma_new[1] / gamma_old[1];
            alpha[0] = gamma_new[0] / (delta[0] - beta[0] / alpha[0] * gamma_new[0]);
            alpha[1] = gamma_new[1] / (delta[1] - beta[1] / alpha[1] * gamma_new[1]);
        }
        else
        {
            beta[0] = 0.0;
            beta[1] = 0.0;
            alpha[0] = gamma_new[0] / delta[0];
            alpha[1] = gamma_new[1] / delta[1];
        }

        Vector_Sum_rvec2( workspace->d_workspace->z2, 1.0, 1.0, workspace->d_workspace->n2,
                beta[0], beta[1], workspace->d_workspace->z2, system->n, control->streams[4] );
        Vector_Sum_rvec2( workspace->d_workspace->q2, 1.0, 1.0, workspace->d_workspace->m2,
                beta[0], beta[1], workspace->d_workspace->q2, system->n, control->streams[4] );
        Vector_Sum_rvec2( workspace->d_workspace->p2, 1.0, 1.0, workspace->d_workspace->u2,
                beta[0], beta[1], workspace->d_workspace->p2, system->n, control->streams[4] );
        Vector_Sum_rvec2( workspace->d_workspace->d2, 1.0, 1.0, workspace->d_workspace->w2,
                beta[0], beta[1], workspace->d_workspace->d2, system->n, control->streams[4] );
        Vector_Sum_rvec2( x, 1.0, 1.0, x,
                alpha[0], alpha[1], workspace->d_workspace->p2, system->n, control->streams[4] );
        Vector_Sum_rvec2( workspace->d_workspace->u2, 1.0, 1.0, workspace->d_workspace->u2,
                -1.0 * alpha[0], -1.0 * alpha[1], workspace->d_workspace->q2, system->n, control->streams[4] );
        Vector_Sum_rvec2( workspace->d_workspace->w2, 1.0, 1.0, workspace->d_workspace->w2,
                -1.0 * alpha[0], -1.0 * alpha[1], workspace->d_workspace->z2, system->n, control->streams[4] );
        Vector_Sum_rvec2( workspace->d_workspace->r2, 1.0, 1.0, workspace->d_workspace->r2,
                -1.0 * alpha[0], -1.0 * alpha[1], workspace->d_workspace->d2, system->n, control->streams[4] );

        gamma_old[0] = gamma_new[0];
        gamma_old[1] = gamma_new[1];

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif
    }

    if ( r_norm[0] / b_norm[0] <= tol
            && r_norm[1] / b_norm[1] > tol )
    {
        Vector_Copy_From_rvec2( workspace->d_workspace->t,
                workspace->d_workspace->x, 1, system->n, control->streams[4] );

        matvecs = Hip_PIPECR( system, control, data, workspace, H,
                workspace->d_workspace->b_t, tol,
                workspace->d_workspace->t, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->d_workspace->x,
                workspace->d_workspace->t, 1, system->n, control->streams[4] );
    }
    else if ( r_norm[1] / b_norm[1] <= tol
            && r_norm[0] / b_norm[0] > tol )
    {
        Vector_Copy_From_rvec2( workspace->d_workspace->s,
                workspace->d_workspace->x, 0, system->n, control->streams[4] );

        matvecs = Hip_PIPECR( system, control, data, workspace, H,
                workspace->d_workspace->b_s, tol,
                workspace->d_workspace->s, mpi_data, FALSE );

        Vector_Copy_To_rvec2( workspace->d_workspace->x,
                workspace->d_workspace->s, 0, system->n, control->streams[4] );
    }
    else
    {
        matvecs = 0;
    }

    if ( i >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] p%d: dual PIPECR convergence failed (%d iters)\n",
                system->my_rank, i );
        fprintf( stderr, "    [INFO] Rel. residual error for s solve: %e\n", r_norm[0] / b_norm[0] );
        fprintf( stderr, "    [INFO] Rel. residual error for t solve: %e\n", r_norm[1] / b_norm[1] );
    }

    return (i + 1) + matvecs;
}


/* Pipelined Preconditioned Conjugate Residual Method
 *
 * References:
 * 1) Hiding global synchronization latency in the preconditioned Conjugate Gradient algorithm,
 *  P. Ghysels and W. Vanroose, Parallel Computing, 2014.
 *  */
int Hip_PIPECR( reax_system const * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace,
        sparse_matrix const * const H, real const * const b, real tol,
        real * const x, mpi_datatypes * const mpi_data, int fresh_pre )
{
    unsigned int i;
    int ret;
    real alpha, beta, delta, gamma_old, gamma_new, r_norm, b_norm;
    real redux[3];
    MPI_Request req;
#if defined(LOG_PERFORMANCE)
    real time;
#endif

    Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, x, system->N, workspace->d_workspace->u );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    Vector_Sum( workspace->d_workspace->r, 1.0, b,
            -1.0, workspace->d_workspace->u, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->r, workspace->d_workspace->n, fresh_pre, LEFT );
    apply_preconditioner( system, workspace, control, data, mpi_data,
            workspace->d_workspace->n, workspace->d_workspace->u, fresh_pre, RIGHT );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

    redux[0] = Dot_local( workspace, b, b, system->n, control->streams[4] );
    redux[1] = Dot_local( workspace, workspace->d_workspace->u,
            workspace->d_workspace->u, system->n, control->streams[4] );

    ret = MPI_Iallreduce( MPI_IN_PLACE, redux, 2, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD, &req );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

    Sparse_MatVec( system, control, data, workspace, mpi_data,
            H, workspace->d_workspace->u, system->N, workspace->d_workspace->w );

#if defined(LOG_PERFORMANCE)
    time = Get_Time( );
#endif

    ret = MPI_Wait( &req, MPI_STATUS_IGNORE );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    b_norm = SQRT( redux[0] );
    r_norm = SQRT( redux[1] );

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

    for ( i = 0; i < control->cm_solver_max_iters && r_norm / b_norm > tol; ++i )
    {
        apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->w, workspace->d_workspace->n, FALSE, LEFT );
        apply_preconditioner( system, workspace, control, data, mpi_data,
                workspace->d_workspace->n, workspace->d_workspace->m, FALSE, RIGHT );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_pre_app );
#endif

        redux[0] = Dot_local( workspace, workspace->d_workspace->w,
                workspace->d_workspace->u, system->n, control->streams[4] );
        redux[1] = Dot_local( workspace, workspace->d_workspace->m,
                workspace->d_workspace->w, system->n, control->streams[4] );
        redux[2] = Dot_local( workspace, workspace->d_workspace->u,
                workspace->d_workspace->u, system->n, control->streams[4] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif

        ret = MPI_Iallreduce( MPI_IN_PLACE, redux, 3, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD, &req );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        Sparse_MatVec( system, control, data, workspace, mpi_data,
                H, workspace->d_workspace->m, system->N, workspace->d_workspace->n );

#if defined(LOG_PERFORMANCE)
        time = Get_Time( );
#endif

        ret = MPI_Wait( &req, MPI_STATUS_IGNORE );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
        gamma_new = redux[0];
        delta = redux[1];
        r_norm = SQRT( redux[2] );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_allreduce );
#endif

        if ( i > 0 )
        {
            beta = gamma_new / gamma_old;
            alpha = gamma_new / (delta - beta / alpha * gamma_new);
        }
        else
        {
            beta = 0.0;
            alpha = gamma_new / delta;
        }

        Vector_Sum( workspace->d_workspace->z, 1.0, workspace->d_workspace->n,
                beta, workspace->d_workspace->z, system->n, control->streams[4] );
        Vector_Sum( workspace->d_workspace->q, 1.0, workspace->d_workspace->m,
                beta, workspace->d_workspace->q, system->n, control->streams[4] );
        Vector_Sum( workspace->d_workspace->p, 1.0, workspace->d_workspace->u,
                beta, workspace->d_workspace->p, system->n, control->streams[4] );
        Vector_Sum( workspace->d_workspace->d, 1.0, workspace->d_workspace->w,
                beta, workspace->d_workspace->d, system->n, control->streams[4] );
        Vector_Sum( x, 1.0, x,
                alpha, workspace->d_workspace->p, system->n, control->streams[4] );
        Vector_Sum( workspace->d_workspace->u, 1.0, workspace->d_workspace->u,
                -1.0 * alpha, workspace->d_workspace->q, system->n, control->streams[4] );
        Vector_Sum( workspace->d_workspace->w, 1.0, workspace->d_workspace->w,
                -1.0 * alpha, workspace->d_workspace->z, system->n, control->streams[4] );
        Vector_Sum( workspace->d_workspace->r, 1.0, workspace->d_workspace->r,
                -1.0 * alpha, workspace->d_workspace->d, system->n, control->streams[4] );

        gamma_old = gamma_new;

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm_solver_vector_ops );
#endif
    }

    if ( i >= control->cm_solver_max_iters )
    {
        fprintf( stderr, "[WARNING] p%d: PIPECR convergence failed (%d iters)\n",
                system->my_rank, i );
        fprintf( stderr, "    [INFO] Rel. residual error: %e\n", r_norm / b_norm );
    }

    return i;
}
