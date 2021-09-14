#ifndef __CUDA_UTILS_H_
#define __CUDA_UTILS_H_

#if defined(PURE_REAX)
    #include "../reax_types.h"
#elif defined(LAMMPS_REAX)
    #include "reaxff_types.h"
#endif



void hip_malloc(void **, size_t, int, const char * );

void hip_free( void *, const char * );

void hip_memset( void *, int , size_t , const char * );

void hip_check_malloc( void **, size_t *, size_t, const char * );

void sHipMemcpy( void * const, void const * const, size_t,
        enum hipMemcpyKind, const char * const, int );

void Hip_Print_Mem_Usage( );


#define hipCheckError() __hipCheckError( __FILE__, __LINE__ )
static inline void __hipCheckError( const char *file, const int line )
{
    hipError_t err;
#if defined(DEBUG_FOCUS)
    int rank;

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    fprintf( stderr, "[INFO] hipCheckError: p%d, file %.*s, line %d\n", rank, (int) strlen(file), file, line );
    fflush( stderr );
#endif

#if defined(DEBUG)
    /* Block until tasks in stream are complete in order to enable
     * more pinpointed error checking. However, this will affect performance. */
    err = hipDeviceSynchronize( );
    if ( hipSuccess != err )
    {
        fprintf( stderr, "[ERROR] runtime error encountered with hipDeviceSynchronize( ) at: %s:%d\n", file, line );
        fprintf( stderr, "    [INFO] HIP API error code: %d\n", err );
        fprintf( stderr, "    [INFO] HIP API error name: %s\n", hipGetErrorName( err ) );
        fprintf( stderr, "    [INFO] HIP API error text: %s\n", hipGetErrorString( err ) );
        exit( RUNTIME_ERROR );
    }
#endif

    err = hipPeekAtLastError( );
    if ( hipSuccess != err )
    {
        fprintf( stderr, "[ERROR] runtime error encountered: %s:%d\n", file, line );
        fprintf( stderr, "    [INFO] HIP API error code: %d\n", err );
        fprintf( stderr, "    [INFO] HIP API error name: %s\n", hipGetErrorName( err ) );
        fprintf( stderr, "    [INFO] HIP API error text: %s\n", hipGetErrorString( err ) );
#if !defined(DEBUG)
        fprintf( stderr, "    [WARNING] HIP error info may not be precise due to async nature of HIP kernels!"
               " Rebuild in debug mode to get more accurate accounts of errors (--enable-debug=yes with configure script).\n" );
#endif
        exit( RUNTIME_ERROR );
    }
}


#endif
