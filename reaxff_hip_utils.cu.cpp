#if defined(PURE_REAX)
    #include "hip_utils.h"
#elif defined(LAMMPS_REAX)
    #include "reaxff_hip_utils.h"
#endif


void hip_malloc( void **ptr, size_t size, int mem_set, const char *msg )
{

    hipError_t retVal = hipSuccess;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] requesting %zu bytes for %s\n",
            size, msg );
    fflush( stderr );
#endif

    retVal = hipMalloc( ptr, size );

    if ( retVal != hipSuccess )
    {
        fprintf( stderr, "[ERROR] failed to allocate memory on device for resouce %s\n", msg );
        fprintf( stderr, "    [INFO] HIP API error code: %d, requested memory size (in bytes): %lu\n",
                retVal, size );
        exit( INSUFFICIENT_MEMORY );
    }  

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] granted memory at address: %p\n", *ptr );
    fflush( stderr );
#endif

    if ( mem_set == TRUE )
    {
        retVal = hipMemset( *ptr, 0, size );

        if( retVal != hipSuccess )
        {
            fprintf( stderr, "[ERROR] failed to memset memory on device for resource %s\n", msg );
            fprintf( stderr, "    [INFO] HIP API error code: %d, requested memory size (in bytes): %lu\n",
                    retVal, size );
            exit( INSUFFICIENT_MEMORY );
        }
    }  
}


void hip_free( void *ptr, const char *msg )
{

    hipError_t retVal = hipSuccess;

    if ( !ptr )
    {
        return;
    }  

    retVal = hipFree( ptr );

    if( retVal != hipSuccess )
    {
        fprintf( stderr, "[WARNING] failed to release memory on device for resource %s\n",
                msg );
        fprintf( stderr, "    [INFO] HIP API error code: %d, memory address: %ld\n",
                retVal, (long int) ptr );
        return;
    }  
}


void hip_memset( void *ptr, int data, size_t count, const char *msg )
{
    hipError_t retVal = hipSuccess;

    retVal = hipMemset( ptr, data, count );

    if( retVal != hipSuccess )
    {
        fprintf( stderr, "[ERROR] failed to memset memory on device for resource %s\n", msg );
        fprintf( stderr, "    [INFO] HIP API error code: %d\n", retVal );
        exit( RUNTIME_ERROR );
    }
}


/* Checks if the amount of space currently allocated to ptr is sufficient,
 * and, if not, frees any space allocated to ptr before allocating the
 * requested amount of space */
void hip_check_malloc( void **ptr, size_t *cur_size, size_t new_size, const char *msg )
{
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] requesting %zu bytes for %s (%zu currently allocated)\n",
            new_size, msg, *cur_size );
    fflush( stderr );
#endif

    assert( new_size > 0 );

    if ( new_size > *cur_size )
    {
        if ( *cur_size > 0 || *ptr != NULL )
        {
            hip_free( *ptr, msg );
        }

        //TODO: look into using aligned alloc's
        /* intentionally over-allocate to reduce the number of allocation operations,
         * and record the new allocation size */
        *cur_size = (size_t) CEIL( new_size * SAFE_ZONE );
        hip_malloc( ptr, *cur_size, 0, msg );
    }
}


/* Safe wrapper around hipMemcpy
 *
 * dest: address to be copied to
 * src: address to be copied from
 * size: num. bytes to copy
 * dir: HIP enum specifying address types for dest and src
 * filename: NULL-terminated source filename where function call originated
 * line: line of source filen where function call originated
 */
void sHipMemcpy( void * const dest, void const * const src, size_t size,
        hipMemcpyKind dir, const char * const filename, int line )
{
    int rank;
    hipError_t ret;

    ret = hipMemcpy( dest, src, size, dir );

    if ( ret != hipSuccess )
    {
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        const char *str = hipGetErrorString( ret );

        fprintf( stderr, "[ERROR] HIP error: memory copy failure\n" );
        fprintf( stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank );
        fprintf( stderr, "  [INFO] Error code: %d\n", ret );
        fprintf( stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str );

        MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
    }
}


void Hip_Print_Mem_Usage( )
{
    size_t total, free;
    hipError_t retVal;

    retVal = hipMemGetInfo( &free, &total );

    if ( retVal != hipSuccess )
    {
        fprintf( stderr,
                "[WARNING] could not get message usage info from device\n"
                "    [INFO] HIP API error code: %d\n",
                retVal );
        return;
    }

    fprintf( stderr, "Total: %zu bytes (%7.2f MB)\nFree %zu bytes (%7.2f MB)\n", 
            total, (long long int) total / (1024.0 * 1024.0),
            free, (long long int) free / (1024.0 * 1024.0) );
}
