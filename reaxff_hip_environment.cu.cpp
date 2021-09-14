#if defined(PURE_REAX)
    #include "hip_environment.h"

    #include "hip_utils.h"
#elif defined(LAMMPS_REAX)
    #include "reaxff_hip_environment.h"

    #include "reaxff_hip_utils.h"
#endif


static void compute_blocks( int *blocks, int *block_size, int threads )
{
    *block_size = DEF_BLOCK_SIZE; // threads per block
    *blocks = (threads + (DEF_BLOCK_SIZE - 1)) / DEF_BLOCK_SIZE; // blocks per grid
}


static void compute_nearest_multiple_warpsize( int blocks, int *result )
{
    *result = ((blocks + warpSize - 1) / warpSize) * warpSize;
}


extern "C" void Hip_Setup_Environment( int rank, int nprocs, int gpus_per_node )
{

    int deviceCount;
    hipError_t ret;
    
    ret = hipGetDeviceCount( &deviceCount );

    if ( ret != hipSuccess || deviceCount < 1 )
    {
        fprintf( stderr, "[ERROR] no HIP capable device(s) found. Terminating...\n" );
        exit( CANNOT_INITIALIZE );
    }
    else if ( deviceCount < gpus_per_node || gpus_per_node < 1 )
    {
        fprintf( stderr, "[ERROR] invalid number of HIP capable devices requested (gpus_per_node = %d). Terminating...\n",
                gpus_per_node );
        exit( INVALID_INPUT );
    }

    /* assign the GPU for each process */
    //TODO: handle condition where # CPU procs > # GPUs
    ret = hipSetDevice( rank % gpus_per_node );

    if ( ret == hipErrorInvalidDevice )
    {
        fprintf( stderr, "[ERROR] invalid HIP device ID set (%d). Terminating...\n",
              rank % gpus_per_node );
        exit( CANNOT_INITIALIZE );
    }
    else if ( ret == hipErrorContextAlreadyInUse )
    {
        fprintf( stderr, "[ERROR] HIP device with specified ID already in use (%d). Terminating...\n",
                rank % gpus_per_node );
        exit( CANNOT_INITIALIZE );
    }

    //TODO: revisit additional device configurations
//    hipDeviceSetLimit( cudaLimitStackSize, 8192 );
//    hipDeviceSetCacheConfig( hipFuncCachePreferL1 );
}


extern "C" void Hip_Init_Block_Sizes( reax_system *system,
        control_params *control )
{
    compute_blocks( &control->blocks, &control->block_size, system->n );
    compute_nearest_multiple_warpsize( control->blocks, &control->blocks_pow_2 );

    compute_blocks( &control->blocks_n, &control->block_size_n, system->N );
    compute_nearest_multiple_warpsize( control->blocks_n, &control->blocks_pow_2_n );
}


extern "C" void Hip_Cleanup_Environment( )
{
    hipDeviceReset( );
    hipDeviceSynchronize( );
}
