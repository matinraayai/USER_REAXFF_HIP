#if defined(LAMMPS_REAX)
    #include "reaxff_hip_environment.h"

    #include "reaxff_hip_utils.h"
#else
    #include "hip_environment.h"

    #include "hip_utils.h"
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


extern "C" void Hip_Setup_Environment( reax_system const * const system,
        control_params * const control )
{
    int i, least_priority, greatest_priority, is_stream_priority_supported;
    int deviceCount;
    hipError_t ret;
    
    ret = hipGetDeviceCount( &deviceCount );

    if ( ret != hipSuccess || deviceCount < 1 )
    {
        fprintf( stderr, "[ERROR] no HIP capable device(s) found. Terminating...\n" );
        exit( CANNOT_INITIALIZE );
    }
    else if ( deviceCount < control->gpus_per_node || control->gpus_per_node < 1 )
    {
        fprintf( stderr, "[ERROR] invalid number of HIP capable devices requested (gpus_per_node = %d). Terminating...\n",
                control->gpus_per_node );
        exit( INVALID_INPUT );
    }

    /* assign the GPU for each process */
    //TODO: handle condition where # CPU procs > # GPUs
    ret = hipSetDevice( system->my_rank % control->gpus_per_node );

    if ( ret == hipErrorInvalidDevice )
    {
        fprintf( stderr, "[ERROR] invalid HIP device ID set (%d). Terminating...\n",
              system->my_rank % control->gpus_per_node );
        exit( CANNOT_INITIALIZE );
    }
    else if ( ret == hipErrorContextAlreadyInUse )
    {
        fprintf( stderr, "[ERROR] HIP device with specified ID already in use (%d). Terminating...\n",
                system->my_rank % control->gpus_per_node );
        exit( CANNOT_INITIALIZE );
    }
#if defined(__HIP_PLATFORM_NVCC__)
    ret = cudaDeviceGetAttribute( &is_stream_priority_supported,
            cudaDevAttrStreamPrioritiesSupported,
            system->my_rank % control->gpus_per_node );

    if ( ret != cudaSuccess )
    {
        fprintf( stderr, "[ERROR] cudaDeviceGetAttribute failure. Terminating...\n" );
        exit( CANNOT_INITIALIZE );
    }
#else
    // For now assume by default stream priorities are supported on AMD devices
    // TODO: change this when a fix is available
    ret = hipSuccess;
    is_stream_priority_supported = 1;
#endif

    if ( is_stream_priority_supported == 1 )
    {
        ret = hipDeviceGetStreamPriorityRange( &least_priority, &greatest_priority );
    
        if ( ret != hipSuccess )
        {
            fprintf( stderr, "[ERROR] HIP stream priority query failed. Terminating...\n" );
            exit( CANNOT_INITIALIZE );
        }
    
        /* stream assignment (default to 0 for any kernel not listed):
         * 0: init dist, bond order (uncorrected/corrected), lone pair/over coord/under coord
         * 1: (after init dist) init bonds, (after bond order) bonds, valence angles, torsions
         * 2: (after init dist) init hbonds, (after bonds) hbonds
         * 3: (after init dist) van der Waals
         * 4: init CM, CM, Coulomb
         */
        for ( i = MAX_HIP_STREAMS - 1; i >= 0; --i )
        {
            if ( MAX_HIP_STREAMS - 1 + -1 * i < control->gpu_streams )
            {
                /* all non-CM streams of equal priority */
                if ( i != MAX_HIP_STREAMS - 1 )
                {
                    ret = hipStreamCreateWithPriority( &control->streams[i], hipStreamNonBlocking, least_priority );
                }
                /* CM gets highest priority due to MPI comms and hipMemcpy's */
                else
                {
                    ret = hipStreamCreateWithPriority( &control->streams[i], hipStreamNonBlocking, greatest_priority );
                }
        
                if ( ret != hipSuccess )
                {
                    fprintf( stderr, "[ERROR] hipStreamCreateWithPriority failure (%d). Terminating...\n",
                            i );
                    exit( CANNOT_INITIALIZE );
                }
            }
            else
            {
                control->streams[i] = control->streams[MAX_HIP_STREAMS - 1 - ((MAX_HIP_STREAMS - 1 + -1 * i) % control->gpu_streams)];
            }
        }
    }
    else
    {
        /* stream assignment (default to 0 for any kernel not listed):
         * 0: init dist, bond order (uncorrected/corrected), lone pair/over coord/under coord
         * 1: (after init dist) init bonds, (after bond order) bonds, valence angles, torsions
         * 2: (after init dist) init hbonds, (after bonds) hbonds
         * 3: (after init dist) van der Waals
         * 4: init CM, CM, Coulomb
         */
        for ( i = MAX_HIP_STREAMS - 1; i >= 0; --i )
        {
            if ( MAX_HIP_STREAMS - 1 + -1 * i < control->gpu_streams )
            {
                ret = hipStreamCreateWithFlags( &control->streams[i], hipStreamNonBlocking );
        
                if ( ret != hipSuccess )
                {
                    fprintf( stderr, "[ERROR] hipStreamCreateWithFlags failure (%d). Terminating...\n",
                            i );
                    exit( CANNOT_INITIALIZE );
                }
            }
            else
            {
                control->streams[i] = control->streams[MAX_HIP_STREAMS - 1 - ((MAX_HIP_STREAMS - 1 + -1 * i) % control->gpu_streams)];
            }
       }
    }

    /* stream event assignment:
     * 0: init dist done (stream 0)
     * 1: init bonds done (stream 1)
     * 2: bond orders done (stream 0)
     * 3: bonds done (stream 1)
     */
    for ( i = 0; i < MAX_HIP_STREAM_EVENTS; ++i )
    {
        ret = hipEventCreateWithFlags( &control->stream_events[i], hipEventDisableTiming );

        if ( ret != hipSuccess )
        {
            fprintf( stderr, "[ERROR] hipEventCreateWithFlags failure (%d). Terminating...\n",
                    i );
            exit( CANNOT_INITIALIZE );
        }
    }

    //TODO: revisit additional device configurations
//    hipDeviceSetLimit( hipLimitStackSize, 8192 );
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


extern "C" void Hip_Cleanup_Environment( control_params const * const control )
{
    int i;
    hipError_t ret;

    for ( i = MAX_HIP_STREAMS - 1; i >= 0; --i )
    {
        if ( MAX_HIP_STREAMS - 1 + -1 * i < control->gpu_streams )
        {
            ret = hipStreamDestroy( control->streams[i] );
    
            if ( ret != hipSuccess )
            {
                fprintf( stderr, "[ERROR] HIP stream destruction failed (%d). Terminating...\n",
                        i );
                exit( CANNOT_INITIALIZE );
            }
        }
    }
}


extern "C" void Hip_Print_Mem_Usage( simulation_data const * const data )
{
    int rank;
    size_t total, free;
    hipError_t ret;

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    ret = hipMemGetInfo( &free, &total );

    if ( ret != hipSuccess )
    {
        fprintf( stderr,
                "[WARNING] could not get message usage info from device\n"
                "    [INFO] HIP API error code: %d\n",
                ret );
        return;
    }

    fprintf( stderr, "[INFO] step %d on MPI processor %d, Total: %zu bytes (%7.2f MB) Free %zu bytes (%7.2f MB)\n", 
            data->step, rank,
            total, (long long int) total / (1024.0 * 1024.0),
            free, (long long int) free / (1024.0 * 1024.0) );
    fflush( stderr );
}
