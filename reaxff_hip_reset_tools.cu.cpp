
#if defined(LAMMPS_REAX)
    #include "reaxff_hip_reset_tools.h"

    #include "reaxff_hip_list.h"
    #include "reaxff_hip_utils.h"
    #include "reaxff_hip_reduction.h"

    #include "reaxff_reset_tools.h"
    #include "reaxff_vector.h"
#else
    #include "hip_reset_tools.h"

    #include "hip_list.h"
    #include "hip_utils.h"
    #include "hip_reduction.h"

    #include "../reset_tools.h"
    #include "../vector.h"
#endif

#include "hip/hip_runtime.h"

HIP_GLOBAL void k_reset_workspace( storage workspace, int N )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    workspace.CdDelta[i] = 0.0;
    rvec_MakeZero( workspace.f[i] );
}


HIP_GLOBAL void k_reset_hindex( reax_atom *my_atoms, single_body_parameters *sbp,
        int * hindex, int N )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    my_atoms[i].Hindex = i;

    if ( sbp[ my_atoms[i].type ].p_hbond == H_ATOM
            || sbp[ my_atoms[i].type ].p_hbond == H_BONDING_ATOM )
    {
#if !defined(HIP_ACCUM_ATOMIC)
        hindex[i] = 1;
    }
    else
    {
        hindex[i] = 0;
    }
#else
        atomicAdd( hindex, 1 );
    }
#endif
}

void Hip_Reset_Workspace( reax_system *system, control_params *control,
        storage *workspace )
{
    int blocks;

    blocks = system->total_cap / DEF_BLOCK_SIZE
        + ((system->total_cap % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);

    k_reset_workspace <<< blocks, DEF_BLOCK_SIZE, 0, control->streams[0] >>>
        ( *(workspace->d_workspace), system->total_cap );
    hipCheckError( );
}


void Hip_Reset_Atoms_HBond_Indices( reax_system* system, control_params *control,
        storage *workspace )
{
#if !defined(HIP_ACCUM_ATOMIC)
    int *hindex;

    sHipCheckMalloc( &workspace->scratch[0], &workspace->scratch_size[0],
            sizeof(int) * system->total_cap, __FILE__, __LINE__ );
    hindex = (int *) workspace->scratch[0];
#else
    sHipMemsetAsync( system->d_numH, 0, sizeof(int),
            control->streams[0], __FILE__, __LINE__ );
#endif

    k_reset_hindex <<< control->blocks_n, control->block_size_n, 0,
                   control->streams[0] >>>
        ( system->d_my_atoms, system->reax_param.d_sbp,
#if !defined(HIP_ACCUM_ATOMIC)
          hindex, 
#else
          system->d_numH,
#endif
          system->total_cap );
    hipCheckError( );

#if !defined(HIP_ACCUM_ATOMIC)
    Hip_Reduction_Sum( hindex, system->d_numH, system->N, 0, control->streams[0] );
#endif

    sHipMemcpyAsync( &system->numH, system->d_numH, sizeof(int),
            hipMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );

    hipStreamSynchronize( control->streams[0] );
}


extern "C" void Hip_Reset( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists )
{
    Hip_Reset_Atoms_HBond_Indices( system, control, workspace );

    Reset_Simulation_Data( data );

    if ( control->virial )
    {
        Reset_Pressures( data );
    }

    Hip_Reset_Workspace( system, control, workspace );
}
