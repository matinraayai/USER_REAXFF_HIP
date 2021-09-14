#include "hip/hip_runtime.h"

#include "reaxff_hip_reset_tools.h"

#include "reaxff_hip_list.h"
#include "reaxff_hip_utils.h"
#include "reaxff_hip_reduction.h"

#include "reaxff_reset_tools.h"
#include "reaxff_vector.h"


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
#if !defined(CUDA_ACCUM_ATOMIC)
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

void Hip_Reset_Workspace( reax_system *system, storage *workspace )
{
    int blocks;

    blocks = system->total_cap / DEF_BLOCK_SIZE
        + ((system->total_cap % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);

    hipLaunchKernelGGL(k_reset_workspace, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  *(workspace->d_workspace), system->total_cap );
    hipCheckError( );
}


void Hip_Reset_Atoms_HBond_Indices( reax_system* system, control_params *control,
        storage *workspace )
{
#if !defined(CUDA_ACCUM_ATOMIC)
    int *hindex;

    hip_check_malloc( &workspace->scratch, &workspace->scratch_size,
            sizeof(int) * system->total_cap,
            "Hip_Reset_Atoms_HBond_Indices::workspace->scratch" );
    hindex = (int *) workspace->scratch;
#endif

    hipLaunchKernelGGL(k_reset_hindex, dim3(control->blocks_n), dim3(control->block_size_n ), 0, 0,  system->d_my_atoms, system->reax_param.d_sbp, 
#if !defined(CUDA_ACCUM_ATOMIC)
          hindex, 
#else
          system->d_numH,
#endif
          system->total_cap );
    hipCheckError( );

#if !defined(CUDA_ACCUM_ATOMIC)
    Hip_Reduction_Sum( hindex, system->d_numH, system->N );
#endif

    sHipMemcpy( &system->numH, system->d_numH, sizeof(int),
            hipMemcpyDeviceToHost, __FILE__, __LINE__ );
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

    Hip_Reset_Workspace( system, workspace );
}
