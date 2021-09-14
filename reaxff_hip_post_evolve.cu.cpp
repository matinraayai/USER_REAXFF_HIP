#include "hip/hip_runtime.h"

#include "reaxff_hip_post_evolve.h"

#include "reaxff_hip_utils.h"

#include "reaxff_vector.h"


/* remove translation and rotational terms from center of mass velocities */
HIP_GLOBAL void k_remove_center_of_mass_velocities( reax_atom *my_atoms,
        simulation_data *data, int n )
{
    int i;
    rvec diff, cross;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    /* remove translational term */
    rvec_ScaledAdd( my_atoms[i].v, -1.0, data->vcm );

    /* remove rotational term */
    rvec_ScaledSum( diff, 1.0, my_atoms[i].x, -1.0, data->xcm );
    rvec_Cross( cross, data->avcm, diff );
    rvec_ScaledAdd( my_atoms[i].v, -1.0, cross );
}


extern "C" void Hip_Remove_CoM_Velocities( reax_system *system,
        control_params *control, simulation_data *data )
{
    hipLaunchKernelGGL(k_remove_center_of_mass_velocities, dim3(control->blocks), dim3(control->block_size ), 0, 0,  system->d_my_atoms, (simulation_data *)data->d_simulation_data, system->n );
    hipCheckError( );
}
