
#if defined(LAMMPS_REAX)
    #include "reaxff_hip_init_md.h"

    #include "reaxff_hip_allocate.h"
    #include "reaxff_hip_list.h"
    #include "reaxff_hip_copy.h"
    #include "reaxff_hip_environment.h"
    #include "reaxff_hip_forces.h"
    #include "reaxff_hip_integrate.h"
    #include "reaxff_hip_neighbors.h"
    #include "reaxff_hip_reset_tools.h"
    #include "reaxff_hip_system_props.h"

    #include "reaxff_box.h"
    #include "reaxff_comm_tools.h"
    #include "reaxff_grid.h"
    #include "reaxff_init_md.h"
    #include "reaxff_io_tools.h"
    #include "reaxff_lookup.h"
    #include "reaxff_random.h"
    #include "reaxff_reset_tools.h"
    #include "reaxff_tool_box.h"
    #include "reaxff_vector.h"
#else
    #include "hip_init_md.h"

    #include "hip_allocate.h"
    #include "hip_list.h"
    #include "hip_copy.h"
    #include "hip_environment.h"
    #include "hip_forces.h"
    #include "hip_integrate.h"
    #include "hip_neighbors.h"
    #include "hip_reset_tools.h"
    #include "hip_system_props.h"

    #include "../box.h"
    #include "../comm_tools.h"
    #include "../grid.h"
    #include "../init_md.h"
    #include "../io_tools.h"
    #include "../lookup.h"
    #include "../random.h"
    #include "../reset_tools.h"
    #include "../tool_box.h"
    #include "../vector.h"
#endif


static void Hip_Init_System( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, mpi_datatypes *mpi_data )
{
    Setup_New_Grid( system, control, MPI_COMM_WORLD );

    /* since all processors read in all atoms and select their local atoms
     * intially, no local atoms comm needed and just bin local atoms */
    Bin_My_Atoms( system, workspace );
    Reorder_My_Atoms( system, workspace );

    system->N = SendRecv( system, mpi_data, mpi_data->boundary_atom_type,
            &Count_Boundary_Atoms, &Sort_Boundary_Atoms,
            &Unpack_Exchange_Message, TRUE );

    system->local_cap = MAX( (int) CEIL( system->n * SAFE_ZONE ), MIN_CAP );
    system->total_cap = MAX( (int) CEIL( system->N * SAFE_ZONE ), MIN_CAP );

    system->total_far_nbrs = 0;
    system->total_bonds = 0;
    system->total_hbonds = 0;
    system->total_cm_entries = 0;
    system->total_thbodies = 0;

    Bin_Boundary_Atoms( system );

    Hip_Init_Block_Sizes( system, control );

    Hip_Allocate_System( system, control );
    Hip_Copy_System_Host_to_Device( system, control );

    Hip_Reset_Atoms_HBond_Indices( system, control, workspace );

    Hip_Compute_Total_Mass( system, control, workspace,
            data, mpi_data->comm_mesh3D );

    Hip_Compute_Center_of_Mass( system, control, workspace,
            data, mpi_data, mpi_data->comm_mesh3D );

//    Reposition_Atoms( system, control, data, mpi_data );

    /* initialize velocities so that desired init T can be attained */
    if ( !control->restart || (control->restart && control->random_vel) )
    {
        Hip_Generate_Initial_Velocities( system, control, control->T_init );
    }

    Hip_Compute_Kinetic_Energy( system, control, workspace,
            data, mpi_data->comm_mesh3D );
}


void Hip_Init_Simulation_Data( reax_system *system, control_params *control,
        simulation_data *data )
{
    Hip_Allocate_Simulation_Data( data, control->streams[0] );

    Reset_Simulation_Data( data );
    Reset_Timing( &data->timing );

    if ( !control->restart )
    {
        data->step = 0;
        data->prev_steps = 0;
    }

    switch ( control->ensemble )
    {
    case NVE:
        data->N_f = 3 * system->bigN;
        control->Hip_Evolve = &Hip_Velocity_Verlet_NVE;
        control->virial = 0;
        break;

    case bNVT:
        data->N_f = 3 * system->bigN + 1;
        control->Hip_Evolve = &Hip_Velocity_Verlet_Berendsen_NVT;
        control->virial = 0;
        break;

    case nhNVT:
        fprintf( stderr, "[WARNING] Nose-Hoover NVT is still under testing.\n" );
        data->N_f = 3 * system->bigN + 1;
        control->Hip_Evolve = &Hip_Velocity_Verlet_Nose_Hoover_NVT_Klein;
        control->virial = 0;
        if ( !control->restart || (control->restart && control->random_vel) )
        {
            data->therm.G_xi = control->Tau_T
                * (2.0 * data->sys_en.e_kin - data->N_f * K_B * control->T );
            data->therm.v_xi = data->therm.G_xi * control->dt;
            data->therm.v_xi_old = 0;
            data->therm.xi = 0;
        }
        break;

    /* Semi-Isotropic NPT */
    case sNPT:
        data->N_f = 3 * system->bigN + 4;
        control->Hip_Evolve = &Hip_Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
        {
            Reset_Pressures( data );
        }
        break;

    /* Isotropic NPT */
    case iNPT:
        data->N_f = 3 * system->bigN + 2;
        control->Hip_Evolve = &Hip_Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
        {
            Reset_Pressures( data );
        }
        break;

    /* Anisotropic NPT */
    case NPT:
        data->N_f = 3 * system->bigN + 9;
        control->Hip_Evolve = &Hip_Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;

        fprintf( stderr, "[ERROR] Anisotropic NPT ensemble not yet implemented\n" );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
        break;

    default:
        fprintf( stderr, "[ERROR] p%d: Init_Simulation_Data: ensemble not recognized\n",
              system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
        break;
    }
}


void Hip_Init_Workspace( reax_system *system, control_params *control,
        storage *workspace, mpi_datatypes *mpi_data )
{
    Hip_Allocate_Workspace_Part1( system, control, workspace->d_workspace,
            system->local_cap );
    Hip_Allocate_Workspace_Part2( system, control, workspace->d_workspace,
            system->total_cap );

    workspace->realloc.far_nbrs = FALSE;
    workspace->realloc.cm = FALSE;
    workspace->realloc.bonds = FALSE;
    workspace->realloc.hbonds = FALSE;
    workspace->realloc.thbody = FALSE;
    workspace->realloc.gcell_atoms = 0;

    workspace->d_workspace->realloc.far_nbrs = FALSE;
    workspace->d_workspace->realloc.cm = FALSE;
    workspace->d_workspace->realloc.bonds = FALSE;
    workspace->d_workspace->realloc.hbonds = FALSE;
    workspace->d_workspace->realloc.thbody = FALSE;
    workspace->d_workspace->realloc.gcell_atoms = 0;

    if ( control->cm_solver_pre_comp_type == SAI_PC )
    {
        workspace->H.allocated = FALSE;
        workspace->H_full.allocated = FALSE;
        workspace->H_spar_patt.allocated = FALSE;
        workspace->H_spar_patt_full.allocated = FALSE;
        workspace->H_app_inv.allocated = FALSE;
        workspace->d_workspace->H_full.allocated = FALSE;
        workspace->d_workspace->H_spar_patt.allocated = FALSE;
        workspace->d_workspace->H_spar_patt_full.allocated = FALSE;
        workspace->d_workspace->H_app_inv.allocated = FALSE;
    }

    Hip_Reset_Workspace( system, control, workspace );

    Init_Taper( control, workspace->d_workspace, mpi_data );
}


void Hip_Init_Lists( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        mpi_datatypes *mpi_data )
{
    Hip_Estimate_Num_Neighbors( system, control, data );

    Hip_Make_List( system->total_cap, system->total_far_nbrs,
            TYP_FAR_NEIGHBOR, lists[FAR_NBRS] );
    Hip_Init_Neighbor_Indices( system, control, lists[FAR_NBRS] );

    Hip_Generate_Neighbor_Lists( system, control, data, workspace, lists );

    /* first call to Hip_Estimate_Storages requires
     * setting these manually before allocation */
    workspace->d_workspace->H.n = system->n;
    workspace->d_workspace->H.n_max = system->local_cap;
    workspace->d_workspace->H.format = SYM_FULL_MATRIX;

    /* estimate storage for bonds, hbonds, and sparse matrix */
    Hip_Estimate_Storages( system, control, workspace, lists,
            TRUE, TRUE, TRUE, data->step - data->prev_steps );

    Hip_Allocate_Matrix( &workspace->d_workspace->H, system->n,
            system->local_cap, system->total_cm_entries, SYM_FULL_MATRIX,
            control->streams[0] );
    Hip_Init_Sparse_Matrix_Indices( system, &workspace->d_workspace->H,
           control->streams[0] );

    Hip_Make_List( system->total_cap, system->total_bonds,
            TYP_BOND, lists[BONDS] );
    Hip_Init_Bond_Indices( system, lists[BONDS], control->streams[0] );

    if ( control->hbond_cut > 0.0 && system->numH > 0 )
    {
        Hip_Make_List( system->total_cap, system->total_hbonds,
                TYP_HBOND, lists[HBONDS] );
        Hip_Init_HBond_Indices( system, workspace, lists[HBONDS],
                control->streams[0] );
    }

    /* 3bodies list: since a more accurate estimate of the num.
     * three body interactions requires that bond orders have
     * been computed, delay estimation until computation */
}


extern "C" void Hip_Initialize( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    int i;

    Init_MPI_Datatypes( system, workspace, mpi_data );

#if defined(HIP_DEVICE_PACK) & defined(__HIP_PLATFORM_NVCC__)
    if ( MPIX_Query_cuda_support( ) != 1 )
    {
        fprintf( stderr, "[ERROR] HIP device-side MPI buffer packing/unpacking enabled "
                "but no HIP-aware support detected. Terminating...\n" );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
    }

    mpi_data->d_in1_buffer = NULL;
    mpi_data->d_in1_buffer_size = 0;
    mpi_data->d_in2_buffer = NULL;
    mpi_data->d_in2_buffer_size = 0;

    for ( i = 0; i < MAX_NBRS; ++i )
    {
        mpi_data->d_out_buffers[i].cnt = 0;
        mpi_data->d_out_buffers[i].index = NULL;
        mpi_data->d_out_buffers[i].index_size = 0;
        mpi_data->d_out_buffers[i].out_atoms = NULL;
        mpi_data->d_out_buffers[i].out_atoms_size = 0;
    }
#endif

    Hip_Init_Simulation_Data( system, control, data );

    /* scratch space - set before Hip_Init_Workspace
     * as Hip_Init_System utilizes these variables */
    for ( i = 0; i < MAX_HIP_STREAMS; ++i )
    {
        workspace->scratch[i] = NULL;
        workspace->scratch_size[i] = 0;
    }
    workspace->host_scratch = NULL;
    workspace->host_scratch_size = 0;

    Hip_Init_System( system, control, data, workspace, mpi_data );
    /* reset for step 0 */
    Reset_Simulation_Data( data );

    Hip_Allocate_Grid( system, control );
    Hip_Copy_Grid_Host_to_Device( control, &system->my_grid, &system->d_my_grid );

    Hip_Init_Workspace( system, control, workspace, mpi_data );

    Hip_Allocate_Control( control );

    Hip_Init_Lists( system, control, data, workspace, lists, mpi_data );

    Init_Output_Files( system, control, out_control, mpi_data );

    if ( control->tabulate > 0 )
    {
        Make_LR_Lookup_Table( system, control, workspace->d_workspace, mpi_data );
    }

#if defined(DEBUG_FOCUS)
    Hip_Print_Mem_Usage( data );
#endif
}
