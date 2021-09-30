#if defined(PURE_REAX)
    #include "hip_copy.h"

    #include "hip_utils.h"

    #include "../list.h"
    #include "../vector.h"
#elif defined(LAMMPS_REAX)
    #include "reaxff_hip_copy.h"

    #include "reaxff_hip_utils.h"

    #include "reaxff_list.h"
    #include "reaxff_vector.h"
#include "reaxff_hip_list.h"

#endif

extern "C" void Output_Sync_Forces(storage *workspace, int total_cap) {
    sHipMemcpy(workspace->f, workspace->d_workspace->f,
               total_cap * sizeof(rvec), hipMemcpyDeviceToHost,
               __FILE__, __LINE__);
}


/* Copy grid info from host to device */
extern "C" void Hip_Copy_Grid_Host_to_Device( grid *host, grid *device )
{
    int total;

    total = host->ncells[0] * host->ncells[1] * host->ncells[2];

    ivec_Copy( device->ncells, host->ncells );
    rvec_Copy( device->cell_len, host->cell_len );
    rvec_Copy( device->inv_len, host->inv_len );

    ivec_Copy( device->bond_span, host->bond_span );
    ivec_Copy( device->nonb_span, host->nonb_span );
    ivec_Copy( device->vlist_span, host->vlist_span );

    ivec_Copy( device->native_cells, host->native_cells );
    ivec_Copy( device->native_str, host->native_str );
    ivec_Copy( device->native_end, host->native_end );

    device->ghost_cut = host->ghost_cut;
    ivec_Copy( device->ghost_span, host->ghost_span );
    ivec_Copy( device->ghost_nonb_span, host->ghost_nonb_span );
    ivec_Copy( device->ghost_hbond_span, host->ghost_hbond_span );
    ivec_Copy( device->ghost_bond_span, host->ghost_bond_span );

    sHipMemcpy( device->str, host->str, sizeof(int) * total,
            hipMemcpyHostToDevice, __FILE__, __LINE__ );
    sHipMemcpy( device->end, host->end, sizeof(int) * total,
            hipMemcpyHostToDevice, __FILE__, __LINE__ );
    sHipMemcpy( device->cutoff, host->cutoff, sizeof(real) * total,
            hipMemcpyHostToDevice, __FILE__, __LINE__ );
    sHipMemcpy( device->nbrs_x, host->nbrs_x, sizeof(ivec) * total * host->max_nbrs,
            hipMemcpyHostToDevice, __FILE__, __LINE__ );
    sHipMemcpy( device->nbrs_cp, host->nbrs_cp, sizeof(rvec) * total * host->max_nbrs,
            hipMemcpyHostToDevice, __FILE__, __LINE__ );

    sHipMemcpy( device->rel_box, host->rel_box, sizeof(ivec) * total,
            hipMemcpyHostToDevice, __FILE__, __LINE__ );

    device->max_nbrs = host->max_nbrs;
}


/* Copy atom info from host to device */
extern "C" void Hip_Copy_Atoms_Host_to_Device( reax_system *system )
{
    sHipMemcpy( system->d_my_atoms, system->my_atoms, sizeof(reax_atom) * system->N,
            hipMemcpyHostToDevice, __FILE__, __LINE__ );
}


/* Copy atomic system info from host to device */
extern "C" void Hip_Copy_System_Host_to_Device( reax_system *system )
{
    Hip_Copy_Atoms_Host_to_Device( system );

    sHipMemcpy( system->d_my_box, &system->my_box,
            sizeof(simulation_box), hipMemcpyHostToDevice, __FILE__, __LINE__ );

    sHipMemcpy( system->d_my_ext_box, &system->my_ext_box,
            sizeof(simulation_box), hipMemcpyHostToDevice, __FILE__, __LINE__ );

    sHipMemcpy( system->reax_param.d_sbp, system->reax_param.sbp,
            sizeof(single_body_parameters) * system->reax_param.num_atom_types,
            hipMemcpyHostToDevice, __FILE__, __LINE__ );
    sHipMemcpy( system->reax_param.d_tbp, system->reax_param.tbp,
            sizeof(two_body_parameters) * POW(system->reax_param.num_atom_types, 2),
            hipMemcpyHostToDevice, __FILE__, __LINE__ );
    sHipMemcpy( system->reax_param.d_thbp, system->reax_param.thbp,
            sizeof(three_body_header) * POW(system->reax_param.num_atom_types, 3),
            hipMemcpyHostToDevice, __FILE__, __LINE__ );
    sHipMemcpy( system->reax_param.d_hbp, system->reax_param.hbp,
            sizeof(hbond_parameters) * POW(system->reax_param.num_atom_types, 3),
            hipMemcpyHostToDevice, __FILE__, __LINE__ );
    sHipMemcpy( system->reax_param.d_fbp, system->reax_param.fbp,
            sizeof(four_body_header) * POW(system->reax_param.num_atom_types, 4),
            hipMemcpyHostToDevice, __FILE__, __LINE__ );

    sHipMemcpy( system->reax_param.d_gp.l, system->reax_param.gp.l,
            sizeof(real) * system->reax_param.gp.n_global,
            hipMemcpyHostToDevice, __FILE__, __LINE__ );

    system->reax_param.d_gp.n_global = system->reax_param.gp.n_global; 
    system->reax_param.d_gp.vdw_type = system->reax_param.gp.vdw_type; 
}


/* Copy atom info from device to host */
extern "C" void Hip_Copy_Atoms_Device_to_Host( reax_system *system )
{
    sHipMemcpy( system->my_atoms, system->d_my_atoms,
            sizeof(reax_atom) * system->N,
            hipMemcpyDeviceToHost, __FILE__, __LINE__ );
}


/* Copy simulation data from device to host */
extern "C" void Hip_Copy_Simulation_Data_Device_to_Host( simulation_data *host, simulation_data *dev )
{
    sHipMemcpy( &host->my_en, &dev->my_en, sizeof(energy_data),
            hipMemcpyDeviceToHost, __FILE__, __LINE__ );
    sHipMemcpy( &host->kin_press, &dev->kin_press, sizeof(real),
            hipMemcpyDeviceToHost, __FILE__, __LINE__ );
    sHipMemcpy( host->int_press, dev->int_press, sizeof(rvec),
            hipMemcpyDeviceToHost, __FILE__, __LINE__ );
    sHipMemcpy( host->ext_press, dev->ext_press, sizeof(rvec),
            hipMemcpyDeviceToHost, __FILE__, __LINE__ );
}


/* Copy interaction lists from device to host,
 * with allocation for the host list */
extern "C" void Hip_Copy_List_Device_to_Host( reax_list *host_list, reax_list *device_list, int type )
{
    int format;

//    assert( device_list != NULL );
//    assert( device_list->allocated == TRUE );

    format = host_list->format;

    if ( host_list != NULL && host_list->allocated == TRUE )
    {
        Delete_List( host_list );
    }
    Make_List( device_list->n, device_list->max_intrs, type, format, host_list );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, " [INFO] trying to copy %d list from device to host\n", type );
#endif

    sHipMemcpy( host_list->index, device_list->index,
            sizeof(int) * device_list->n,
            hipMemcpyDeviceToHost, __FILE__, __LINE__ );
    sHipMemcpy( host_list->end_index, device_list->end_index,
            sizeof(int) * device_list->n,
            hipMemcpyDeviceToHost, __FILE__, __LINE__ );

    switch ( type )
    {   
        case TYP_FAR_NEIGHBOR:
            sHipMemcpy( host_list->far_nbr_list.nbr, device_list->far_nbr_list.nbr,
                    sizeof(int) * device_list->max_intrs,
                    hipMemcpyDeviceToHost, __FILE__, __LINE__ );
            sHipMemcpy( host_list->far_nbr_list.rel_box, device_list->far_nbr_list.rel_box,
                    sizeof(ivec) * device_list->max_intrs,
                    hipMemcpyDeviceToHost, __FILE__, __LINE__ );
            sHipMemcpy( host_list->far_nbr_list.d, device_list->far_nbr_list.d,
                    sizeof(real) * device_list->max_intrs,
                    hipMemcpyDeviceToHost, __FILE__, __LINE__ );
            sHipMemcpy( host_list->far_nbr_list.dvec, device_list->far_nbr_list.dvec,
                    sizeof(rvec) * device_list->max_intrs,
                    hipMemcpyDeviceToHost, __FILE__, __LINE__ );
            break;

        case TYP_BOND:
            sHipMemcpy( host_list->bond_list, device_list->bond_list,
                    sizeof(bond_data) * device_list->max_intrs,
                    hipMemcpyDeviceToHost, __FILE__, __LINE__ );
            break;

        case TYP_HBOND:
            sHipMemcpy( host_list->hbond_list, device_list->hbond_list,
                    sizeof(hbond_data) * device_list->max_intrs,
                    hipMemcpyDeviceToHost, __FILE__, __LINE__ );
            break;

        case TYP_THREE_BODY:
            sHipMemcpy( host_list->three_body_list,
                    device_list->three_body_list,
                    sizeof(three_body_interaction_data ) * device_list->max_intrs,
                    hipMemcpyDeviceToHost, __FILE__, __LINE__ );
            break;

        default:
            fprintf( stderr, "[ERROR] Unknown list synching from device to host (%d)\n",
                    type );
            exit( INVALID_INPUT );
            break;
    }  
}

//extern "C" void Hip_Copy_List_Host_to_Device( reax_list *host_list, reax_list *device_list, int type )
//{
//    int format;
//
//    //    assert( device_list != NULL );
//    //    assert( device_list->allocated == TRUE );
//
//    format = host_list->format;
//
//    if ( host_list != NULL && host_list->allocated == TRUE )
//    {
//        Delete_List( host_list );
//    }
//    Hip_Make_List( device_list->n, device_list->max_intrs, type, format, host_list );
//
//#if defined(DEBUG_FOCUS)
//    fprintf( stderr, " [INFO] trying to copy %d list from device to host\n", type );
//#endif
//
//    sHipMemcpy( host_list->index, device_list->index,
//                sizeof(int) * device_list->n,
//                hipMemcpyDeviceToHost, __FILE__, __LINE__ );
//    sHipMemcpy( host_list->end_index, device_list->end_index,
//                sizeof(int) * device_list->n,
//                hipMemcpyDeviceToHost, __FILE__, __LINE__ );
//
//    switch ( type )
//    {
//        case TYP_FAR_NEIGHBOR:
//            sHipMemcpy( host_list->far_nbr_list.nbr, device_list->far_nbr_list.nbr,
//                        sizeof(int) * device_list->max_intrs,
//                        hipMemcpyDeviceToHost, __FILE__, __LINE__ );
//            sHipMemcpy( host_list->far_nbr_list.rel_box, device_list->far_nbr_list.rel_box,
//                        sizeof(ivec) * device_list->max_intrs,
//                        hipMemcpyDeviceToHost, __FILE__, __LINE__ );
//            sHipMemcpy( host_list->far_nbr_list.d, device_list->far_nbr_list.d,
//                        sizeof(real) * device_list->max_intrs,
//                        hipMemcpyDeviceToHost, __FILE__, __LINE__ );
//            sHipMemcpy( host_list->far_nbr_list.dvec, device_list->far_nbr_list.dvec,
//                        sizeof(rvec) * device_list->max_intrs,
//                        hipMemcpyDeviceToHost, __FILE__, __LINE__ );
//            break;
//
//            case TYP_BOND:
//                sHipMemcpy( host_list->bond_list, device_list->bond_list,
//                            sizeof(bond_data) * device_list->max_intrs,
//                            hipMemcpyDeviceToHost, __FILE__, __LINE__ );
//                break;
//
//                case TYP_HBOND:
//                    sHipMemcpy( host_list->hbond_list, device_list->hbond_list,
//                                sizeof(hbond_data) * device_list->max_intrs,
//                                hipMemcpyDeviceToHost, __FILE__, __LINE__ );
//                    break;
//
//                    case TYP_THREE_BODY:
//                        sHipMemcpy( host_list->three_body_list,
//                                    device_list->three_body_list,
//                                    sizeof(three_body_interaction_data ) * device_list->max_intrs,
//                                    hipMemcpyDeviceToHost, __FILE__, __LINE__ );
//                        break;
//
//                        default:
//                            fprintf( stderr, "[ERROR] Unknown list synching from device to host (%d)\n",
//                                     type );
//                            exit( INVALID_INPUT );
//                            break;
//    }
//}



/* Copy atom info from device to host */
extern "C" void Hip_Copy_MPI_Data_Host_to_Device( mpi_datatypes *mpi_data )
{
    hip_check_malloc( &mpi_data->d_in1_buffer, &mpi_data->d_in1_buffer_size,
            mpi_data->in1_buffer_size, "Hip_Copy_MPI_Data_Host_to_Device::mpi_data->d_in1_buffer" );

    hip_check_malloc( &mpi_data->d_in2_buffer, &mpi_data->d_in2_buffer_size,
            mpi_data->in2_buffer_size, "Hip_Copy_MPI_Data_Host_to_Device::mpi_data->d_in2_buffer" );

    for ( int i = 0; i < MAX_NBRS; ++i )
    {
        mpi_data->d_out_buffers[i].cnt = mpi_data->out_buffers[i].cnt;
        hip_check_malloc( (void **) &mpi_data->d_out_buffers[i].index, &mpi_data->d_out_buffers[i].index_size,
                mpi_data->out_buffers[i].index_size, "Hip_Copy_MPI_Data_Host_to_Device::mpi_data->d_out_buffers[i].index" );
        hip_check_malloc( &mpi_data->d_out_buffers[i].out_atoms, &mpi_data->d_out_buffers[i].out_atoms_size,
                mpi_data->out_buffers[i].out_atoms_size, "Hip_Copy_MPI_Data_Host_to_Device::mpi_data->d_out_buffers[i].out_atoms" );
    }

    for ( int i = 0; i < MAX_NBRS; ++i )
    {
        /* index is set during SendRecv and reused during MPI comms afterward,
         * so copy to device while SendRecv is still done on the host */
        sHipMemcpy( mpi_data->d_out_buffers[i].index, mpi_data->out_buffers[i].index,
                mpi_data->d_out_buffers[i].index_size,
                hipMemcpyHostToDevice, __FILE__, __LINE__ );
    }
}
