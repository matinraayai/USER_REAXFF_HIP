#include "hip/hip_runtime.h"
#if defined(LAMMPS_REAX)
    #include "reaxff_hip_forces.h"

    #include "reaxff_hip_bonds.h"
    #include "reaxff_hip_bond_orders.h"
    #include "reaxff_hip_charges.h"
    #include "reaxff_hip_helpers.h"
    #include "reaxff_hip_hydrogen_bonds.h"
    #include "reaxff_hip_list.h"
    #include "reaxff_hip_multi_body.h"
    #include "reaxff_hip_neighbors.h"
    #include "reaxff_hip_nonbonded.h"
    #include "reaxff_hip_reduction.h"
    #include "reaxff_hip_spar_lin_alg.h"
    #include "reaxff_hip_torsion_angles.h"
    #include "reaxff_hip_utils.h"
    #include "reaxff_hip_valence_angles.h"

    #include "reaxff_basic_comm.h"
    #include "reaxff_forces.h"
    #include "reaxff_index_utils.h"
    #include "reaxff_tool_box.h"
    #include "reaxff_vector.h"
#else
    #include "hip_forces.h"

    #include "hip_bonds.h"
    #include "hip_bond_orders.h"
    #include "hip_charges.h"
    #include "hip_helpers.h"
    #include "hip_hydrogen_bonds.h"
    #include "hip_list.h"
    #include "hip_multi_body.h"
    #include "hip_neighbors.h"
    #include "hip_nonbonded.h"
    #include "hip_reduction.h"
    #include "hip_spar_lin_alg.h"
    #include "hip_torsion_angles.h"
    #include "hip_utils.h"
    #include "hip_valence_angles.h"

    #include "../basic_comm.h"
    #include "../forces.h"
    #include "../index_utils.h"
    #include "../tool_box.h"
    #include "../vector.h"
#endif

#include <hipcub/hipcub.hpp>




typedef enum
{
    DIAGONAL = 0,
    OFF_DIAGONAL = 1,
} MATRIX_ENTRY_POSITION;


HIP_DEVICE real Init_Charge_Matrix_Entry( single_body_parameters const * const sbp_i,
        real const * const workspace_Tap, control_params const * const control,
        int i, int j, real r_ij, real gamma, MATRIX_ENTRY_POSITION pos )
{
    real Tap, dr3gamij_1, dr3gamij_3, ret;

    ret = 0.0;

    switch ( control->charge_method )
    {
    case QEQ_CM:
    case EE_CM:
    case ACKS2_CM:
        switch ( pos )
        {
            case OFF_DIAGONAL:
                Tap = workspace_Tap[7] * r_ij + workspace_Tap[6];
                Tap = Tap * r_ij + workspace_Tap[5];
                Tap = Tap * r_ij + workspace_Tap[4];
                Tap = Tap * r_ij + workspace_Tap[3];
                Tap = Tap * r_ij + workspace_Tap[2];
                Tap = Tap * r_ij + workspace_Tap[1];
                Tap = Tap * r_ij + workspace_Tap[0];    

                /* shielding */
                dr3gamij_1 = r_ij * r_ij * r_ij
                    + POW( gamma, -3.0 );
                dr3gamij_3 = POW( dr3gamij_1 , 1.0 / 3.0 );

                /* i == j: periodic self-interaction term
                 * i != j: general interaction term */
                ret = ((i == j) ? 0.5 : 1.0) * Tap * EV_to_KCALpMOL / dr3gamij_3;
                break;

            case DIAGONAL:
                ret = sbp_i->eta;
                break;

            default:
//                fprintf( stderr, "[ERROR] Invalid matrix position. Terminating...\n" );
//                exit( INVALID_INPUT );
                break;
        }
        break;


    default:
//        fprintf( stderr, "[ERROR] Invalid charge method. Terminating...\n" );
//        exit( INVALID_INPUT );
        break;
    }

    return ret;
}


HIP_DEVICE real Init_Charge_Matrix_Entry_Tab( LR_lookup_table const * const t_LR, real r_ij,
        int ti, int tj, int num_atom_types )
{
    int r, tmin, tmax;
    real val, dif, base;

    tmin = MIN( ti, tj );
    tmax = MAX( ti, tj );
    LR_lookup_table const * const t = &t_LR[ index_lr(tmin,tmax, num_atom_types) ];

    /* cubic spline interpolation */
    r = (int)(r_ij * t->inv_dx);
    if ( r == 0 )
    {
        ++r;
    }
    base = (real)(r + 1) * t->dx;
    dif = r_ij - base;
    val = ((t->ele[r].d * dif + t->ele[r].c) * dif + t->ele[r].b) * dif
        + t->ele[r].a;
    val *= EV_to_KCALpMOL / C_ELE;

    return val;
}


HIP_GLOBAL void k_disable_hydrogen_bonding( control_params * const control )
{
    control->hbond_cut = 0.0;
}


HIP_GLOBAL void k_init_end_index( int const * const intr_cnt,
        int const * const indices, int * const end_indices, int N )
{
    int i;

    i = blockIdx.x * blockDim.x  + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    end_indices[i] = indices[i] + intr_cnt[i];
}


HIP_GLOBAL void k_init_hbond_indices( reax_atom * const atoms,
        single_body_parameters const * const sbp,
        int const * const hbonds, int const * const max_hbonds,
        int * const indices, int * const end_indices, int N )
{
    int i, hindex, flag;

    i = blockIdx.x * blockDim.x  + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    hindex = atoms[i].Hindex;

    flag = (sbp[ atoms[i].type ].p_hbond == H_ATOM
            || sbp[ atoms[i].type ].p_hbond == H_BONDING_ATOM ? TRUE : FALSE);

    indices[hindex] = (flag == TRUE ? max_hbonds[i] : 0);
    end_indices[hindex] = (flag == TRUE ? indices[hindex] + hbonds[i] : 0);
    atoms[i].num_hbonds = (flag == TRUE ? hbonds[i] : 0);
}


HIP_GLOBAL void k_print_hbond_info( reax_atom *my_atoms, single_body_parameters *sbp,
        control_params *control, reax_list hbond_list, int N )
{
    int i;
    int type_i;
    single_body_parameters *sbp_i;
    reax_atom *atom_i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    atom_i = &my_atoms[i];
    type_i = atom_i->type;
    sbp_i = &sbp[type_i];

    printf( "atom %6d: ihb = %2d, ihb_top = %2d\n", i, sbp_i->p_hbond,
            Start_Index( atom_i->Hindex, &hbond_list ) );
}


/* 1 thread computes the distances and displacement vectors of an atom for its neighbors
 * in the far neighbors list if it's a NOT re-neighboring step
 */
HIP_GLOBAL void k_init_dist( reax_atom const * const my_atoms,
        reax_list far_nbr_list, int N )
{
    int i, j, pj, start_i, end_i;
    rvec x_i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    rvec_Copy( x_i, my_atoms[i].x );

    /* update distance and displacement vector between atoms i and j (i-j) */
    for ( pj = start_i; pj < end_i; ++pj )
    {
        j = far_nbr_list.far_nbr_list.nbr[pj];

        far_nbr_list.far_nbr_list.dvec[pj][0] = my_atoms[j].x[0] - x_i[0];
        far_nbr_list.far_nbr_list.dvec[pj][1] = my_atoms[j].x[1] - x_i[1];
        far_nbr_list.far_nbr_list.dvec[pj][2] = my_atoms[j].x[2] - x_i[2];
        far_nbr_list.far_nbr_list.d[pj] = rvec_Norm( far_nbr_list.far_nbr_list.dvec[pj] );
    }
}


/* 1 warp of threads computes the distances and displacement vectors of an atom for its neighbors
 * in the far neighbors list if it's a NOT re-neighboring step
 */
HIP_GLOBAL void k_init_dist_opt( reax_atom const * const my_atoms,
        reax_list far_nbr_list, int N )
{
    int j, pj, start_i, end_i, thread_id, i, lane_id;
    rvec x_i;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    i = thread_id / warpSize;

    if ( i >= N )
    {
        return;
    }

    lane_id = thread_id % warpSize; 
    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    rvec_Copy( x_i, my_atoms[i].x );

    /* update distance and displacement vector between atoms i and j (i-j) */
    for ( pj = start_i + lane_id; pj < end_i; pj += warpSize )
    {
        j = far_nbr_list.far_nbr_list.nbr[pj];

        far_nbr_list.far_nbr_list.dvec[pj][0] = my_atoms[j].x[0] - x_i[0];
        far_nbr_list.far_nbr_list.dvec[pj][1] = my_atoms[j].x[1] - x_i[1];
        far_nbr_list.far_nbr_list.dvec[pj][2] = my_atoms[j].x[2] - x_i[2];
        far_nbr_list.far_nbr_list.d[pj] = rvec_Norm( far_nbr_list.far_nbr_list.dvec[pj] );
    }
}



/* Reinitialize all bond order data structures. This kernel is needed
 * to make the bond list initialization an atomic transaction (which
 * is needed for the computation / reallocation logic).
 */
HIP_GLOBAL void k_reset_bond_orders( storage workspace, int N )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    workspace.total_bond_order[i] = 0.0;
    rvec_MakeZero( workspace.dDeltap_self[i] );
}


/* Compute the charge matrix entries and store the matrix in half format
 * using the far neighbors list (stored in full format) and according to
 * the full shell communication method */
HIP_GLOBAL void k_init_cm_half_fs( reax_atom * const my_atoms,
        single_body_parameters const * const sbp, two_body_parameters const * const tbp,
        storage workspace, control_params const * const control,
        reax_list far_nbr_list, int num_atom_types,
        int * const max_cm_entries, int * const realloc_cm_entries )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, orig_id_i;
    int cm_top;
    int num_cm_entries;
    real r_ij;
    sparse_matrix *H;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= workspace.H.n_max )
    {
        return;
    }

    H = &workspace.H;
    cm_top = H->start[i];

    if ( i < H->n )
    {
        type_i = my_atoms[i].type;
        orig_id_i = my_atoms[i].orig_id;
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );

        /* diagonal entry in the matrix */
        H->j[cm_top] = i;
        H->val[cm_top] = Init_Charge_Matrix_Entry( &sbp[type_i], workspace.Tap, control,
                i, i, 0.0, 0.0, DIAGONAL );
        ++cm_top;

        for ( pj = start_i; pj < end_i; ++pj )
        {
            if ( far_nbr_list.far_nbr_list.d[pj] <= control->nonb_cut )
            {
                j = far_nbr_list.far_nbr_list.nbr[pj];

                /* if j is a local OR ghost atom in the upper triangular region of the matrix */
                if ( orig_id_i < my_atoms[j].orig_id )
                {
                    r_ij = far_nbr_list.far_nbr_list.d[pj];

                    H->j[cm_top] = j;
                    H->val[cm_top] = Init_Charge_Matrix_Entry( &sbp[type_i], workspace.Tap,
                            control, i, H->j[cm_top], r_ij,
                            tbp[ index_tbp(type_i, my_atoms[j].type, num_atom_types) ].gamma, OFF_DIAGONAL );
                    ++cm_top;
                }
            }
        }
    }

    __syncthreads( );

    H->end[i] = cm_top;
    num_cm_entries = cm_top - H->start[i];

    /* reallocation check */
    if ( num_cm_entries > max_cm_entries[i] )
    {
        *realloc_cm_entries = TRUE;
    }
}


/* Compute the tabulated charge matrix entries and store the matrix in half format
 * using the far neighbors list (stored in full format) and according to
 * the full shell communication method */
HIP_GLOBAL void k_init_cm_half_fs_tab( reax_atom * const my_atoms,
        single_body_parameters const * const sbp,
        storage workspace, control_params const * const control,
        reax_list far_nbr_list, LR_lookup_table const * const t_LR, int num_atom_types,
        int * const max_cm_entries, int * const realloc_cm_entries )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int cm_top;
    int num_cm_entries;
    real r_ij;
    reax_atom *atom_i, *atom_j;
    sparse_matrix *H;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= workspace.H.n_max )
    {
        return;
    }

    H = &workspace.H;
    cm_top = H->start[i];

    if ( i < H->n )
    {
        atom_i = &my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );

        /* diagonal entry in the matrix */
        H->j[cm_top] = i;
        H->val[cm_top] = Init_Charge_Matrix_Entry( &sbp[type_i], workspace.Tap, control,
                i, i, 0.0, 0.0, DIAGONAL );
        ++cm_top;

        for ( pj = start_i; pj < end_i; ++pj )
        {
            j = far_nbr_list.far_nbr_list.nbr[pj];

            if ( far_nbr_list.far_nbr_list.d[pj] <= control->nonb_cut )
            {
                atom_j = &my_atoms[j];
                type_j = atom_j->type;

                /* if j is a local OR ghost atom in the upper triangular region of the matrix */
                if ( atom_i->orig_id < atom_j->orig_id )
                {
                    r_ij = far_nbr_list.far_nbr_list.d[pj];

                    H->j[cm_top] = j;
                    H->val[cm_top] = Init_Charge_Matrix_Entry_Tab( t_LR, r_ij,
                            type_i, type_j, num_atom_types );
                    ++cm_top;
                }
            }
        }
    }

    __syncthreads( );

    H->end[i] = cm_top;
    num_cm_entries = cm_top - H->start[i];

    /* reallocation check */
    if ( num_cm_entries > max_cm_entries[i] )
    {
        *realloc_cm_entries = TRUE;
    }
}


/* Compute the charge matrix entries and store the matrix in full format
 * using the far neighbors list (stored in full format) and according to
 * the full shell communication method */
HIP_GLOBAL void k_init_cm_full_fs( reax_atom * const my_atoms,
        single_body_parameters const * const sbp, two_body_parameters const * const tbp,
        storage workspace, control_params const * const control,
        reax_list far_nbr_list, int num_atom_types,
        int * const max_cm_entries, int * const realloc_cm_entries )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i;
    int cm_top;
    int num_cm_entries;
    reax_atom *atom_i;
    sparse_matrix *H;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= workspace.H.n_max )
    {
        return;
    }

    H = &workspace.H;
    cm_top = H->start[i];

    if ( i < H->n )
    {
        atom_i = &my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );

        /* diagonal entry in the matrix */
        H->j[cm_top] = i;
        H->val[cm_top] = Init_Charge_Matrix_Entry( &sbp[type_i], workspace.Tap, control,
                i, i, 0.0, 0.0, DIAGONAL );
        ++cm_top;

        for ( pj = start_i; pj < end_i; ++pj )
        {
            if ( far_nbr_list.far_nbr_list.d[pj] <= control->nonb_cut )
            {
                j = far_nbr_list.far_nbr_list.nbr[pj];

                H->j[cm_top] = j;
                H->val[cm_top] = Init_Charge_Matrix_Entry( &sbp[type_i], workspace.Tap,
                        control, i, j, far_nbr_list.far_nbr_list.d[pj],
                        tbp[ index_tbp(type_i, my_atoms[j].type, num_atom_types) ].gamma,
                        OFF_DIAGONAL );
                ++cm_top;
            }
        }
    }

    __syncthreads( );

    H->end[i] = cm_top;
    num_cm_entries = cm_top - H->start[i];

    /* reallocation check */
    if ( num_cm_entries > max_cm_entries[i] )
    {
        *realloc_cm_entries = TRUE;
    }
}


/* Compute the charge matrix entries and store the matrix in full format
 * using the far neighbors list (stored in full format) and according to
 * the full shell communication method */
HIP_GLOBAL void k_init_cm_full_fs_opt( reax_atom * const my_atoms,
        single_body_parameters const * const sbp, two_body_parameters const * const tbp,
        storage workspace, control_params const * const control,
        reax_list far_nbr_list, int num_atom_types,
        int * const max_cm_entries, int * const realloc_cm_entries )
{
    extern __shared__ hipcub::WarpScan<int>::TempStorage temp1[];
    int i, j, pj, thread_id, warp_id, lane_id, itr;
    int start_i, end_i, type_i;
    int cm_top, num_cm_entries, offset, flag;
    reax_atom *atom_i;
    sparse_matrix *H;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    /* all threads within a warp are assigned the same unique row 
     * in the charge matrix */
    i = thread_id / warpSize;

    if ( i >= workspace.H.n_max )
    {
        return;
    }

    warp_id = threadIdx.x / warpSize;
    lane_id = thread_id % warpSize;
    H = &workspace.H;
    cm_top = H->start[i];

    if ( i < H->n )
    {
        atom_i = &my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );

        /* diagonal entry in the matrix */
        if ( lane_id == 0 )
        {
            H->j[cm_top] = i;
            H->val[cm_top] = Init_Charge_Matrix_Entry( &sbp[type_i], workspace.Tap, control,
                    i, i, 0.0, 0.0, DIAGONAL );
        }
        ++cm_top;

        for ( itr = 0, pj = start_i + lane_id; itr < (end_i - start_i + warpSize - 1) / warpSize; ++itr )
        {
            offset = (pj < end_i && far_nbr_list.far_nbr_list.d[pj] <= control->nonb_cut) ? 1 : 0;
            flag = (offset == 1) ? TRUE : FALSE;
            hipcub::WarpScan<int>(temp1[warp_id]).ExclusiveSum(offset, offset);

            if ( flag == TRUE )
            {
                j = far_nbr_list.far_nbr_list.nbr[pj];

                H->j[cm_top + offset] = j;
                H->val[cm_top + offset] = Init_Charge_Matrix_Entry( &sbp[type_i], workspace.Tap,
                        control, i, j, far_nbr_list.far_nbr_list.d[pj],
                        tbp[ index_tbp(type_i, my_atoms[j].type, num_atom_types) ].gamma, OFF_DIAGONAL );
            }

            /* get cm_top from thread in last lane */
            cm_top = cm_top + offset + (flag == TRUE ? 1 : 0);
            cm_top = __shfl(cm_top, warpSize - 1 );

            pj += warpSize;
        }
    }

    if ( lane_id == 0 )
    {
        H->end[i] = cm_top;
        num_cm_entries = cm_top - H->start[i];

        /* reallocation check */
        if ( num_cm_entries > max_cm_entries[i] )
        {
            *realloc_cm_entries = TRUE;
        }
    }
}


/* Compute the tabulated charge matrix entries and store the matrix in full format
 * using the far neighbors list (stored in full format) and according to
 * the full shell communication method */
HIP_GLOBAL void k_init_cm_full_fs_tab( reax_atom * const my_atoms,
        single_body_parameters const * const sbp,
        storage workspace, control_params const * const control,
        reax_list far_nbr_list, LR_lookup_table *t_LR, int num_atom_types,
        int * const max_cm_entries, int * const realloc_cm_entries )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i;
    int cm_top;
    int num_cm_entries;
    sparse_matrix *H;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= workspace.H.n_max )
    {
        return;
    }

    H = &workspace.H;
    cm_top = H->start[i];

    if ( i < H->n )
    {
        type_i = my_atoms[i].type;
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );

        /* diagonal entry in the matrix */
        H->j[cm_top] = i;
        H->val[cm_top] = Init_Charge_Matrix_Entry( &sbp[type_i], workspace.Tap, control,
                i, i, 0.0, 0.0, DIAGONAL );
        ++cm_top;

        for ( pj = start_i; pj < end_i; ++pj )
        {
            j = far_nbr_list.far_nbr_list.nbr[pj];

            if ( far_nbr_list.far_nbr_list.d[pj] <= control->nonb_cut )
            {
                H->j[cm_top] = j;
                H->val[cm_top] = Init_Charge_Matrix_Entry_Tab( t_LR,
                        far_nbr_list.far_nbr_list.d[pj],
                        type_i, my_atoms[j].type, num_atom_types );
                ++cm_top;
            }
        }
    }

    __syncthreads( );

    H->end[i] = cm_top;
    num_cm_entries = cm_top - H->start[i];

    /* reallocation check */
    if ( num_cm_entries > max_cm_entries[i] )
    {
        *realloc_cm_entries = TRUE;
    }
}


HIP_GLOBAL void k_init_bonds( reax_atom * const my_atoms,
        single_body_parameters const * const sbp, two_body_parameters const * const tbp,
        storage workspace, control_params const * const control,
        reax_list far_nbr_list, reax_list bond_list, int n, int N,
        int num_atom_types, int * const max_bonds, int * const realloc_bonds )
{
    int i, j, pj, start_i, end_i;
    int type_i, type_j, tbp_ij;
    int btop_i, num_bonds;
    real total_bond_order_i;
    rvec dDeltap_self_i;
    real cutoff, r_ij;
    real C12, C34, C56;
    real BO_s, BO_pi, BO_pi2, BO;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    type_i = my_atoms[i].type;
    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    btop_i = Start_Index( i, &bond_list );
    total_bond_order_i = 0.0;
    rvec_MakeZero( dDeltap_self_i );

    if ( i < n )
    {
        cutoff = MIN( control->nonb_cut, control->bond_cut );
//        workspace.bond_mark[i] = 0;
    }
    else
    {
        cutoff = control->bond_cut;
        /* put ghost atoms to an infinite distance (i.e., 1000) */
//        workspace.bond_mark[i] = 1000;
    }

    /* check if j is within cutoff */
    for ( pj = start_i; pj < end_i; ++pj )
    {
        /* uncorrected bond orders */
        if ( far_nbr_list.far_nbr_list.d[pj] <= cutoff )
        {
            j = far_nbr_list.far_nbr_list.nbr[pj];
            type_j = my_atoms[j].type;
            r_ij = far_nbr_list.far_nbr_list.d[pj];
            tbp_ij = index_tbp(type_i, type_j, num_atom_types);

            /* uncorrected bond orders */
            if ( sbp[type_i].r_s > 0.0 && sbp[type_j].r_s > 0.0 )
            {
                C12 = tbp[tbp_ij].p_bo1 * POW( r_ij / tbp[tbp_ij].r_s, tbp[tbp_ij].p_bo2 );
                BO_s = (1.0 + control->bo_cut) * EXP( C12 );
            }
            else
            {
                C12 = 0.0;
                BO_s = 0.0;
            }

            if ( sbp[type_i].r_pi > 0.0 && sbp[type_j].r_pi > 0.0 )
            {
                C34 = tbp[tbp_ij].p_bo3 * POW( r_ij / tbp[tbp_ij].r_p, tbp[tbp_ij].p_bo4 );
                BO_pi = EXP( C34 );
            }
            else
            {
                C34 = 0.0;
                BO_pi = 0.0;
            }

            if ( sbp[type_i].r_pi_pi > 0.0 && sbp[type_j].r_pi_pi > 0.0 )
            {
                C56 = tbp[tbp_ij].p_bo5 * POW( r_ij / tbp[tbp_ij].r_pp, tbp[tbp_ij].p_bo6 );
                BO_pi2 = EXP( C56 );
            }
            else
            {
                C56 = 0.0;
                BO_pi2 = 0.0;
            }

            /* initially BO values are the uncorrected ones, page 1 */
            BO = BO_s + BO_pi + BO_pi2;

            if ( BO >= control->bo_cut )
            {
                /* compute and append bond info to list */
                Hip_Compute_BOp( bond_list, control->bo_cut, i, btop_i,
                        far_nbr_list.far_nbr_list.nbr[pj],
                        C12, C34, C56, BO_s, BO_pi, BO_pi2, BO,
                        &far_nbr_list.far_nbr_list.rel_box[pj],
                        far_nbr_list.far_nbr_list.d[pj],
                        &far_nbr_list.far_nbr_list.dvec[pj], far_nbr_list.format,
                        &tbp[tbp_ij], dDeltap_self_i, &total_bond_order_i );

                ++btop_i;

                /* TODO: future optimization if bond_mark implemented */
//                if ( workspace->bond_mark[j] > workspace->bond_mark[i] + 1 )
//                {
//                    workspace->bond_mark[j] = workspace->bond_mark[i] + 1;
//                }
//                else if ( workspace->bond_mark[i] > workspace->bond_mark[j] + 1 )
//                {
//                    workspace->bond_mark[i] = workspace->bond_mark[j] + 1;
//                }
            }
        }
    }

    Set_End_Index( i, btop_i, &bond_list );

    num_bonds = btop_i - Start_Index( i, &bond_list );

    /* copy bond info to atom structure
     * (needed for atom ownership transfer via MPI) */
    my_atoms[i].num_bonds = num_bonds;

    workspace.total_bond_order[i] = total_bond_order_i;
    rvec_Copy( workspace.dDeltap_self[i], dDeltap_self_i );

    /* reallocation check */
    if ( num_bonds > max_bonds[i] )
    {
        *realloc_bonds = TRUE;
    }
}


HIP_GLOBAL void k_init_bonds_opt( reax_atom * const my_atoms,
        single_body_parameters const * const sbp, two_body_parameters const * const tbp,
        storage workspace, control_params const * const control,
        reax_list far_nbr_list, reax_list bond_list, int n, int N,
        int num_atom_types, int * const max_bonds, int * const realloc_bonds )
{
    extern __shared__ hipcub::WarpScan<int>::TempStorage temp21[];
    hipcub::WarpReduce<double>::TempStorage *temp22;
    int i, j, pj, thread_id, warp_id, lane_id, itr;
    int start_i, end_i, tbp_ij;
    int type_i, type_j;
    int btop_i, offset, flag, num_bonds;
    real cutoff, r_ij;
    real C12, C34, C56;
    real BO_s, BO_pi, BO_pi2, BO;
    real total_bond_order_i;
    rvec dDeltap_self_i;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    /* all threads within a warp are assigned the bonds
     * for a unique atom */
    i = thread_id / warpSize;

    if ( i >= N )
    {
        return;
    }

    warp_id = threadIdx.x / warpSize;
    lane_id = thread_id % warpSize;
    temp22 = (hipcub::WarpReduce<double>::TempStorage *) &temp21[warp_id];
    type_i = my_atoms[i].type;
    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    btop_i = Start_Index( i, &bond_list );
    total_bond_order_i = 0.0;
    rvec_MakeZero( dDeltap_self_i );

    if ( i < n )
    {
        cutoff = MIN( control->nonb_cut, control->bond_cut );
//        workspace.bond_mark[i] = 0;
    }
    else
    {
        cutoff = control->bond_cut;
        /* put ghost atoms to an infinite distance (i.e., 1000) */
//        workspace.bond_mark[i] = 1000;
    }

    for ( itr = 0, pj = start_i + lane_id; itr < (end_i - start_i + warpSize - 1) / warpSize; ++itr )
    {
        /* uncorrected bond orders */
        if ( pj < end_i && far_nbr_list.far_nbr_list.d[pj] <= cutoff )
        {
            j = far_nbr_list.far_nbr_list.nbr[pj];
            type_j = my_atoms[j].type;
            r_ij = far_nbr_list.far_nbr_list.d[pj];
            tbp_ij = index_tbp(type_i, type_j, num_atom_types);

            /* uncorrected bond orders */
            if ( sbp[type_i].r_s > 0.0 && sbp[type_j].r_s > 0.0 )
            {
                C12 = tbp[tbp_ij].p_bo1 * POW( r_ij / tbp[tbp_ij].r_s, tbp[tbp_ij].p_bo2 );
                BO_s = (1.0 + control->bo_cut) * EXP( C12 );
            }
            else
            {
                C12 = 0.0;
                BO_s = 0.0;
            }

            if ( sbp[type_i].r_pi > 0.0 && sbp[type_j].r_pi > 0.0 )
            {
                C34 = tbp[tbp_ij].p_bo3 * POW( r_ij / tbp[tbp_ij].r_p, tbp[tbp_ij].p_bo4 );
                BO_pi = EXP( C34 );
            }
            else
            {
                C34 = 0.0;
                BO_pi = 0.0;
            }

            if ( sbp[type_i].r_pi_pi > 0.0 && sbp[type_j].r_pi_pi > 0.0 )
            {
                C56 = tbp[tbp_ij].p_bo5 * POW( r_ij / tbp[tbp_ij].r_pp, tbp[tbp_ij].p_bo6 );
                BO_pi2 = EXP( C56 );
            }
            else
            {
                C56 = 0.0;
                BO_pi2 = 0.0;
            }
        }
        else
        {
            BO_s = 0.0;
            BO_pi = 0.0;
            BO_pi2 = 0.0;
        }

        /* initially BO values are the uncorrected ones, page 1 */
        BO = BO_s + BO_pi + BO_pi2;

        offset = (pj < end_i && far_nbr_list.far_nbr_list.d[pj] <= cutoff && BO >= control->bo_cut) ? 1 : 0;
        flag = (offset == 1) ? TRUE : FALSE;
        hipcub::WarpScan<int>(temp21[warp_id]).ExclusiveSum(offset, offset);

        if ( flag == TRUE )
        {
            /* compute and append bond info to list */
            Hip_Compute_BOp( bond_list, control->bo_cut, i, btop_i + offset,
                    far_nbr_list.far_nbr_list.nbr[pj],
                    C12, C34, C56, BO_s, BO_pi, BO_pi2, BO,
                    &far_nbr_list.far_nbr_list.rel_box[pj],
                    far_nbr_list.far_nbr_list.d[pj],
                    &far_nbr_list.far_nbr_list.dvec[pj], far_nbr_list.format,
                    &tbp[tbp_ij], dDeltap_self_i, &total_bond_order_i );

            /* TODO: future optimization if bond_mark implemented */
//            if ( workspace->bond_mark[j] > workspace->bond_mark[i] + 1 )
//            {
//                workspace->bond_mark[j] = workspace->bond_mark[i] + 1;
//            }
//            else if ( workspace->bond_mark[i] > workspace->bond_mark[j] + 1 )
//            {
//                workspace->bond_mark[i] = workspace->bond_mark[j] + 1;
//            }
        }

        /* get btop_i from thread in last lane */
        btop_i = btop_i + offset + (flag == TRUE ? 1 : 0);
        btop_i = __shfl(btop_i, warpSize - 1 );

        pj += warpSize;
    }

    total_bond_order_i = hipcub::WarpReduce<double>(temp22[warp_id]).Sum(total_bond_order_i);
    dDeltap_self_i[0] = hipcub::WarpReduce<double>(temp22[warp_id]).Sum(dDeltap_self_i[0]);
    dDeltap_self_i[1] = hipcub::WarpReduce<double>(temp22[warp_id]).Sum(dDeltap_self_i[1]);
    dDeltap_self_i[2] = hipcub::WarpReduce<double>(temp22[warp_id]).Sum(dDeltap_self_i[2]);

    if ( lane_id == 0 )
    {
        Set_End_Index( i, btop_i, &bond_list );

        num_bonds = btop_i - Start_Index( i, &bond_list );

        /* copy bond info to atom structure
         * (needed for atom ownership transfer via MPI) */
        my_atoms[i].num_bonds = num_bonds;

        workspace.total_bond_order[i] = total_bond_order_i;
        rvec_Copy( workspace.dDeltap_self[i], dDeltap_self_i );

        /* reallocation check */
        if ( num_bonds > max_bonds[i] )
        {
            *realloc_bonds = TRUE;
        }
    }
}


/* Construct the interaction list for hydrogen bonds */
HIP_GLOBAL void k_init_hbonds( reax_atom * const my_atoms,
        single_body_parameters const * const sbp, control_params const * const control,
        reax_list far_nbr_list, reax_list hbond_list,
        int n, int N, int num_atom_types, int * const max_hbonds, int * const realloc_hbonds )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int ihb, jhb, ihb_top;
    int num_hbonds;
    real cutoff;
    reax_atom *atom_i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    atom_i = &my_atoms[i];
    type_i = atom_i->type;
    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    ihb = sbp[type_i].p_hbond;

    cutoff = MIN( control->nonb_cut, control->hbond_cut );

    ihb_top = Start_Index( atom_i->Hindex, &hbond_list );

    if ( (i < n && ihb == H_ATOM) || ihb == H_BONDING_ATOM )
    {
        /* check if j is within cutoff */
        for ( pj = start_i; pj < end_i; ++pj )
        {
            if ( far_nbr_list.far_nbr_list.d[pj] <= cutoff )
            {
                j = far_nbr_list.far_nbr_list.nbr[pj];
                type_j = my_atoms[j].type;
                jhb = sbp[type_j].p_hbond;

                /* atom i: H bonding, ghost
                 * atom j: H atom, native */
                if ( i >= n && j < n
                        && ihb == H_BONDING_ATOM && jhb == H_ATOM )
                {
                    hbond_list.hbond_list[ihb_top].nbr = j;
                    hbond_list.hbond_list[ihb_top].scl = -1;
                    hbond_list.hbond_list[ihb_top].ptr = pj;

#if !defined(HIP_ACCUM_ATOMIC)
                    hbond_list.hbond_list[ihb_top].sym_index = -1;
                    rvec_MakeZero( hbond_list.hbond_list[ihb_top].hb_f );
#endif

                    ++ihb_top;
                }
                /* atom i: H atom, native
                 * atom j: H bonding atom */
                else if ( i < n
                        && ihb == H_ATOM && jhb == H_BONDING_ATOM )
                {
                    hbond_list.hbond_list[ihb_top].nbr = j;
                    hbond_list.hbond_list[ihb_top].scl = 1;
                    hbond_list.hbond_list[ihb_top].ptr = pj;

#if !defined(HIP_ACCUM_ATOMIC)
                    hbond_list.hbond_list[ihb_top].sym_index = -1;
                    rvec_MakeZero( hbond_list.hbond_list[ihb_top].hb_f );
#endif

                    ++ihb_top;
                }
                /* atom i: H bonding atom, native
                 * atom j: H atom, native */
                else if ( i < n
                        && ihb == H_BONDING_ATOM && jhb == H_ATOM && j < n )
                {
                    hbond_list.hbond_list[ihb_top].nbr = j;
                    hbond_list.hbond_list[ihb_top].scl = -1;
                    hbond_list.hbond_list[ihb_top].ptr = pj;

#if !defined(HIP_ACCUM_ATOMIC)
                    hbond_list.hbond_list[ihb_top].sym_index = -1;
                    rvec_MakeZero( hbond_list.hbond_list[ihb_top].hb_f );
#endif

                    ++ihb_top;
                }
            }
        }
    }

    Set_End_Index( atom_i->Hindex, ihb_top, &hbond_list );

    num_hbonds = ihb_top - Start_Index( atom_i->Hindex, &hbond_list );

    /* copy hbond info to atom structure
     * (needed for atom ownership transfer via MPI) */
    my_atoms[i].num_hbonds = num_hbonds;

    /* reallocation check */
    if ( num_hbonds > max_hbonds[i] )
    {
        *realloc_hbonds = TRUE;
    }
}


/* Construct the interaction list for hydrogen bonds */
HIP_GLOBAL void k_init_hbonds_opt( reax_atom * const my_atoms,
        single_body_parameters const * const sbp, control_params const * const control,
        reax_list far_nbr_list, reax_list hbond_list,
        int n, int N, int num_atom_types, int * const max_hbonds, int * const realloc_hbonds )
{
    extern __shared__ hipcub::WarpScan<int>::TempStorage temp[];
    int i, j, pj, thread_id, warp_id, lane_id, itr;
    int start_i, end_i;
    int type_i, type_j;
    int ihb, jhb, ihb_top, offset, flag;
    int num_hbonds;
    real cutoff;
    reax_atom *atom_i;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    /* all threads within a warp are assigned the bonds
     * for a unique atom */
    i = thread_id / warpSize;

    if ( i >= N )
    {
        return;
    }

    warp_id = threadIdx.x / warpSize;
    lane_id = thread_id % warpSize;
    atom_i = &my_atoms[i];
    type_i = atom_i->type;
    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    ihb = sbp[type_i].p_hbond;

    cutoff = MIN( control->nonb_cut, control->hbond_cut );

    ihb_top = Start_Index( atom_i->Hindex, &hbond_list );

    if ( (i < n && ihb == H_ATOM) || ihb == H_BONDING_ATOM )
    {
        for ( itr = 0, pj = start_i + lane_id; itr < (end_i - start_i + warpSize - 1) / warpSize; ++itr )
        {
            if ( pj < end_i )
            {
                j = far_nbr_list.far_nbr_list.nbr[pj];
                type_j = my_atoms[j].type;
                jhb = sbp[type_j].p_hbond;

                offset = (pj < end_i && far_nbr_list.far_nbr_list.d[pj] <= cutoff
                        && ((i >= n && j < n && ihb == H_BONDING_ATOM && jhb == H_ATOM)
                            || (i < n && ihb == H_ATOM && jhb == H_BONDING_ATOM)
                            || (i < n && ihb == H_BONDING_ATOM && jhb == H_ATOM && j < n))) ? 1 : 0;
            }
            else
            {
                offset = 0;
            }

            flag = (offset == 1) ? TRUE : FALSE;
            hipcub::WarpScan<int>(temp[warp_id]).ExclusiveSum(offset, offset);

            if ( flag == TRUE )
            {
                /* atom i: H bonding, ghost
                 * atom j: H atom, native */
                if ( i >= n && j < n
                        && ihb == H_BONDING_ATOM && jhb == H_ATOM )
                {
                    hbond_list.hbond_list[ihb_top + offset].nbr = j;
                    hbond_list.hbond_list[ihb_top + offset].scl = -1;
                    hbond_list.hbond_list[ihb_top + offset].ptr = pj;

#if !defined(HIP_ACCUM_ATOMIC)
                    hbond_list.hbond_list[ihb_top + offset].sym_index = -1;
                    rvec_MakeZero( hbond_list.hbond_list[ihb_top + offset].hb_f );
#endif
                }
                /* atom i: H atom, native
                 * atom j: H bonding atom */
                else if ( i < n
                        && ihb == H_ATOM && jhb == H_BONDING_ATOM )
                {
                    hbond_list.hbond_list[ihb_top + offset].nbr = j;
                    hbond_list.hbond_list[ihb_top + offset].scl = 1;
                    hbond_list.hbond_list[ihb_top + offset].ptr = pj;

#if !defined(HIP_ACCUM_ATOMIC)
                    hbond_list.hbond_list[ihb_top + offset].sym_index = -1;
                    rvec_MakeZero( hbond_list.hbond_list[ihb_top + offset].hb_f );
#endif
                }
                /* atom i: H bonding atom, native
                 * atom j: H atom, native */
                else if ( i < n
                        && ihb == H_BONDING_ATOM && jhb == H_ATOM && j < n )
                {
                    hbond_list.hbond_list[ihb_top + offset].nbr = j;
                    hbond_list.hbond_list[ihb_top + offset].scl = -1;
                    hbond_list.hbond_list[ihb_top + offset].ptr = pj;

#if !defined(HIP_ACCUM_ATOMIC)
                    hbond_list.hbond_list[ihb_top + offset].sym_index = -1;
                    rvec_MakeZero( hbond_list.hbond_list[ihb_top + offset].hb_f );
#endif
                }
            }

            /* get ihb_top from thread in last lane */
            ihb_top = ihb_top + offset + (flag == TRUE ? 1 : 0);
            ihb_top = __shfl(ihb_top, warpSize - 1 );

            pj += warpSize;
        }
    }

    if ( lane_id == 0 )
    {
        Set_End_Index( atom_i->Hindex, ihb_top, &hbond_list );

        num_hbonds = ihb_top - Start_Index( atom_i->Hindex, &hbond_list );

        /* copy hbond info to atom structure
         * (needed for atom ownership transfer via MPI) */
        my_atoms[i].num_hbonds = num_hbonds;

        /* reallocation check */
        if ( num_hbonds > max_hbonds[i] )
        {
            *realloc_hbonds = TRUE;
        }
    }
}


/* Construct the interaction list for bonds */
HIP_GLOBAL void k_estimate_storages_cm_half( reax_atom const * const my_atoms,
        control_params const * const control, reax_list far_nbr_list,
        int cm_n, int cm_n_max, int * const cm_entries, int * const max_cm_entries )
{
    int i, j, pj; 
    int start_i, end_i;
    int num_cm_entries;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= cm_n_max )
    {
        return;
    }

    num_cm_entries = 0;

    if ( i < cm_n )
    {
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );

        /* diagonal entry */
        ++num_cm_entries;

        for ( pj = start_i; pj < end_i; ++pj )
        { 
            j = far_nbr_list.far_nbr_list.nbr[pj];

            if ( far_nbr_list.far_nbr_list.d[pj] <= control->nonb_cut
                    && (j < cm_n || my_atoms[i].orig_id < my_atoms[j].orig_id) )
            {
                ++num_cm_entries;
            }
        }
    }

    __syncthreads( );

    cm_entries[i] = num_cm_entries;
    /* round up to the nearest multiple of 32 to ensure that reads along
     * rows can be coalesced for 1 warp per row SpMV implementation */
    max_cm_entries[i] = MAX( ((int) CEIL( num_cm_entries * SAFE_ZONE )
                + warpSize - 1) / warpSize * warpSize, MIN_CM_ENTRIES );
}


HIP_GLOBAL void k_estimate_storages_cm_full( control_params const * const control,
        reax_list far_nbr_list, int cm_n, int cm_n_max,
        int * const cm_entries, int * const max_cm_entries )
{
    int i, pj; 
    int start_i, end_i;
    int num_cm_entries;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= cm_n_max )
    {
        return;
    }

    num_cm_entries = 0;

    if ( i < cm_n )
    {
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );

        /* diagonal entry */
        ++num_cm_entries;

        for ( pj = start_i; pj < end_i; ++pj )
        { 
            if ( far_nbr_list.far_nbr_list.d[pj] <= control->nonb_cut )
            {
                ++num_cm_entries;
            }
        }
    }

    __syncthreads( );

    cm_entries[i] = num_cm_entries;
    /* round up to the nearest multiple of 32 to ensure that reads along
     * rows can be coalesced for 1 warp per row SpMV implementation */
    max_cm_entries[i] = MAX( ((int) CEIL( num_cm_entries * SAFE_ZONE )
                + warpSize - 1) / warpSize * warpSize, MIN_CM_ENTRIES );
}


HIP_GLOBAL void k_estimate_storage_bonds( reax_atom const * const my_atoms,
        single_body_parameters const * const sbp, two_body_parameters const * const tbp,
        control_params *control, reax_list far_nbr_list, 
        int num_atom_types, int n, int N, int total_cap,
        int * const bonds, int * const max_bonds )
{
    int i, j, pj; 
    int start_i, end_i;
    int type_i, type_j, tbp_ij;
    int num_bonds;
    real cutoff, r_ij; 
    real C12, C34, C56;
    real BO_s, BO_pi, BO_pi2;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= total_cap )
    {
        return;
    }

    num_bonds = 0;

    if ( i < N )
    {
        type_i = my_atoms[i].type;
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );

        if ( i < n )
        {
            cutoff = MIN( control->nonb_cut, control->bond_cut );
        }
        else
        {
            cutoff = control->bond_cut;
        }

        for ( pj = start_i; pj < end_i; ++pj )
        { 
            if ( far_nbr_list.far_nbr_list.d[pj] <= cutoff )
            {
                j = far_nbr_list.far_nbr_list.nbr[pj];
                type_j = my_atoms[j].type;
                r_ij = far_nbr_list.far_nbr_list.d[pj];
                tbp_ij = index_tbp(type_i ,type_j, num_atom_types);

                /* uncorrected bond orders */
                if ( sbp[type_i].r_s > 0.0 && sbp[type_j].r_s > 0.0 )
                {
                    C12 = tbp[tbp_ij].p_bo1 * POW( r_ij / tbp[tbp_ij].r_s, tbp[tbp_ij].p_bo2 );
                    BO_s = (1.0 + control->bo_cut) * EXP( C12 );
                }
                else
                {
                    C12 = 0.0;
                    BO_s = 0.0;
                }

                if ( sbp[type_i].r_pi > 0.0 && sbp[type_j].r_pi > 0.0 )
                {
                    C34 = tbp[tbp_ij].p_bo3 * POW( r_ij / tbp[tbp_ij].r_p, tbp[tbp_ij].p_bo4 );
                    BO_pi = EXP( C34 );
                }
                else
                {
                    C34 = 0.0;
                    BO_pi = 0.0;
                }

                if ( sbp[type_i].r_pi_pi > 0.0 && sbp[type_j].r_pi_pi > 0.0 )
                {
                    C56 = tbp[tbp_ij].p_bo5 * POW( r_ij / tbp[tbp_ij].r_pp, tbp[tbp_ij].p_bo6 );
                    BO_pi2= EXP( C56 );
                }
                else
                {
                    C56 = 0.0;
                    BO_pi2 = 0.0;
                }

                /* initially BO values are the uncorrected ones, page 1 */
                if ( BO_s + BO_pi + BO_pi2 >= control->bo_cut )
                {
                    ++num_bonds;
                }
            }
        }
    }

    __syncthreads( );

    bonds[i] = num_bonds;
    max_bonds[i] = MAX( (int) CEIL(2 * num_bonds * SAFE_ZONE), MIN_BONDS );
}


HIP_GLOBAL void k_estimate_storage_hbonds( reax_atom const * const my_atoms,
        single_body_parameters const * const sbp, control_params const * const control,
        reax_list far_nbr_list, int num_atom_types, int n, int N,
        int total_cap, int * const hbonds, int * const max_hbonds )
{
    int i, j, pj; 
    int start_i, end_i;
    int type_i, type_j;
    int ihb, jhb;
    int num_hbonds;
    real cutoff;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= total_cap )
    {
        return;
    }

    num_hbonds = 0;

    if ( i < N )
    {
        type_i = my_atoms[i].type;
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );
        ihb = sbp[type_i].p_hbond;

        if ( i < n )
        { 
            cutoff = control->nonb_cut;
        }   
        else
        {
            cutoff = control->bond_cut;
        } 

        if ( i < n && ihb == H_ATOM || ihb == H_BONDING_ATOM )
        {
            for ( pj = start_i; pj < end_i; ++pj )
            { 
                j = far_nbr_list.far_nbr_list.nbr[pj];
                type_j = my_atoms[j].type;
                jhb = sbp[type_j].p_hbond;

                /* atom i: H bonding, ghost
                 * atom j: H atom, native */
                if ( i >= n && j < n && ihb == H_BONDING_ATOM && jhb == H_ATOM
                        && far_nbr_list.far_nbr_list.d[pj] <= control->nonb_cut
                        && far_nbr_list.far_nbr_list.d[pj] <= control->hbond_cut )
                {
                    ++num_hbonds;
                }
                else if ( i < n && far_nbr_list.far_nbr_list.d[pj] <= cutoff
                        && far_nbr_list.far_nbr_list.d[pj] <= control->hbond_cut )
                {
                    /* atom i: H atom, native
                     * atom j: H bonding atom */
                    if ( ihb == H_ATOM && jhb == H_BONDING_ATOM )
                    {
                        ++num_hbonds;
                    }
                    /* atom i: H bonding atom, native
                     * atom j: H atom, native */
                    else if ( ihb == H_BONDING_ATOM && jhb == H_ATOM && j < n )
                    {
                        ++num_hbonds;
                    }
                }
            }
        }
    }

    __syncthreads( );

    hbonds[i] = num_hbonds;
    max_hbonds[i] = MAX( (int) CEIL(num_hbonds * SAFE_ZONE), MIN_HBONDS );
}


HIP_GLOBAL void k_update_sym_dbond_indices( reax_list bond_list, int N )
{
    int i, pj, pk, nbr_ij, nbr_jk;
    bond_data *ibond, *jbond;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    /* i-j bonds */
    for ( pj = Start_Index(i, &bond_list); pj < End_Index(i, &bond_list); ++pj )
    {
        ibond = &bond_list.bond_list[pj];
        nbr_ij = ibond->nbr;

        /* j-k bonds */
        for ( pk = Start_Index(nbr_ij, &bond_list); pk < End_Index(nbr_ij, &bond_list); ++pk )
        {
            jbond = &bond_list.bond_list[pk];
            nbr_jk = jbond->nbr;

            if ( i == nbr_jk && i > nbr_ij )
            {
                ibond->dbond_index = pj;
                jbond->dbond_index = pj;

                ibond->sym_index = pk;
                jbond->sym_index = pj;
                break;
            }
        }
    }
}


#if !defined(HIP_ACCUM_ATOMIC)
HIP_GLOBAL void k_update_sym_hbond_indices_opt( reax_atom *my_atoms,
        reax_list hbond_list, int N )
{
    int i, pj, pk;
    int nbr, nbrstart, nbrend;
    int start, end;
    hbond_data *ihbond, *jhbond;
    int thread_id, lane_id;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    i = thread_id / warpSize;

    if ( i > N )
    {
        return;
    }

    lane_id = thread_id % warpSize; 
    start = Start_Index( my_atoms[i].Hindex, &hbond_list );
    end = End_Index( my_atoms[i].Hindex, &hbond_list );
    pj = start + lane_id;

    while ( pj < end )
    {
        ihbond = &hbond_list.hbond_list[pj];
        nbr = ihbond->nbr;

        nbrstart = Start_Index( my_atoms[nbr].Hindex, &hbond_list );
        nbrend = End_Index( my_atoms[nbr].Hindex, &hbond_list );

        for ( pk = nbrstart; pk < nbrend; pk++ )
        {
            jhbond = &hbond_list.hbond_list[pk];

            if ( jhbond->nbr == i )
            {
                ihbond->sym_index = pk;
                jhbond->sym_index = pj;
                break;
            }
        }

        pj += warpSize;
    }
}
#endif


#if defined(DEBUG_FOCUS)
HIP_GLOBAL void k_print_forces( reax_atom *my_atoms, rvec *f, int n )
{
    int i; 

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    printf( "%8d: %24.15f, %24.15f, %24.15f\n",
            my_atoms[i].orig_id, f[i][0], f[i][1], f[i][2] );
}


HIP_GLOBAL void k_print_hbonds( reax_atom *my_atoms, reax_list hbond_list, int n, int rank, int step )
{
    int i, k, pj, start, end; 
    hbond_data *hbond_jk;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    start = Start_Index( my_atoms[i].Hindex, &hbond_list );
    end = End_Index( my_atoms[i].Hindex, &hbond_list );

    for ( pj = start; pj < end; ++pj )
    {
        k = hbond_list.hbond_list[pj].nbr;
        hbond_jk = &hbond_list.hbond_list[pj];

#if !defined(HIP_ACCUM_ATOMIC)
        printf( "p%03d, step %05d: %8d: %8d, %24.15f, %24.15f, %24.15f\n",
                rank, step, my_atoms[i].Hindex, k,
                hbond_jk->hb_f[0],
                hbond_jk->hb_f[1],
                hbond_jk->hb_f[2] );
#else
        printf( "p%03d, step %05d: %8d: %8d\n",
                rank, step, my_atoms[i].Hindex, k );
#endif
    }
}
#endif


#if defined(DEBUG_FOCUS)
static void Print_Forces( reax_system *system, control_params *control )
{
    int blocks;
    
    blocks = (system->n) / DEF_BLOCK_SIZE
        + (((system->n % DEF_BLOCK_SIZE) == 0) ? 0 : 1);

    k_print_forces <<< blocks, DEF_BLOCK_SIZE, 0, control->streams[0] >>>
        ( system->d_my_atoms, workspace->d_workspace->f, system->n );
    hipCheckError( );
}


static void Print_HBonds( reax_system *system, control_params *control,
        int step )
{
    int blocks;
    
    blocks = (system->n) / DEF_BLOCK_SIZE
        + (((system->n % DEF_BLOCK_SIZE) == 0) ? 0 : 1);

    k_print_hbonds <<< blocks, DEF_BLOCK_SIZE, 0, control->streams[0] >>>
        ( system->d_my_atoms, *(lists[HBONDS]), system->n, system->my_rank, step );
    hipCheckError( );
}
#endif


/* Initialize indices for far neighbors list post reallocation
 *
 * system: atomic system info. */
void Hip_Init_Neighbor_Indices( reax_system *system, control_params *control,
        reax_list *far_nbr_list )
{
    int blocks;

    blocks = far_nbr_list->n / DEF_BLOCK_SIZE
        + (far_nbr_list->n % DEF_BLOCK_SIZE == 0 ? 0 : 1);

    /* init indices */
    Hip_Scan_Excl_Sum( system->d_max_far_nbrs, far_nbr_list->index,
            far_nbr_list->n, 0, control->streams[0] );

    /* init end_indices */
    k_init_end_index <<< blocks, DEF_BLOCK_SIZE, 0, control->streams[0] >>>
        ( system->d_far_nbrs, far_nbr_list->index, far_nbr_list->end_index,
          far_nbr_list->n );
    hipCheckError( );
}


/* Initialize indices for far hydrogen bonds list post reallocation
 *
 * system: atomic system info. */
void Hip_Init_HBond_Indices( reax_system *system, storage *workspace,
        reax_list *hbond_list, hipStream_t s )
{
    int blocks, *temp;

    blocks = system->total_cap / DEF_BLOCK_SIZE
        + (system->total_cap % DEF_BLOCK_SIZE == 0 ? 0 : 1);

    sHipCheckMalloc( &workspace->scratch[2], &workspace->scratch_size[2],
            sizeof(int) * system->total_cap, __FILE__, __LINE__ );
    temp = (int *) workspace->scratch[2];

    /* init indices and end_indices */
    Hip_Scan_Excl_Sum( system->d_max_hbonds, temp, system->total_cap, 2, s );

    k_init_hbond_indices <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, system->d_hbonds, temp, 
          hbond_list->index, hbond_list->end_index, system->total_cap );
    hipCheckError( );
}


/* Initialize indices for far bonds list post reallocation
 *
 * system: atomic system info. */
void Hip_Init_Bond_Indices( reax_system *system, reax_list * bond_list,
        hipStream_t s )
{
    int blocks;

    blocks = system->total_cap / DEF_BLOCK_SIZE + 
        (system->total_cap % DEF_BLOCK_SIZE == 0 ? 0 : 1);

    /* init indices */
    Hip_Scan_Excl_Sum( system->d_max_bonds, bond_list->index,
            system->total_cap, 1, s );

    /* init end_indices */
    k_init_end_index <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( system->d_bonds, bond_list->index, bond_list->end_index, system->total_cap );
    hipCheckError( );
}


/* Initialize indices for charge matrix post reallocation
 *
 * system: atomic system info.
 * H: charge matrix */
void Hip_Init_Sparse_Matrix_Indices( reax_system *system, sparse_matrix *H,
        hipStream_t s )
{
    int blocks;

    blocks = H->n_max / DEF_BLOCK_SIZE
        + (H->n_max % DEF_BLOCK_SIZE == 0 ? 0 : 1);

    /* init indices */
    Hip_Scan_Excl_Sum( system->d_max_cm_entries, H->start, H->n_max, 4, s );

    //TODO: not needed for full format (Init_Forces sets H->end)
    /* init end_indices */
    k_init_end_index <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( system->d_cm_entries, H->start, H->end, H->n_max );
    hipCheckError( );
}


void Hip_Estimate_Storages( reax_system *system, control_params *control,
        storage *workspace, reax_list **lists, int realloc_cm,
        int realloc_bonds, int realloc_hbonds, int step )
{
    int blocks;

    if ( realloc_cm == TRUE )
    {
        blocks = workspace->d_workspace->H.n_max / DEF_BLOCK_SIZE
            + (workspace->d_workspace->H.n_max % DEF_BLOCK_SIZE == 0 ? 0 : 1);

        if ( workspace->d_workspace->H.format == SYM_HALF_MATRIX )
        {
            k_estimate_storages_cm_half <<< blocks, DEF_BLOCK_SIZE, 0,
                                        control->streams[4] >>>
                ( system->d_my_atoms, (control_params *) control->d_control_params,
                  *(lists[FAR_NBRS]), workspace->d_workspace->H.n,
                  workspace->d_workspace->H.n_max,
                  system->d_cm_entries, system->d_max_cm_entries );
        }
        else
        {
            k_estimate_storages_cm_full <<< blocks, DEF_BLOCK_SIZE, 0,
                                        control->streams[4] >>>
                ( (control_params *) control->d_control_params,
                  *(lists[FAR_NBRS]), workspace->d_workspace->H.n,
                  workspace->d_workspace->H.n_max,
                  system->d_cm_entries, system->d_max_cm_entries );
        }
        hipCheckError( );

        Hip_Reduction_Sum( system->d_max_cm_entries, system->d_total_cm_entries,
                workspace->d_workspace->H.n_max, 4, control->streams[4] );
        sHipMemcpyAsync( &system->total_cm_entries, system->d_total_cm_entries,
                sizeof(int), hipMemcpyDeviceToHost, control->streams[4], __FILE__, __LINE__ );
    }

    if ( realloc_bonds == TRUE )
    {
        blocks = system->total_cap / DEF_BLOCK_SIZE
            + (system->total_cap % DEF_BLOCK_SIZE == 0 ? 0 : 1);

        k_estimate_storage_bonds <<< blocks, DEF_BLOCK_SIZE, 0,
                                 control->streams[1] >>>
            ( system->d_my_atoms, system->reax_param.d_sbp, system->reax_param.d_tbp, 
              (control_params *) control->d_control_params,
              *(lists[FAR_NBRS]), system->reax_param.num_atom_types,
              system->n, system->N, system->total_cap,
              system->d_bonds, system->d_max_bonds );
        hipCheckError( );

        Hip_Reduction_Sum( system->d_max_bonds, system->d_total_bonds,
                system->total_cap, 1, control->streams[1] );
        sHipMemcpyAsync( &system->total_bonds, system->d_total_bonds, sizeof(int), 
                hipMemcpyDeviceToHost, control->streams[1], __FILE__, __LINE__ );
    }

    if ( system->numH > 0 && control->hbond_cut > 0.0 && realloc_hbonds == TRUE )
    {
        blocks = system->total_cap / DEF_BLOCK_SIZE
            + (system->total_cap % DEF_BLOCK_SIZE == 0 ? 0 : 1);

        k_estimate_storage_hbonds <<< blocks, DEF_BLOCK_SIZE, 0,
                                  control->streams[2] >>>
            ( system->d_my_atoms, system->reax_param.d_sbp,
              (control_params *) control->d_control_params,
              *(lists[FAR_NBRS]), system->reax_param.num_atom_types,
              system->n, system->N, system->total_cap,
              system->d_hbonds, system->d_max_hbonds );
        hipCheckError( );

        Hip_Reduction_Sum( system->d_max_hbonds, system->d_total_hbonds,
                system->total_cap, 2, control->streams[2] );
        sHipMemcpyAsync( &system->total_hbonds, system->d_total_hbonds, sizeof(int), 
                hipMemcpyDeviceToHost, control->streams[2], __FILE__, __LINE__ );
    }
    else if ( step == 0 && (system->numH == 0 || control->hbond_cut <= 0.0) )
    {
#if defined(DEBUG_FOCUS)
        if ( system->numH == 0 )
        {
            fprintf( stderr, "[INFO] DISABLING HYDROGEN BOND COMPUTATION: NO HYDROGEN ATOMS FOUND\n" );
        }

        if ( control->hbond_cut <= 0.0 )
        {
            fprintf( stderr, "[INFO] DISABLING HYDROGEN BOND COMPUTATION: BOND CUTOFF LENGTH IS ZERO\n" );
        }
#endif

        control->hbond_cut = 0.0;
        k_disable_hydrogen_bonding <<< 1, 1, 0, control->streams[2] >>>
            ( (control_params *) control->d_control_params );
    }

    hipStreamSynchronize( control->streams[4] );
    hipStreamSynchronize( control->streams[1] );
    hipStreamSynchronize( control->streams[2] );
}


/* Initialize the bond list, hydrogen bond list, and charge matrix
 * data structures along with updating the pairwise distances in the
 * far neighbor (Verlet) list if required.
 *
 * NOTE: the control flow of the code follows a
 * try-compute-else-reallocate-and-retry logic which requires that
 * the initialization kernels be atomic transactions. Locks are used
 * to mark if the transaction succeeds (and thus should not be repeated).
 */
int Hip_Init_Forces( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control ) 
{
    int renbr, blocks, ret, realloc_cm, realloc_bonds, realloc_hbonds;
    static int dist_done = FALSE, cm_done = FALSE, bonds_done = FALSE, hbonds_done = FALSE;
#if defined(LOG_PERFORMANCE)
    float time_elapsed;
    static hipEvent_t time_event[8];
    static int time_events_alloc = FALSE;
    
    if ( time_events_alloc == FALSE )
    {
        for ( int i = 0; i < 8; ++i )
        {
            hipEventCreate( &time_event[i] );
        }

        time_events_alloc = TRUE;
    }
#endif

    renbr = (data->step - data->prev_steps) % control->reneighbor == 0 ? TRUE : FALSE;

    /* reset reallocation flags on device */
    if ( cm_done == FALSE )
    {
        sHipMemsetAsync( system->d_realloc_cm_entries, FALSE, sizeof(int), 
                control->streams[4], __FILE__, __LINE__ );
    }
    if ( bonds_done == FALSE )
    {
        sHipMemsetAsync( system->d_realloc_bonds, FALSE, sizeof(int), 
                control->streams[1], __FILE__, __LINE__ );
    }
    if ( hbonds_done == FALSE )
    {
        sHipMemsetAsync( system->d_realloc_hbonds, FALSE, sizeof(int), 
                control->streams[2], __FILE__, __LINE__ );
    }

#if defined(LOG_PERFORMANCE)
    hipEventRecord( time_event[0], control->streams[0] );
#endif

    if ( renbr == FALSE && dist_done == FALSE )
    {
//        k_init_dist <<< control->blocks_n, control->block_size_n, 0, control->streams[0] >>>
//            ( system->d_my_atoms, *(lists[FAR_NBRS]), system->N );

        blocks = system->N * warpSize / DEF_BLOCK_SIZE
            + (system->N * warpSize % DEF_BLOCK_SIZE == 0 ? 0 : 1);

        k_init_dist_opt <<< blocks, DEF_BLOCK_SIZE, 0, control->streams[0] >>>
            ( system->d_my_atoms, *(lists[FAR_NBRS]), system->N );
        hipCheckError( );

        hipEventRecord( control->stream_events[0], control->streams[0] );

        dist_done = TRUE;
    }

#if defined(LOG_PERFORMANCE)
    hipEventRecord( time_event[1], control->streams[0] );
    hipEventRecord( time_event[2], control->streams[4] );
#endif

    if ( cm_done == FALSE )
    {
        blocks = workspace->d_workspace->H.n_max / DEF_BLOCK_SIZE
            + (workspace->d_workspace->H.n_max % DEF_BLOCK_SIZE == 0 ? 0 : 1);

        /* update num. rows in matrix for this GPU */
        workspace->d_workspace->H.n = system->n;

        Hip_Init_Sparse_Matrix_Indices( system, &workspace->d_workspace->H,
                control->streams[4] );

        hipStreamWaitEvent( control->streams[4], control->stream_events[0], 0 );

        if ( workspace->d_workspace->H.format == SYM_HALF_MATRIX )
        {
            if ( control->tabulate <= 0 )
            {
                k_init_cm_half_fs <<< blocks, DEF_BLOCK_SIZE, 0, control->streams[4] >>>
                    ( system->d_my_atoms, system->reax_param.d_sbp, system->reax_param.d_tbp,
                      *(workspace->d_workspace), (control_params *) control->d_control_params,
                      *(lists[FAR_NBRS]), system->reax_param.num_atom_types,
                      system->d_max_cm_entries, system->d_realloc_cm_entries );
            }
            else
            {
                k_init_cm_half_fs_tab <<< blocks, DEF_BLOCK_SIZE, 0, control->streams[4] >>>
                    ( system->d_my_atoms, system->reax_param.d_sbp,
                      *(workspace->d_workspace), (control_params *) control->d_control_params,
                      *(lists[FAR_NBRS]), workspace->d_LR, system->reax_param.num_atom_types,
                      system->d_max_cm_entries, system->d_realloc_cm_entries );
            }
        }
        else
        {
            if ( control->tabulate <= 0 )
            {
//                k_init_cm_full_fs <<< blocks, DEF_BLOCK_SIZE, 0, control->streams[4] >>>
//                    ( system->d_my_atoms, system->reax_param.d_sbp, system->reax_param.d_tbp,
//                      *(workspace->d_workspace), (control_params *) control->d_control_params,
//                      *(lists[FAR_NBRS]), system->reax_param.num_atom_types,
//                      system->d_max_cm_entries, system->d_realloc_cm_entries );

                blocks = workspace->d_workspace->H.n_max * warpSize / DEF_BLOCK_SIZE
                    + (workspace->d_workspace->H.n_max * warpSize % DEF_BLOCK_SIZE == 0 ? 0 : 1);

                k_init_cm_full_fs_opt <<< blocks, DEF_BLOCK_SIZE,
                           sizeof(hipcub::WarpScan<int>::TempStorage) * (DEF_BLOCK_SIZE / warpSize),
                           control->streams[4] >>>
                    ( system->d_my_atoms, system->reax_param.d_sbp, system->reax_param.d_tbp,
                      *(workspace->d_workspace), (control_params *) control->d_control_params,
                      *(lists[FAR_NBRS]), system->reax_param.num_atom_types,
                      system->d_max_cm_entries, system->d_realloc_cm_entries );
            }
            else
            {
                k_init_cm_full_fs_tab <<< blocks, DEF_BLOCK_SIZE, 0,
                                      control->streams[4] >>>
                    ( system->d_my_atoms, system->reax_param.d_sbp,
                      *(workspace->d_workspace), (control_params *) control->d_control_params,
                      *(lists[FAR_NBRS]), workspace->d_LR, system->reax_param.num_atom_types,
                      system->d_max_cm_entries, system->d_realloc_cm_entries );
            }
        }
        hipCheckError( );

        hipEventRecord( control->stream_events[1], control->streams[4] );
    }

#if defined(LOG_PERFORMANCE)
    hipEventRecord( time_event[3], control->streams[4] );
    hipEventRecord( time_event[4], control->streams[1] );
#endif

    if ( bonds_done == FALSE )
    {
        blocks = system->total_cap / DEF_BLOCK_SIZE
            + ((system->total_cap % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);

        k_reset_bond_orders <<< blocks, DEF_BLOCK_SIZE, 0, control->streams[1] >>>
            ( *(workspace->d_workspace), system->total_cap );
        hipCheckError( );

        Hip_Init_Bond_Indices( system, lists[BONDS], control->streams[1] );

        hipStreamWaitEvent( control->streams[1], control->stream_events[0], 0 );

//        k_init_bonds <<< control->blocks_n, control->block_size_n, 0, control->streams[1] >>>
//            ( system->d_my_atoms, system->reax_param.d_sbp,
//              system->reax_param.d_tbp, *(workspace->d_workspace),
//              (control_params *) control->d_control_params,
//              *(lists[FAR_NBRS]), *(lists[BONDS]),
//              system->n, system->N, system->reax_param.num_atom_types,
//              system->d_max_bonds, system->d_realloc_bonds );
//        hipCheckError( );

        blocks = system->N * warpSize / DEF_BLOCK_SIZE
            + (system->N * warpSize % DEF_BLOCK_SIZE == 0 ? 0 : 1);
        k_init_bonds_opt <<< blocks, DEF_BLOCK_SIZE,
                           sizeof(hipcub::WarpScan<int>::TempStorage) * (DEF_BLOCK_SIZE / warpSize),
                           control->streams[1] >>>
            ( system->d_my_atoms, system->reax_param.d_sbp,
              system->reax_param.d_tbp, *(workspace->d_workspace),
              (control_params *) control->d_control_params,
              *(lists[FAR_NBRS]), *(lists[BONDS]),
              system->n, system->N, system->reax_param.num_atom_types,
              system->d_max_bonds, system->d_realloc_bonds );
        hipCheckError( );

        hipEventRecord( control->stream_events[1], control->streams[1] );
    }

#if defined(LOG_PERFORMANCE)
    hipEventRecord( time_event[5], control->streams[1] );
    hipEventRecord( time_event[6], control->streams[2] );
#endif

    if ( control->hbond_cut > 0.0 && system->numH > 0 && hbonds_done == FALSE )
    {
        Hip_Init_HBond_Indices( system, workspace, lists[HBONDS],
                control->streams[2] );

        hipStreamWaitEvent( control->streams[2], control->stream_events[0], 0 );

//        k_init_hbonds <<< control->blocks_n, control->block_size_n, 0, control->streams[2] >>>
//            ( system->d_my_atoms, system->reax_param.d_sbp,
//              (control_params *) control->d_control_params,
//              *(lists[FAR_NBRS]), *(lists[HBONDS]),
//              system->n, system->N, system->reax_param.num_atom_types,
//              system->d_max_hbonds, system->d_realloc_hbonds );
//        hipCheckError( );

        blocks = system->N * warpSize / DEF_BLOCK_SIZE
            + (system->N * warpSize % DEF_BLOCK_SIZE == 0 ? 0 : 1);

        k_init_hbonds_opt <<< blocks, DEF_BLOCK_SIZE,
                           sizeof(hipcub::WarpScan<int>::TempStorage) * (DEF_BLOCK_SIZE / warpSize),
                           control->streams[2] >>>
            ( system->d_my_atoms, system->reax_param.d_sbp,
              (control_params *) control->d_control_params,
              *(lists[FAR_NBRS]), *(lists[HBONDS]),
              system->n, system->N, system->reax_param.num_atom_types,
              system->d_max_hbonds, system->d_realloc_hbonds );
        hipCheckError( );

        hipEventRecord( control->stream_events[3], control->streams[2] );
    }
#if defined(LOG_PERFORMANCE)
    hipEventRecord( time_event[7], control->streams[2] );
#endif

    /* check reallocation flags on device */
    if ( cm_done == FALSE )
    {
        sHipMemcpyAsync( &realloc_cm, system->d_realloc_cm_entries, sizeof(int), 
                hipMemcpyDeviceToHost, control->streams[4], __FILE__, __LINE__ );
    }
    else
    {
        realloc_cm = FALSE;
    }
    if ( bonds_done == FALSE )
    {
        sHipMemcpyAsync( &realloc_bonds, system->d_realloc_bonds, sizeof(int), 
                hipMemcpyDeviceToHost, control->streams[1], __FILE__, __LINE__ );
    }
    else
    {
        realloc_bonds = FALSE;
    }
    if ( hbonds_done == FALSE )
    {
        sHipMemcpyAsync( &realloc_hbonds, system->d_realloc_hbonds, sizeof(int), 
                hipMemcpyDeviceToHost, control->streams[2], __FILE__, __LINE__ );
    }
    else
    {
        realloc_hbonds = FALSE;
    }

    hipStreamSynchronize( control->streams[0] );
    hipStreamSynchronize( control->streams[4] );
    hipStreamSynchronize( control->streams[1] );
    hipStreamSynchronize( control->streams[2] );

#if defined(LOG_PERFORMANCE)
    if ( hipEventQuery( time_event[0] ) != hipSuccess ) 
    {
        hipEventSynchronize( time_event[0] );
    }

    if ( hipEventQuery( time_event[1] ) != hipSuccess ) 
    {
        hipEventSynchronize( time_event[1] );
    }

    hipEventElapsedTime( &time_elapsed, time_event[0], time_event[1] ); 
    data->timing.init_dist += (real) (time_elapsed / 1000.0);

    if ( hipEventQuery( time_event[2] ) != hipSuccess ) 
    {
        hipEventSynchronize( time_event[2] );
    }

    if ( hipEventQuery( time_event[3] ) != hipSuccess ) 
    {
        hipEventSynchronize( time_event[3] );
    }

    hipEventElapsedTime( &time_elapsed, time_event[2], time_event[3] ); 
    data->timing.init_cm += (real) (time_elapsed / 1000.0);

    if ( hipEventQuery( time_event[4] ) != hipSuccess ) 
    {
        hipEventSynchronize( time_event[4] );
    }

    if ( hipEventQuery( time_event[5] ) != hipSuccess ) 
    {
        hipEventSynchronize( time_event[5] );
    }

    hipEventElapsedTime( &time_elapsed, time_event[4], time_event[5] ); 
    data->timing.init_bond += (real) (time_elapsed / 1000.0);

    if ( hipEventQuery( time_event[6] ) != hipSuccess ) 
    {
        hipEventSynchronize( time_event[6] );
    }

    if ( hipEventQuery( time_event[7] ) != hipSuccess ) 
    {
        hipEventSynchronize( time_event[7] );
    }

    hipEventElapsedTime( &time_elapsed, time_event[6], time_event[7] ); 
    data->timing.init_bond += (real) (time_elapsed / 1000.0);
#endif

    ret = (realloc_cm == FALSE && realloc_bonds == FALSE && realloc_hbonds == FALSE
            ? SUCCESS : FAILURE);

    if ( realloc_cm == FALSE )
    {
        cm_done = TRUE;
    }
    if ( realloc_bonds == FALSE )
    {
        bonds_done = TRUE;
    }
    if ( realloc_hbonds == FALSE )
    {
        hbonds_done = TRUE;
    }

    if ( ret == SUCCESS )
    {
        k_update_sym_dbond_indices <<< control->blocks_n, control->block_size_n,
                                   0, control->streams[0] >>> 
            ( *(lists[BONDS]), system->N );
        hipCheckError( );

#if !defined(HIP_ACCUM_ATOMIC)
        if ( control->hbond_cut > 0.0 && system->numH > 0 )
        {
            blocks = system->N * warpSize / DEF_BLOCK_SIZE
                + (system->N * warpSize % DEF_BLOCK_SIZE == 0 ? 0 : 1);

            /* make hbond_list symmetric */
            k_update_sym_hbond_indices_opt <<< blocks, DEF_BLOCK_SIZE,
                                           0, control->streams[0] >>>
                ( system->d_my_atoms, *(lists[HBONDS]), system->N );
            hipCheckError( );
        }
#endif

        dist_done = FALSE;
        cm_done = FALSE;
        bonds_done = FALSE;
        hbonds_done = FALSE;
    }
    else
    {
        Hip_Estimate_Storages( system, control, workspace, lists,
               realloc_cm, realloc_bonds, realloc_hbonds,
               data->step - data->prev_steps );

        /* schedule reallocations after updating allocation sizes */
        workspace->d_workspace->realloc.cm = realloc_cm;
        workspace->d_workspace->realloc.bonds = realloc_bonds;
        workspace->d_workspace->realloc.hbonds = realloc_hbonds;
    }

    return ret;
}


int Hip_Init_Forces_No_Charges( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control ) 
{
    int renbr, blocks, ret, realloc_bonds, realloc_hbonds;
    static int dist_done = FALSE, bonds_done = FALSE, hbonds_done = FALSE;
#if defined(LOG_PERFORMANCE)
    float time_elapsed;
    static hipEvent_t time_event[3];
    static int time_events_alloc = FALSE;
    
    if ( time_events_alloc == FALSE )
    {
        for ( int i = 0; i < 3; ++i )
        {
            hipEventCreate( &time_event[i] );
        }

        time_events_alloc = TRUE;
    }
#endif

    renbr = (data->step - data->prev_steps) % control->reneighbor == 0 ? TRUE : FALSE;

    /* reset reallocation flags on device */
    sHipMemsetAsync( system->d_realloc_bonds, FALSE, sizeof(int), 
            control->streams[0], __FILE__, __LINE__ );
    sHipMemsetAsync( system->d_realloc_hbonds, FALSE, sizeof(int), 
            control->streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->streams[0] );

#if defined(LOG_PERFORMANCE)
    hipEventRecord( time_event[0], control->streams[0] );
#endif

    if ( renbr == FALSE && dist_done == FALSE )
    {
//        k_init_dist <<< control->blocks_n, control->block_size_n, 0, control->streams[0] >>>
//            ( system->d_my_atoms, *(lists[FAR_NBRS]), system->N );

        blocks = system->N * warpSize / DEF_BLOCK_SIZE
            + (system->N * warpSize % DEF_BLOCK_SIZE == 0 ? 0 : 1);

        k_init_dist_opt <<< blocks, DEF_BLOCK_SIZE, 0, control->streams[0] >>>
            ( system->d_my_atoms, *(lists[FAR_NBRS]), system->N );
        hipCheckError( );

        dist_done = TRUE;
    }

#if defined(LOG_PERFORMANCE)
    hipEventRecord( time_event[1], control->streams[0] );
#endif

    if ( bonds_done == FALSE )
    {
        blocks = system->total_cap / DEF_BLOCK_SIZE
            + ((system->total_cap % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);

        k_reset_bond_orders <<< blocks, DEF_BLOCK_SIZE, 0, control->streams[0] >>>
            ( *(workspace->d_workspace), system->total_cap );
        hipCheckError( );

        Hip_Init_Bond_Indices( system, lists[BONDS], control->streams[0] );

//        k_init_bonds <<< control->blocks_n, control->block_size_n, 0, control->streams[0] >>>
//            ( system->d_my_atoms, system->reax_param.d_sbp,
//              system->reax_param.d_tbp, *(workspace->d_workspace),
//              (control_params *) control->d_control_params,
//              *(lists[FAR_NBRS]), *(lists[BONDS]),
//              system->n, system->N, system->reax_param.num_atom_types,
//              system->d_max_bonds, system->d_realloc_bonds );

        blocks = control->block_size_n * warpSize / DEF_BLOCK_SIZE
            + (control->block_size_n * warpSize % DEF_BLOCK_SIZE == 0 ? 0 : 1);

        k_init_bonds_opt <<< blocks, DEF_BLOCK_SIZE,
                     sizeof(hipcub::WarpScan<int>::TempStorage) * (DEF_BLOCK_SIZE / warpSize),
                     control->streams[0] >>>
            ( system->d_my_atoms, system->reax_param.d_sbp,
              system->reax_param.d_tbp, *(workspace->d_workspace),
              (control_params *) control->d_control_params,
              *(lists[FAR_NBRS]), *(lists[BONDS]),
              system->n, system->N, system->reax_param.num_atom_types,
              system->d_max_bonds, system->d_realloc_bonds );
        hipCheckError( );
    }

    if ( control->hbond_cut > 0.0 && system->numH > 0 && hbonds_done == FALSE )
    {
        Hip_Init_HBond_Indices( system, workspace, lists[HBONDS],
                control->streams[0] );

//        k_init_hbonds <<< control->blocks_n, control->block_size_n, 0, control->streams[0] >>>
//            ( system->d_my_atoms, system->reax_param.d_sbp,
//              (control_params *) control->d_control_params,
//              *(lists[FAR_NBRS]), *(lists[HBONDS]),
//              system->n, system->N, system->reax_param.num_atom_types,
//              system->d_max_hbonds, system->d_realloc_hbonds );
//        hipCheckError( );

        blocks = system->N * warpSize / DEF_BLOCK_SIZE
            + (system->N * warpSize % DEF_BLOCK_SIZE == 0 ? 0 : 1);

        k_init_hbonds_opt <<< blocks, DEF_BLOCK_SIZE,
                          sizeof(hipcub::WarpScan<int>::TempStorage) * (DEF_BLOCK_SIZE / warpSize),
                          control->streams[0] >>>
            ( system->d_my_atoms, system->reax_param.d_sbp,
              (control_params *) control->d_control_params,
              *(lists[FAR_NBRS]), *(lists[HBONDS]),
              system->n, system->N, system->reax_param.num_atom_types,
              system->d_max_hbonds, system->d_realloc_hbonds );
        hipCheckError( );
    }

#if defined(LOG_PERFORMANCE)
    hipEventRecord( time_event[2], control->streams[0] );
#endif

    /* check reallocation flags on device */
    if ( bonds_done == FALSE )
    {
        sHipMemcpyAsync( &realloc_bonds, system->d_realloc_bonds, sizeof(int), 
                hipMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );
    }
    else
    {
        realloc_bonds = FALSE;
    }
    if ( hbonds_done == FALSE )
    {
        sHipMemcpyAsync( &realloc_hbonds, system->d_realloc_hbonds, sizeof(int), 
                hipMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );
    }
    else
    {
        realloc_hbonds = FALSE;
    }

    hipStreamSynchronize( control->streams[0] );

#if defined(LOG_PERFORMANCE)
    if ( hipEventQuery( time_event[0] ) != hipSuccess ) 
    {
        hipEventSynchronize( time_event[0] );
    }

    if ( hipEventQuery( time_event[1] ) != hipSuccess ) 
    {
        hipEventSynchronize( time_event[1] );
    }

    hipEventElapsedTime( &time_elapsed, time_event[0], time_event[1] ); 
    data->timing.init_dist += (real) (time_elapsed / 1000.0);

    if ( hipEventQuery( time_event[2] ) != hipSuccess ) 
    {
        hipEventSynchronize( time_event[2] );
    }

    hipEventElapsedTime( &time_elapsed, time_event[1], time_event[2] ); 
    data->timing.init_bond += (real) (time_elapsed / 1000.0);
#endif

    ret = (realloc_bonds == FALSE && realloc_hbonds == FALSE
            ? SUCCESS : FAILURE);

    if ( realloc_bonds == FALSE )
    {
        bonds_done = TRUE;
    }
    if ( realloc_hbonds == FALSE )
    {
        hbonds_done = TRUE;
    }

    if ( ret == SUCCESS )
    {
        k_update_sym_dbond_indices <<< control->blocks_n, control->block_size_n,
                                   0, control->streams[0] >>> 
           ( *(lists[BONDS]), system->N );
        hipCheckError( );

#if !defined(HIP_ACCUM_ATOMIC)
        if ( control->hbond_cut > 0.0 && system->numH > 0 )
        {
            blocks = system->N * warpSize / DEF_BLOCK_SIZE
                + (system->N * warpSize % DEF_BLOCK_SIZE == 0 ? 0 : 1);

            /* make hbond_list symmetric */
            k_update_sym_hbond_indices_opt <<< blocks, DEF_BLOCK_SIZE,
                                           0, control->streams[0] >>>
                ( system->d_my_atoms, *(lists[HBONDS]), system->N );
            hipCheckError( );
        }
#endif

        dist_done = FALSE;
        bonds_done = FALSE;
        hbonds_done = FALSE;
    }
    else
    {
        Hip_Estimate_Storages( system, control, workspace, lists,
               FALSE, realloc_bonds, realloc_hbonds,
               data->step - data->prev_steps );

        /* schedule reallocations after updating allocation sizes */
        workspace->d_workspace->realloc.bonds = realloc_bonds;
        workspace->d_workspace->realloc.hbonds = realloc_hbonds;
    }

    return ret;
}


int Hip_Compute_Bonded_Forces( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, 
        reax_list **lists, output_controls *out_control )
{
    int ret;
    static int compute_bonded_part1 = FALSE;

    ret = SUCCESS;

    if ( compute_bonded_part1 == FALSE )
    {
        Hip_Compute_Bond_Orders( system, control, data, workspace, lists,
                out_control );

        Hip_Compute_Bonds( system, control, data, workspace, lists,
                out_control );

        Hip_Compute_Atom_Energy( system, control, data, workspace, lists,
                out_control );

        compute_bonded_part1 = TRUE;
    }

    ret = Hip_Compute_Valence_Angles( system, control, data, workspace,
            lists, out_control );

    if ( ret == SUCCESS )
    {
        Hip_Compute_Torsion_Angles( system, control, data, workspace, lists,
                out_control );

        if ( control->hbond_cut > 0.0 && system->numH > 0 )
        {
            Hip_Compute_Hydrogen_Bonds( system, control, data, workspace,
                    lists, out_control );
        }

        compute_bonded_part1 = FALSE;
    }

    return ret;
}


static void Hip_Compute_Total_Force( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, mpi_datatypes *mpi_data )
{
    rvec *f;

    smalloc_check( &workspace->host_scratch, &workspace->host_scratch_size,
            sizeof(rvec) * system->N, TRUE, SAFE_ZONE,
            __FILE__, __LINE__ );
    f = (rvec *) workspace->host_scratch;
    memset( f, 0, sizeof(rvec) * system->N );

    Hip_Total_Forces_Part1( system, control, data, workspace, lists );

    /* now all forces are computed to their partially-final values
     * based on the neighbors information each processor has had.
     * final values of force on each atom needs to be computed by adding up
     * all partially-final pieces */
    sHipMemcpyAsync( f, workspace->d_workspace->f, sizeof(rvec) * system->N,
            hipMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->streams[0] );

    Coll_FS( system, mpi_data, f, RVEC_PTR_TYPE, mpi_data->mpi_rvec );

    sHipMemcpyAsync( workspace->d_workspace->f, f, sizeof(rvec) * system->N,
            hipMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );

    Hip_Total_Forces_Part2( system, control, workspace );
}


extern "C" int Hip_Compute_Forces( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int i, charge_flag, ret;
    static int init_forces_done = FALSE;
#if defined(LOG_PERFORMANCE)
    float time_elapsed;
    static hipEvent_t time_event[8];
    static int time_events_alloc = FALSE;
    
    if ( time_events_alloc == FALSE )
    {
        for ( int i = 0; i < 8; ++i )
        {
            hipEventCreate( &time_event[i] );
        }

        time_events_alloc = TRUE;
    }
#endif

    ret = SUCCESS;

    if ( control->charge_freq > 0
            && (data->step - data->prev_steps) % control->charge_freq == 0 )
    {
        charge_flag = TRUE;
    }
    else
    {
        charge_flag = FALSE;
    }

#if defined(LOG_PERFORMANCE)
    hipEventRecord( time_event[0], control->streams[0] );
#endif

    if ( init_forces_done == FALSE )
    {
        if ( charge_flag == TRUE )
        {
            ret = Hip_Init_Forces( system, control, data,
                    workspace, lists, out_control );
        }
        else
        {
            ret = Hip_Init_Forces_No_Charges( system, control, data,
                    workspace, lists, out_control );
        }

        if ( ret == SUCCESS )
        {
            init_forces_done = TRUE;
        }
    }

#if defined(LOG_PERFORMANCE)
    hipEventRecord( time_event[1], control->streams[0] );
#endif

    if ( ret == SUCCESS )
    {
        ret = Hip_Compute_Bonded_Forces( system, control, data,
                workspace, lists, out_control );
    }

#if defined(LOG_PERFORMANCE)
    hipEventRecord( time_event[2], control->streams[0] );
    hipEventRecord( time_event[3], control->streams[4] );
#endif

    if ( ret == SUCCESS )
    {
        if ( charge_flag == TRUE )
        {
            Hip_Compute_Charges( system, control, data,
                    workspace, out_control, mpi_data );
        }

#if defined(LOG_PERFORMANCE)
        hipEventRecord( time_event[4], control->streams[4] );
#endif

        Hip_Compute_NonBonded_Forces( system, control, data, workspace,
                lists, out_control );

#if defined(LOG_PERFORMANCE)
        hipEventRecord( time_event[5], control->streams[4] );
        hipEventRecord( time_event[6], control->streams[0] );
#endif

        for ( i = 0; i < MAX_HIP_STREAMS; ++i )
        {
            hipStreamSynchronize( control->streams[i] );
        }

        Hip_Compute_Total_Force( system, control, data, workspace, lists, mpi_data );

#if defined(LOG_PERFORMANCE)
        hipEventRecord( time_event[7], control->streams[0] );
#endif

        init_forces_done = FALSE;
    }

#if defined(LOG_PERFORMANCE)
    if ( hipEventQuery( time_event[0] ) != hipSuccess )
    {
        hipEventSynchronize( time_event[0] );
    }

    if ( hipEventQuery( time_event[1] ) != hipSuccess )
    {
        hipEventSynchronize( time_event[1] );
    }

    hipEventElapsedTime( &time_elapsed, time_event[0], time_event[1] );
    data->timing.init_forces += (real) (time_elapsed / 1000.0);

    if ( hipEventQuery( time_event[2] ) != hipSuccess )
    {
        hipEventSynchronize( time_event[2] );
    }

    hipEventElapsedTime( &time_elapsed, time_event[1], time_event[2] );
    data->timing.bonded += (real) (time_elapsed / 1000.0);

    if ( ret == SUCCESS )
    {
        if ( hipEventQuery( time_event[3] ) != hipSuccess )
        {
            hipEventSynchronize( time_event[3] );
        }

        if ( hipEventQuery( time_event[4] ) != hipSuccess )
        {
            hipEventSynchronize( time_event[4] );
        }

        hipEventElapsedTime( &time_elapsed, time_event[3], time_event[4] );
        data->timing.cm += (real) (time_elapsed / 1000.0);

        if ( hipEventQuery( time_event[5] ) != hipSuccess )
        {
            hipEventSynchronize( time_event[5] );
        }

        hipEventElapsedTime( &time_elapsed, time_event[4], time_event[5] );
        data->timing.nonb += (real) (time_elapsed / 1000.0);

        if ( hipEventQuery( time_event[6] ) != hipSuccess )
        {
            hipEventSynchronize( time_event[6] );
        }

        if ( hipEventQuery( time_event[7] ) != hipSuccess )
        {
            hipEventSynchronize( time_event[7] );
        }

        hipEventElapsedTime( &time_elapsed, time_event[6], time_event[7] );
        data->timing.bonded += (real) (time_elapsed / 1000.0);
    }
#endif

    return ret;
}
