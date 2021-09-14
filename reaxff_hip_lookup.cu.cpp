
#include "reaxff_hip_lookup.h"

#include "reaxff_hip_utils.h"

#include "reaxff_hip_index_utils.h"


extern "C" void Hip_Copy_LR_Lookup_Table_Host_to_Device( reax_system *system,
        control_params *control, storage *workspace, int *aggregated )
{
    int i, j;
    int num_atom_types;
    LR_data *d_y;
    cubic_spline_coef *temp;

    num_atom_types = system->reax_param.num_atom_types;

    fprintf( stderr, "Copying the LR Lookyp Table to the device ... \n" );

    hip_malloc( (void **) &workspace->d_LR,
            sizeof(LR_lookup_table) * num_atom_types * num_atom_types,
            FALSE, "LR_lookup:table" );

    /*
       for( i = 0; i < MAX_ATOM_TYPES; ++i )
       existing_types[i] = 0;

       for( i = 0; i < system->N; ++i )
       existing_types[ system->atoms[i].type ] = 1;
     */

    sHipMemcpy( workspace->d_LR, workspace->LR,
            sizeof(LR_lookup_table) * (num_atom_types * num_atom_types), 
            hipMemcpyHostToDevice, __FILE__, __LINE__ );

    for( i = 0; i < num_atom_types; ++i )
    {
        if ( aggregated[i] )
        {
            for( j = i; j < num_atom_types; ++j )
            {
                if ( aggregated[j] )
                {
                    hip_malloc( (void **) &d_y, sizeof(LR_data) * (control->tabulate + 1),
                            FALSE, "LR_lookup:d_y" );
                    sHipMemcpy( d_y, workspace->LR[ index_lr(i, j, num_atom_types) ].y,
                            sizeof(LR_data) * (control->tabulate + 1),
                            hipMemcpyHostToDevice, __FILE__, __LINE__ );
                    sHipMemcpy( &workspace->d_LR[ index_lr(i, j, num_atom_types) ].y, &d_y,
                            sizeof(LR_data *),
                            hipMemcpyHostToDevice, __FILE__, __LINE__ );

                    hip_malloc( (void **) &temp, sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            FALSE, "LR_lookup:h" );
                    sHipMemcpy( temp, workspace->LR[ index_lr(i, j, num_atom_types) ].H,
                            sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            hipMemcpyHostToDevice, __FILE__, __LINE__ );
                    sHipMemcpy( &workspace->d_LR[ index_lr(i, j, num_atom_types) ].H, &temp,
                            sizeof(cubic_spline_coef *),
                            hipMemcpyHostToDevice, __FILE__, __LINE__ );

                    hip_malloc( (void **) &temp, sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            FALSE, "LR_lookup:vdW" );
                    sHipMemcpy( temp, workspace->LR[ index_lr(i, j, num_atom_types) ].vdW,
                            sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            hipMemcpyHostToDevice, __FILE__, __LINE__ );
                    sHipMemcpy( &workspace->d_LR[ index_lr(i, j, num_atom_types) ].vdW, &temp,
                            sizeof(cubic_spline_coef *),
                            hipMemcpyHostToDevice, __FILE__, __LINE__ );

                    hip_malloc( (void **) &temp, sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            FALSE, "LR_lookup:CEvd" );
                    sHipMemcpy( temp, workspace->LR[ index_lr(i, j, num_atom_types) ].CEvd,
                            sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            hipMemcpyHostToDevice, __FILE__, __LINE__ );
                    sHipMemcpy( &workspace->d_LR[ index_lr(i, j, num_atom_types) ].CEvd, &temp,
                            sizeof(cubic_spline_coef *),
                            hipMemcpyHostToDevice, __FILE__, __LINE__ );

                    hip_malloc( (void **) &temp, sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            FALSE, "LR_lookup:ele" );
                    sHipMemcpy( temp,workspace->LR[ index_lr(i, j, num_atom_types) ].ele,
                            sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            hipMemcpyHostToDevice, __FILE__, __LINE__ );
                    sHipMemcpy( &workspace->d_LR[ index_lr(i, j, num_atom_types) ].ele, &temp,
                            sizeof(cubic_spline_coef *),
                            hipMemcpyHostToDevice, __FILE__, __LINE__ );

                    hip_malloc( (void **) &temp, sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            FALSE, "LR_lookup:ceclmb" );
                    sHipMemcpy( temp, workspace->LR[ index_lr(i, j, num_atom_types) ].CEclmb,
                            sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            hipMemcpyHostToDevice, __FILE__, __LINE__ );
                    sHipMemcpy( &workspace->d_LR[ index_lr(i, j, num_atom_types) ].CEclmb, &temp,
                            sizeof(cubic_spline_coef *),
                            hipMemcpyHostToDevice, __FILE__, __LINE__ );
                }
            }
        }
    }
}
