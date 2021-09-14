
#ifndef __CUDA_NEIGHBORS_H__
#define __CUDA_NEIGHBORS_H__

#include "reaxff_types.h"


#ifdef __cplusplus
extern "C" {
#endif

int Hip_Generate_Neighbor_Lists( reax_system *, simulation_data *,
        storage *, reax_list ** );

#ifdef __cplusplus
}
#endif

void Hip_Estimate_Num_Neighbors( reax_system *, simulation_data * );


#endif