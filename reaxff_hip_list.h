/*----------------------------------------------------------------------
  PuReMD - Purdue ReaxFF Molecular Dynamics Program

  Copyright (2010) Purdue University
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
  Joseph Fogarty, jcfogart@mail.usf.edu
  Sagar Pandit, pandit@usf.edu
  Ananth Y Grama, ayg@cs.purdue.edu

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License as
  published by the Free Software Foundation; either version 2 of
  the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details:
  <http://www.gnu.org/licenses/>.
  ----------------------------------------------------------------------*/

#ifndef __CUDA_LIST_H_
#define __CUDA_LIST_H_

#include "reaxff_types.h"

#include "reaxff_list.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline HIP_HOST_DEVICE int Hip_Num_Entries( int i, reax_list *l )
{
    return l->end_index[i] - l->index[i];
}

static inline HIP_HOST_DEVICE int Hip_Start_Index( int i, reax_list *l )
{
    return l->index[i];
}

static inline HIP_HOST_DEVICE int Hip_End_Index( int i, reax_list *l )
{
    return l->end_index[i];
}

static inline HIP_HOST_DEVICE void Hip_Set_Start_Index( int i, int val, reax_list *l )
{
    l->index[i] = val;
}

static inline HIP_HOST_DEVICE void Hip_Set_End_Index( int i, int val, reax_list *l )
{
    l->end_index[i] = val;
}

void Hip_Adjust_End_Index_Before_ReAllocation(int oldN, int systemN, reax_list **gpu_lists);

void Hip_Copy_Far_Neighbors_List_Host_to_Device(reax_system *system, reax_list **gpu_lists, reax_list *cpu_lists);

void Hip_Make_List( int, int, int, reax_list * );

void Hip_Delete_List( reax_list * );

#ifdef __cplusplus
}
#endif

#endif
