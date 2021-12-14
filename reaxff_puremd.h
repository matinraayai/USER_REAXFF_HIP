/*----------------------------------------------------------------------
  SerialReax - Reax Force Field Simulator

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

#ifndef __PUREMD_H_
#define __PUREMD_H_

#if defined(LAMMPS_REAX)
    #include "reaxff_types.h"
#else
    #include "reax_types.h"
#endif


#define PUREMD_SUCCESS (0)
#define PUREMD_FAILURE (-1)


#ifdef __cplusplus
extern "C"  {
#endif

void* allocate_handle();

void* setup( const char * const, const char * const,
        const char * const );



int setup_callback( const void * const, const callback_function );

int simulate( const void * const );

int cleanup( const void * const );

reax_atom* get_atoms( const void * const );

int set_output_enabled( const void * const, const int );

#ifdef __cplusplus
}
#endif


#endif
