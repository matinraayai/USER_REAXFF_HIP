/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Hasan Metin Aktulga, Purdue University
   (now at Lawrence Berkeley National Laboratory, hmaktulga@lbl.gov)
   Trinayan Baruah, Northeastern University(baruah.t@northeastern.edu)
   Nicholas Curtis, AMD(nicholas.curtis@amd.com)
   David Kaeli,     Northeastern University(kaeli@ece.neu.edu)
   Per-atom energy/virial added by Ray Shan (Sandia)
   Hybrid and hybrid/overlay compatibility added by Ray Shan (Sandia)
------------------------------------------------------------------------- */

#include "pair_reaxff.h"
#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include "atom.h"
#include "update.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "modify.h"
#include "fix.h"
#include "fix_reaxff.h"
#include "citeme.h"
#include "memory.h"
#include "error.h"
#include "utils.h"



#include "reaxff_defs.h"
#include "reaxff_allocate.h"
#include "reaxff_control.h"
#include "reaxff_ffield.h"
#include "reaxff_init_md.h"
#include "reaxff_io_tools.h"
#include "reaxff_list.h"
#include "reaxff_reset_tools.h"
#include "reaxff_forces.h"
#include "reaxff_vector.h"
#include "reaxff_hip_init_md.h"
#include "reaxff_hip_environment.h"
#include "reaxff_hip_list.h"
#include "reaxff_hip_reset_tools.h"
#include "reaxff_hip_forces.h"
#include "reaxff_hip_allocate.h"
#include "reaxff_hip_copy.h"
#include "reaxff_box.h"
#include "reaxff_puremd.h"
#include "reaxff_hip_neighbors.h"


using namespace LAMMPS_NS;

static const char cite_pair_reax_ff[] =
		"pair reax/ff command:\n\n"
		"@Article{Aktulga12,\n"
		" author = {H. M. Aktulga, J. C. Fogarty, S. A. Pandit, A. Y. Grama},\n"
		" title = {Parallel reactive molecular dynamics: Numerical methods and algorithmic techniques},\n"
		" journal = {Parallel Computing},\n"
		" year =    2012,\n"
		" volume =  38,\n"
		" pages =   {245--259}\n"
		"}\n\n";

/* ---------------------------------------------------------------------- */

PairReaxFFHIP::PairReaxFFHIP(LAMMPS *lmp) : Pair(lmp)
{
	if (lmp->citeme) lmp->citeme->add(cite_pair_reax_ff);

	single_enable = 0;
	restartinfo = 0;
	one_coeff = 1;
	manybody_flag = 1;
	ghostneigh = 1;

	fix_id = new char[24];
	snprintf(fix_id,24,"REAXFF_%d",instance_me);

	fix_reax = NULL;
	tmpid = NULL;
	tmpbo = NULL;

	nextra = 14;
	pvector = new double[nextra];

	setup_flag = 0;
	fixspecies_flag = 0;

	nmax = 0;
  handle = static_cast<puremd_handle*>(allocate_handle()); //TODO: Check if it is possible to use setup() instead

  //TODO: either remove the error ptr or add it somewhere reasonable
//  handle->system->pair_ptr = this;
//  handle->system->error_ptr = error;
//  handle->control->error_ptr = error;
//  handle->system->omp_active = 0;
}

/* ---------------------------------------------------------------------- */

PairReaxFFHIP::~PairReaxFFHIP()
{

	if (copymode) return;

	if (fix_reax) modify->delete_fix(fix_id);
	delete[] fix_id;

	if (setup_flag)
    cleanup(handle);

	// deallocate interface storage
	if (allocated) {
		memory->destroy(setflag);
		memory->destroy(cutsq);
		memory->destroy(cutghost);
		delete [] map;

		delete [] chi;
		delete [] eta;
		delete [] gamma;
	}

	memory->destroy(tmpid);
	memory->destroy(tmpbo);

	delete [] pvector;
  cleanup(handle);
}

/* ---------------------------------------------------------------------- */

void PairReaxFFHIP::allocate()
{
	int n = atom->ntypes;
	memory->create(setflag,n+1,n+1,"pair:setflag");
	memory->create(cutsq,n+1,n+1,"pair:cutsq");
	memory->create(cutghost,n+1,n+1,"pair:cutghost");
	map = new int[n+1];
	chi = new double[n+1];
	eta = new double[n+1];
	gamma = new double[n+1];
	allocated = 1;
}

/* ---------------------------------------------------------------------- */

void PairReaxFFHIP::settings(int narg, char **arg)
{
	if (narg < 1) error->all(FLERR,"Illegal pair_style command");

	// read name of control file or use default controls
  auto control = handle->control;
  auto out_control = handle->out_control;
  auto system = handle->system;
	if (strcmp(arg[0], "NULL") == 0) {
		strcpy( control->sim_name, "simulate" );
		control->ensemble = 0;
		out_control->energy_update_freq = 0;
		control->tabulate = 0;

		control->reneighbor = 1;
		control->vlist_cut = control->nonb_cut;
		control->bond_cut = 5.;
		control->hbond_cut = 7.50;
		control->thb_cut = 0.001;
//		control->thb_cutsq = 0.00001;
		control->bg_cut = 0.3;

		// Initialize for when omp style included
		//TODO: Fix?
//		control->nthreads = 1;

		out_control->write_steps = 0;
		out_control->traj_method = 0;
		strcpy( out_control->traj_title, "default_title" );
		out_control->atom_info = 0;
		out_control->bond_info = 0;
		out_control->angle_info = 0;
	} else Read_Control_File(arg[0], control, out_control);

	// default values

	qeqflag = 1;
	lgflag = 0;
	enobondsflag = 1;
	min_cap = MIN_CAP; //TODO: Add min_cap, safezone, and saferzone to parameters that can be extracted from PairReaxFF.
	// Any reference to these variables in FixReaxFFHip must return these
	safezone = SAFE_ZONE;
	saferzone = SAFER_ZONE;

	// process optional keywords

	int iarg = 1;

	while (iarg < narg) {
		if (strcmp(arg[iarg], "checkqeq") == 0) {
			if (iarg + 2 > narg) error->all(FLERR,"Illegal pair_style reax/c command");
			if (strcmp(arg[iarg + 1],"yes") == 0) qeqflag = 1;
			else if (strcmp(arg[iarg + 1],"no") == 0) qeqflag = 0;
			else error->all(FLERR,"Illegal pair_style reax/c command");
			iarg += 2;
		} else if (strcmp(arg[iarg],"enobonds") == 0) {
			if (iarg + 2 > narg) error->all(FLERR,"Illegal pair_style reax/c command");
			if (strcmp(arg[iarg + 1],"yes") == 0) enobondsflag = 1;
			else if (strcmp(arg[iarg + 1],"no") == 0) enobondsflag = 0;
			else error->all(FLERR,"Illegal pair_style reax/c command");
			iarg += 2;
		} else if (strcmp(arg[iarg],"lgvdw") == 0) {
			if (iarg + 2 > narg) error->all(FLERR,"Illegal pair_style reax/c command");
			if (strcmp(arg[iarg + 1],"yes") == 0) lgflag = 1;
			else if (strcmp(arg[iarg + 1],"no") == 0) lgflag = 0;
			else error->all(FLERR,"Illegal pair_style reax/c command");
			iarg += 2;
		} else if (strcmp(arg[iarg],"safezone") == 0) {
			if (iarg + 2 > narg) error->all(FLERR,"Illegal pair_style reax/c command");
			safezone = utils::numeric(FLERR,arg[iarg + 1],false, lmp);
			if (safezone < 0.0)
				error->all(FLERR,"Illegal pair_style reax/c safezone command");
			saferzone = safezone * 1.2 + 0.2;
			iarg += 2;
		} else if (strcmp(arg[iarg],"mincap") == 0) {
			if (iarg + 2 > narg) error->all(FLERR,"Illegal pair_style reax/c command");
			min_cap = utils::inumeric(FLERR,arg[iarg + 1],false, lmp);
			if (min_cap < 0)
				error->all(FLERR,"Illegal pair_style reax/c mincap command");
			iarg += 2;
		} else error->all(FLERR,"Illegal pair_style reax/c command");
	}

	// LAMMPS is responsible for generating nbrs

	control->reneighbor = 1;
}

/* ---------------------------------------------------------------------- */

void PairReaxFFHIP::coeff(int nargs, char **args)
{
	if (!allocated) allocate();

	if (nargs != 3 + atom->ntypes)
		error->all(FLERR,"Incorrect args for pair coefficients");

	// insure I,J args are * *

	if (strcmp(args[0],"*") != 0 || strcmp(args[1],"*") != 0)
		error->all(FLERR,"Incorrect args for pair coefficients");

	// read ffield file

	char *file = args[2];
	auto system = handle->system;
	auto control = handle->control;
  Read_Force_Field_File(file, &system->reax_param, system, control);

	// read args that map atom types to elements in potential file
	// map[i] = which element the Ith atom type is, -1 if NULL
	//TODO: Determine if this can be done by one of the restart file readers. If not, keep it
	//TODO: Determine if this can be replaced with system->my_atoms[i].type. They seem to do the same job
	int itmp = 0;
	int nreax_types = system->reax_param.num_atom_types;
	for (int i = 3; i < nargs; i++) {
		if (strcmp(args[i],"NULL") == 0) {
			map[i - 2] = -1;
			itmp ++;
			continue;
		}
	}

	int n = atom->ntypes;

	// pair_coeff element map
	for (int i = 3; i < nargs; i++)
		for (int j = 0; j < nreax_types; j++)
			if (strcasecmp(args[i],system->reax_param.sbp[j].name) == 0) {
				map[i - 2] = j;
				itmp ++;
			}

	// error check
	if (itmp != n)
		error->all(FLERR,"Non-existent ReaxFF type");

	for (int i = 1; i <= n; i++)
		for (int j = i; j <= n; j++)
			setflag[i][j] = 0;

	// set setflag i,j for type pairs where both are mapped to elements

	int count = 0;
	for (int i = 1; i <= n; i++)
		for (int j = i; j <= n; j++)
			if (map[i] >= 0 && map[j] >= 0) {
				setflag[i][j] = 1;
				count++;
			}

	if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");

}

/* ---------------------------------------------------------------------- */

void PairReaxFFHIP::init_style( )
{
	if (!atom->q_flag)
		error->all(FLERR,"Pair style reax/c requires atom attribute q");

	// firstwarn = 1;
	auto system = handle->system;
  auto control = handle->control;
	bool have_qeq = ((modify->find_fix_by_style("^qeq/reax") != -1)
			|| (modify->find_fix_by_style("^qeq/shielded") != -1));
	if (!have_qeq && qeqflag == 1)
		error->all(FLERR,"Pair reax/c requires use of fix qeq/reax or qeq/shielded");

	system->n = atom->nlocal; // my atoms
	system->N = atom->nlocal + atom->nghost; // mine + ghosts
	system->bigN = static_cast<int> (atom->natoms);  // all atoms in the system
	//TODO: how to get/ where to place wsize? Remove if needed
//	system->wsize = comm->nprocs;

	system->big_box.V = 0;
	system->big_box.box_norms[0] = 0;
	system->big_box.box_norms[1] = 0;
	system->big_box.box_norms[2] = 0;

	if (atom->tag_enable == 0)
		error->all(FLERR,"Pair style reax/c requires atom IDs");
	if (force->newton_pair == 0)
		error->all(FLERR,"Pair style reax/c requires newton pair on");
	if ((atom->map_tag_max > 99999999) && (comm->me == 0))
		error->warning(FLERR,"Some Atom-IDs are too large. Pair style reax/c "
				"native output files may get misformatted or corrupted");

	// because system->bigN is an int, we cannot have more atoms than MAXSMALLINT

	if (atom->natoms > MAXSMALLINT)
		error->all(FLERR,"Too many atoms for pair style reax/c");

	// need a half neighbor list w/ Newton off and ghost neighbors
	// built whenever re-neighboring occurs

	int irequest = neighbor->request(this, instance_me);
	//neighbor->requests[irequest]->newton = 2;
	neighbor->requests[irequest]->ghost = 1;
	neighbor->requests[irequest]->half = 0;
	neighbor->requests[irequest]->full = 1;

	cutmax = MAX3(control->nonb_cut, control->hbond_cut, control->bond_cut);
	if ((cutmax < 2.0*control->bond_cut) && (comm->me == 0))
		error->warning(FLERR,"Total cutoff < 2*bond cutoff. May need to use an "
				"increased neighbor list skin.");

	/*  for( int i = 0; i < LIST_N; ++i )
    if (lists[i].allocated != 1)
      lists[i].allocated = 0;
	 */
	if (fix_reax == NULL) {
		char **fixarg = new char*[3];
		fixarg[0] = (char *) fix_id;
		fixarg[1] = (char *) "all";
		fixarg[2] = (char *) "REAXC";
		modify->add_fix(3, fixarg);
		delete [] fixarg;
		fix_reax = (FixReaxC *) modify->fix[modify->nfix-1];
	}
}

/* ---------------------------------------------------------------------- */

void PairReaxFFHIP::setup()
{
  auto system = handle->system;
  auto control = handle->control;
  auto data = handle->data;
  auto workspace = handle->workspace;
  auto lists = handle->lists;
  auto out_control = handle->out_control;
  auto mpi_data = handle->mpi_data;
	int oldN;
	int mincap = MIN_CAP;
	double safezone = SAFER_ZONE;

	system->n = atom->nlocal; // my atoms
	system->N = atom->nlocal + atom->nghost; // mine + ghosts
	oldN = system->N;
	system->bigN = static_cast<int> (atom->natoms);  // all atoms in the system

	if (setup_flag == 0) {
#if defined(HAVE_HIP)
	  Hip_Initialize( system, control, data, workspace, lists, out_control, mpi_data );

#if defined(CUDA_DEVICE_PACK)
	  //TODO: remove once Comm_Atoms ported
	  Hip_Copy_MPI_Data_Host_to_Device( mpi_data );
#endif

	  Hip_Init_Block_Sizes( system, control );

	  Hip_Copy_Atoms_Host_to_Device( system, control );
	  Hip_Copy_Grid_Host_to_Device( control, &system->my_grid, &system->d_my_grid);

	  Hip_Reset( system, control, data, workspace, lists );

	  Hip_Generate_Neighbor_Lists( system, control, data, workspace, lists );
#else
	  Initialize( system, control, data, workspace, lists, out_control, mpi_data );

	  /* compute f_0 */
	  Comm_Atoms( system, control, data, workspace, mpi_data, TRUE );

	  Reset( system, control, data, workspace, lists );

	  ret = Generate_Neighbor_Lists( system, data, workspace, lists );

	  if ( ret != SUCCESS )
	  {
	    fprintf( stderr, "[ERROR] cannot generate initial neighbor lists. Terminating...\n" );
	    MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
	  }
#endif

		int *num_bonds = fix_reax->num_bonds;
		int *num_hbonds = fix_reax->num_hbonds;

		control->vlist_cut = neighbor->cutneighmax;

		// determine the local and total capacity

		system->local_cap = MAX( (int)(system->n * safezone), mincap );
		system->total_cap = MAX( (int)(system->N * safezone), mincap );

		// initialize my data structures
		PreAllocate_Space(system, control, workspace);
    Setup_Environment(system, control, mpi_data);

		for( int k = 0; k < system->N; ++k )
		{
			num_bonds[k] = system->my_atoms[k].num_bonds;
			num_hbonds[k] = system->my_atoms[k].num_hbonds;
		}
		update_and_copy_reax_atoms_to_device();
		setup_flag = 1;
	}
	else {
		//printf("Realloc setup \n");
		// fill in reax datastructures
		update_and_copy_reax_atoms_to_device();

		// reset the bond list info for new atoms
		//printf("Initial setup done. far numbers gpu %d \n", gpu_lists[FAR_NBRS]->num_intrs);

  //TODO: This function has been removed in previous commits. I'm not sure what it does or if it's necessary.
  // Look for this in previous commits
//		Hip_Adjust_End_Index_Before_ReAllocation(oldN, system->N, lists);

		//printf("Initial setup done. far numbers gpu %d \n", gpu_lists[FAR_NBRS]->num_intrs);

		// check if I need to shrink/extend my data-structs

		Hip_Reallocate_Part1(system, control, data, workspace, lists, mpi_data);
		Hip_Reallocate_Part2(system, control, data, workspace, lists, mpi_data);
	}

	bigint local_ngroup = list->inum;
	MPI_Allreduce( &local_ngroup, &ngroup, 1, MPI_LMP_BIGINT, MPI_SUM, world );
}

/* ---------------------------------------------------------------------- */

double PairReaxFFHIP::init_one(int i, int j)
{
	if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

	cutghost[i][j] = cutghost[j][i] = cutmax;
	return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairReaxFFHIP::compute(int eflag, int vflag)
{
  auto control = handle->control;
  auto system = handle->system;
  auto data = handle->data;
  auto workspace = handle->workspace;
  auto lists = handle->lists;
  auto mpi_data = handle->mpi_data;
  auto out_control = handle->out_control;
	double evdwl, ecoul;
	double t_start, t_end;

	// communicate num_bonds once every reneighboring
	// 2 num arrays stored by fix, grab ptr to them

	if (neighbor->ago == 0) comm->forward_comm_fix(fix_reax);
	int *num_bonds = fix_reax->num_bonds;
	int *num_hbonds = fix_reax->num_hbonds;

	evdwl = ecoul = 0.0;
	ev_init(eflag, vflag);

	if (vflag_global) control->virial = 1;
	else control->virial = 0;

	system->n = atom->nlocal; // my atoms
	system->N = atom->nlocal + atom->nghost; // mine + ghosts
	system->bigN = static_cast<int> (atom->natoms);  // all atoms in the system

	system->big_box.V = 0;
	system->big_box.box_norms[0] = 0;
	system->big_box.box_norms[1] = 0;
	system->big_box.box_norms[2] = 0;
	if (comm->me == 0 ) t_start = MPI_Wtime();

	// setup data structures
	setup();

	Hip_Reset(system, control, data, workspace, lists);
//	workspace->realloc.far_nbrs = update_and_write_reax_lists_to_device();

	// timing for filling in the reax lists
	if (comm->me == 0) {
		t_end = MPI_Wtime();
		data->timing.nbrs = t_end - t_start;
	}


	// forces
	Hip_Compute_Forces(system, control, data, workspace, lists, out_control, mpi_data);


	read_reax_forces_from_device(vflag);
	Hip_Copy_Atoms_Device_to_Host(system, control);


	for(int k = 0; k < system->N; ++k)
	{
		num_bonds[k] = system->my_atoms[k].num_bonds;
		num_hbonds[k] = system->my_atoms[k].num_hbonds;
	}


	Hip_Copy_Simulation_Data_Device_to_Host( control, data, (simulation_data *)data->d_simulation_data);



	// energies and pressure

	if (eflag_global) {

		evdwl += data->my_en.e_bond;
		evdwl += data->my_en.e_ov;
		evdwl += data->my_en.e_un;
		evdwl += data->my_en.e_lp;
		evdwl += data->my_en.e_ang;
		evdwl += data->my_en.e_pen;
		evdwl += data->my_en.e_coa;
		evdwl += data->my_en.e_hb;
		evdwl += data->my_en.e_tor;
		evdwl += data->my_en.e_con;
		evdwl += data->my_en.e_vdW;

		ecoul += data->my_en.e_ele;
		ecoul += data->my_en.e_pol;

		// eng_vdwl += evdwl;
		// eng_coul += ecoul;

		// Store the different parts of the energy
		// in a list for output by compute pair command

		pvector[0] = data->my_en.e_bond;
		pvector[1] = data->my_en.e_ov + data->my_en.e_un;
		pvector[2] = data->my_en.e_lp;
		pvector[3] = 0.0;
		pvector[4] = data->my_en.e_ang;
		pvector[5] = data->my_en.e_pen;
		pvector[6] = data->my_en.e_coa;
		pvector[7] = data->my_en.e_hb;
		pvector[8] = data->my_en.e_tor;
		pvector[9] = data->my_en.e_con;
		pvector[10] = data->my_en.e_vdW;
		pvector[11] = data->my_en.e_ele;
		pvector[12] = 0.0;
		pvector[13] = data->my_en.e_pol;
	}

	if (vflag_fdotr)
	{
		virial_fdotr_compute();
	}



	// Set internal timestep counter to that of LAMMPS

	data->step = update->ntimestep;



	Output_Results( system, control, data, lists, out_control, mpi_data );

	// populate tmpid and tmpbo arrays for fix reax/c/species
	int i, j;

	if(fixspecies_flag) {
		printf("fix species not implemented for GPU  %d\n",fixspecies_flag);

	}

	//printf("Finished loop \n");
}

/* ---------------------------------------------------------------------- */

void PairReaxFFHIP::update_and_copy_reax_atoms_to_device()
{
  auto system = handle->system;
  auto control = handle->control;
	int *num_bonds = fix_reax->num_bonds;
	int *num_hbonds = fix_reax->num_hbonds;

	if (system->N > system->total_cap)
		error->all(FLERR,"Too many ghost atoms");

	for( int i = 0; i < system->N; ++i ){
		system->my_atoms[i].orig_id = atom->tag[i];
		system->my_atoms[i].type = map[atom->type[i]];
		system->my_atoms[i].x[0] = atom->x[i][0];
		system->my_atoms[i].x[1] = atom->x[i][1];
		system->my_atoms[i].x[2] = atom->x[i][2];
		system->my_atoms[i].q = atom->q[i];
		system->my_atoms[i].num_bonds = num_bonds[i];
		system->my_atoms[i].num_hbonds = num_hbonds[i];
	}

	Hip_Copy_Atoms_Device_to_Host(system, control);
}

/* ---------------------------------------------------------------------- */

void PairReaxFFHIP::get_distance(rvec xj, rvec xi, double *d_sqr, rvec *dvec )
{
	(*dvec)[0] = xj[0] - xi[0];
	(*dvec)[1] = xj[1] - xi[1];
	(*dvec)[2] = xj[2] - xi[2];
	*d_sqr = SQR((*dvec)[0]) + SQR((*dvec)[1]) + SQR((*dvec)[2]);
}

/* ---------------------------------------------------------------------- */

void PairReaxFFHIP::set_far_nbr(far_neighbor_data *fdest, int dest_idx,
                                int j, double d, rvec dvec )
		{
    fdest->nbr[dest_idx] = j;
    fdest->d[dest_idx] = d;
    //	*(fdest->nbr) = j;
    //	*(fdest->d) = d;
    rvec_Copy(fdest->dvec[dest_idx], dvec);
    ivec_MakeZero(fdest->rel_box[dest_idx]);
    //	rvec_Copy( *(fdest->dvec), dvec );
    //	ivec_MakeZero( *(fdest->rel_box) );
		}

/* ---------------------------------------------------------------------- */

int PairReaxFFHIP::estimate_reax_lists()
{
  auto system = handle->system;
  auto control = handle->control;
	int itr_i, itr_j, i, j;
	int num_nbrs, num_marked;
	int *ilist, *jlist, *numneigh, **firstneigh, *marked;
	double d_sqr;
	rvec dvec;
	double **x;

	x = atom->x;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	num_nbrs = 0;
	num_marked = 0;
	marked = (int*) calloc( system->N, sizeof(int) );

	int numall = list->inum + list->gnum;

	for( itr_i = 0; itr_i < numall; ++itr_i ){
		i = ilist[itr_i];
		marked[i] = 1;
		++num_marked;
		jlist = firstneigh[i];

		for( itr_j = 0; itr_j < numneigh[i]; ++itr_j ){
			j = jlist[itr_j];
			if( i < j)
			{
				j &= NEIGHMASK;
				get_distance( x[j], x[i], &d_sqr, &dvec );

				if (d_sqr <= SQR(control->nonb_cut))
					++num_nbrs;
			}
		}

		for( itr_j = 0; itr_j < numneigh[i]; ++itr_j ){
			j = jlist[itr_j];
			if( i > j)
			{
				j &= NEIGHMASK;
				get_distance( x[i], x[j], &d_sqr, &dvec );

				if (d_sqr <= SQR(control->nonb_cut))
				{
					++num_nbrs;
				}
			}
		}

	}


	free(marked);


	return static_cast<int> (MAX( num_nbrs*safezone, min_cap*MIN_NBRS ));
}

/* ---------------------------------------------------------------------- */
//TODO: Not called anywhere, can be removed unless there is a use to it
//int PairReaxFFHIP::update_and_write_reax_lists_to_device()
//{
//
//
//	int itr_i, itr_j, i, j;
//	int num_nbrs;
//	int *ilist, *jlist, *numneigh, **firstneigh;
//	double d_sqr, cutoff_sqr;
//	rvec dvec;
//	double *dist, **x;
//	reax_list *far_nbrs;
//	far_neighbor_data *far_list;
//
//	x = atom->x;
//	ilist = list->ilist;
//	numneigh = list->numneigh;
//	firstneigh = list->firstneigh;
//
//	far_nbrs = (cpu_lists +FAR_NBRS);
//	far_list = &(far_nbrs->far_nbr_list);
//
//
//
//	num_nbrs = 0;
//	int inum = list->inum;
//	dist = (double*) calloc( system->N, sizeof(double) );
//
//
//
//	int numall = list->inum + list->gnum;
//
//	for( itr_i = 0; itr_i < numall; ++itr_i ){
//		i = ilist[itr_i];
//		jlist = firstneigh[i];
//
//		Set_Start_Index( i, num_nbrs, far_nbrs );
//
//		if (i < inum)
//			cutoff_sqr = control->nonb_cut*control->nonb_cut;
//		else
//			cutoff_sqr = control->bond_cut*control->bond_cut;
//
//		for( itr_j = 0; itr_j < numneigh[i]; ++itr_j ) {
//			j = jlist[itr_j];
//			if ( i <  j)
//			{
//				j &= NEIGHMASK;
//
//				get_distance( x[j], x[i], &d_sqr, &dvec );
//
//				if (d_sqr <= (cutoff_sqr)) {
//					dist[j] = sqrt( d_sqr );
//					set_far_nbr( far_list, num_nbrs, j, dist[j], dvec );
//					++num_nbrs;
//				}
//			}
//		}
//
//		for( itr_j = 0; itr_j < numneigh[i]; ++itr_j ) {
//			j = jlist[itr_j];
//			if ( i >  j)
//			{
//				j &= NEIGHMASK;
//
//				get_distance( x[i], x[j], &d_sqr, &dvec );
//
//				if (d_sqr <= (cutoff_sqr)) {
//					dist[j] = sqrt( d_sqr );
//					set_far_nbr( far_list, num_nbrs, j, dist[j], dvec );
//					++num_nbrs;
//				}
//			}
//		}
//
//
//
//
//
//		Set_End_Index( i, num_nbrs, far_nbrs );
//	}
//
//	free( dist );
//
//	Hip_Copy_List_Device_to_Host(cpu_lists, gpu_lists[FAR_NBRS], TYP_FAR_NEIGHBOR);
//
//
//	return num_nbrs;
//}

/* ---------------------------------------------------------------------- */

void PairReaxFFHIP::read_reax_forces_from_device(int /*vflag*/)
{

  auto workspace = handle->workspace;
  auto system = handle->system;
  //TODO: this has been removed in previous commits. Not sure what it was supposed to do
  // Look for it in previous commits
//	Output_Sync_Forces(handle->workspace, handle->system->total_cap);


	int world_rank;
	MPI_Comm_rank(world, &world_rank);


	for( int i = 0; i < system->N; ++i ) {
		system->my_atoms[i].f[0] = workspace->f[i][0];
		system->my_atoms[i].f[1] = workspace->f[i][1];
		system->my_atoms[i].f[2] = workspace->f[i][2];



		//if(i < 20)
			//printf("%d,%f,%f,%f\n",system->my_atoms[i].orig_id, system->my_atoms[i].f[0],system->my_atoms[i].f[1],system->my_atoms[i].f[2]);


		atom->f[i][0] += -workspace->f[i][0];
		atom->f[i][1] += -workspace->f[i][1];
		atom->f[i][2] += -workspace->f[i][2];
	}

	//exit(0);

	//printf("Computation done\n");


}

/* ---------------------------------------------------------------------- */

void *PairReaxFFHIP::extract(const char *str, int &dim)
{
  //TODO: is this necessary? If so, do we need a more extensive way of extracting data?
  auto system = handle->system;
	dim = 1;
	if (strcmp(str, "chi") == 0 && chi) {
		for (int i = 1; i <= atom->ntypes; i++)
			if (map[i] >= 0) chi[i] = system->reax_param.d_sbp[map[i]].chi;
			else chi[i] = 0.0;
		return (void *) chi;
	}
	if (strcmp(str, "eta") == 0 && eta) {
		for (int i = 1; i <= atom->ntypes; i++)
			if (map[i] >= 0) eta[i] = system->reax_param.d_sbp[map[i]].eta;
			else eta[i] = 0.0;
		return (void *) eta;
	}
	if (strcmp(str, "gamma") == 0 && gamma) {
		for (int i = 1; i <= atom->ntypes; i++)
			if (map[i] >= 0) gamma[i] = system->reax_param.d_sbp[map[i]].gamma;
			else gamma[i] = 0.0;
		return (void *) gamma;
	}
	return NULL;
}

/* ---------------------------------------------------------------------- */

double PairReaxFFHIP::memory_usage()
{
  auto system = handle->system;
	double bytes = 0.0;

	// From pair_reax_c
	bytes += 1.0 * system->N * sizeof(int);
	bytes += 1.0 * system->N * sizeof(double);

	// From reaxc_allocate: BO
	bytes += 1.0 * system->total_cap * sizeof(reax_atom);
	bytes += 19.0 * system->total_cap * sizeof(double);
	bytes += 3.0 * system->total_cap * sizeof(int);

	// From reaxc_lists
	/*bytes += 2.0 * lists->n * sizeof(int);
  bytes += lists->num_intrs * sizeof(three_body_interaction_data);
  bytes += lists->num_intrs * sizeof(bond_data);
  bytes += lists->num_intrs * sizeof(dbond_data);
  bytes += lists->num_intrs * sizeof(dDelta_data);
  bytes += lists->num_intrs * sizeof(far_neighbor_data);
  bytes += lists->num_intrs * sizeof(hbond_data);

  if(fixspecies_flag)
    bytes += 2 * nmax * MAXSPECBOND * sizeof(double);*/

	return bytes;
}

/* ---------------------------------------------------------------------- */

void PairReaxFFHIP::FindBond()
{//TODO: Remove
}
