/* -*- c++ -*- ----------------------------------------------------------
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
   Contributing author: 
   Trinayan Baruah, Northeastern University(baruah.t@northeastern.edu)
   Matin Raayai Ardakani, Northeastern University(raayaiardakani.m@northeastern.edu)
   Nicholas Curtis, AMD(nicholas.curtis@amd.com)
   David Kaeli,     Northeastern University(kaeli@ece.neu.edu)
   
   Hasan Metin Aktulga, Purdue University
   (now at Lawrence Berkeley National Laboratory, hmaktulga@lbl.gov)

   Please cite the related publication:
   H. M. Aktulga, J. C. Fogarty, S. A. Pandit, A. Y. Grama,
   "Parallel Reactive Molecular Dynamics: Numerical Methods and
   Algorithmic Techniques", Parallel Computing, in press.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(qeq/reax,FixQEqReax)

#else

#ifndef LMP_FIX_QEQ_REAX_H
#define LMP_FIX_QEQ_REAX_H

#include "fix.h"
#include "reaxff_types.h"

namespace LAMMPS_NS {

	class FixQEqReax : public Fix {
	public:
		FixQEqReax(class LAMMPS *, int, char **);
		~FixQEqReax();
		int setmask();
		virtual void post_constructor();
		virtual void init();
		void init_list(int, class NeighList *);
		virtual void init_storage();
		void setup_pre_force(int);
		virtual void pre_force(int);
		virtual int updateReaxLists(class PairReaxFFHIP *reaxc);
		virtual void get_distance( rvec xj, rvec xi, double *d_sqr, rvec *dvec );
		virtual void set_far_nbr( far_neighbor_data *fdest, int dest_idx, int j, double d, rvec dvec);



		void setup_pre_force_respa(int, int);
		void pre_force_respa(int, int, int);

		void min_setup_pre_force(int);
		void min_pre_force(int);

//		fix_qeq_gpu *qeq_gpu;

	protected:
		int nevery, reaxflag;
		int n, N, m_fill;
		int n_cap, nmax, m_cap;
		int pack_flag;
		int nlevels_respa;
		class NeighList *list;
		class PairReaxFFHIP *reaxc;

		double swa, swb;      // lower/upper Taper cutoff radius
		double tolerance;     // tolerance for the norm of the rel residual in CG


		bigint ngroup;

		// fictitious charges

		double **s_hist, **t_hist;
		int nprev;


		char *pertype_option;  // argument to determine how per-type info is obtained
		virtual void pertype_parameters(char*);
		void init_shielding();
		void init_taper();
		virtual void allocate_storage();
		virtual void deallocate_storage();
		void reallocate_storage();
		virtual void allocate_matrix();
		void deallocate_matrix();
		void reallocate_matrix();

		virtual void init_matvec();
		void init_H();
		double calculate_H(double,double);
		virtual void hip_calculate_Q();
		virtual int Hip_CG( double *, double *);

		//int GMRES(double*,double*);
		virtual void hip_sparse_matvec(double *,double*);
    virtual void compute_H();

		virtual int pack_forward_comm(int, int *, double *, int, int *);
		virtual void unpack_forward_comm(int, int, double *);
		virtual int pack_reverse_comm(int, int, double *);
		virtual void unpack_reverse_comm(int, int *, double *);
		virtual double memory_usage();
		virtual void grow_arrays(int);
		virtual void copy_arrays(int, int, int);
		virtual int pack_exchange(int, double *);
		virtual int unpack_exchange(int, double *);

		virtual double parallel_norm( double*, int );
		virtual double parallel_dot( double*, double*, int );
		virtual double parallel_vector_acc( double*, int );

		virtual void vector_sum(double*,double,double*,double,double*,int);
		virtual void vector_add(double*, double, double*,int);
		void intializeAtomsAndCopyToDevice();

		// dual CG support
		int dual_enabled;  // 0: Original, separate s & t optimization; 1: dual optimization
		int matvecs_s, matvecs_t; // Iteration count for each system
	};

}

#endif
#endif
