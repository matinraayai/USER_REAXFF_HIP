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
   Contributing author: Trinayan Baruah, Northeastern University(baruah.t@northeastern.edu)
                        Nicholas Curtis, AMD(nicholas.curtis@amd.com)
			                  David Kaeli,     Northeastern University(kaeli@ece.neu.edu)
   Please cite the related publication:
   H. M. Aktulga, J. C. Fogarty, S. A. Pandit, A. Y. Grama,
   "Parallel Reactive Molecular Dynamics: Numerical Methods and
   Algorithmic Techniques", Parallel Computing, in press.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

    PairStyle(reax/f/hip,PairReaxFFHIP)

#endif

#ifndef LMP_PAIR_REAXC__GPU_H
#define LMP_PAIR_REAXC__GPU_H

#include "pair.h"
#include "reaxff_types.h"

namespace LAMMPS_NS {

class PairReaxFFHIP : public Pair {
  public:
    explicit PairReaxFFHIP(class LAMMPS *);

    ~PairReaxFFHIP() override;

    void compute(int, int) override;

    void settings(int, char **) override;

    void coeff(int, char **) override;

    virtual void init_style() override;

    double init_one(int, int) override;

    void *extract(const char *, int &) override;

    int fixbond_flag, fixspecies_flag;

    int **tmpid;

    double **tmpbo, **tmpr;

    control_params *control;

    reax_system *system;

    output_controls *out_control;

    simulation_data *data;

    storage *workspace;

    reax_list **gpu_lists;

    reax_list *cpu_lists;

    mpi_datatypes *mpi_data;

    bigint ngroup;

    protected:
      char *fix_id;
      double cutmax;
      int nelements;                // # of unique elements
      char **elements;              // names of unique elements
      int *map;
      class FixReaxC *fix_reax;

      double *chi,*eta,*gamma;
      int qeqflag;
      int setup_flag;
      int firstwarn;

      void allocate();
      void setup();
      void create_compute();
      void create_fix();
      void update_and_copy_reax_atoms_to_device();
      int update_and_write_reax_lists_to_device();
      void get_distance(rvec, rvec, double *, rvec *);
      void set_far_nbr(far_neighbor_data *, int, int, double, rvec);
      int estimate_reax_lists();
      void read_reax_forces_from_device(int);

      int nmax;
      void FindBond();
      double memory_usage();

};

}

#endif

/* ERROR/WARNING messages:

E: Too many ghost atoms

Number of ghost atoms has increased too much during simulation and has exceeded
the size of reax/c arrays.  Increase safe_zone and min_cap in pair_style reax/c
command

*/
