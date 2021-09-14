#include "reaxff_hip_allocate.h"
#include "reaxff_hip_forces.h"
#include "reaxff_hip_dense_lin_alg.h"
//#include "reaxff_hip_forces.cu.cpp"
#include "reaxff_hip_list.h"
#include "reaxff_hip_neighbors.h"
#include "reaxff_hip_utils.h"
#include "reaxff_hip_fix_qeq.h"
#include "reaxff_hip_reduction.h"
#include "hip/hip_runtime.h"
#include "reaxff_hip_index_utils.h"
#include "reaxff_tool_box.h"
#include "reaxff_vector.h"
#include "reaxff_hip_fix_qeq.h"
#include "reaxff_hip_forces.h"

extern "C"
{

void Hip_Allocate_Hist_ST(fix_qeq_gpu *qeq_gpu,int nmax)
{
	hip_malloc( (void **) &qeq_gpu->s_hist, sizeof(rvec4) * nmax, TRUE, "b" );
	hip_malloc( (void **) &qeq_gpu->t_hist, sizeof(rvec4) * nmax, TRUE, "x" );
}


void  HipAllocateStorageForFixQeq(int nmax, int dual_enabled, fix_qeq_gpu *qeq_gpu)
{
	hip_malloc( (void **) &qeq_gpu->s, sizeof(double) * nmax, TRUE,
			"Hip_Allocate_Matrix::start" );
	hip_malloc( (void **) &qeq_gpu->t, sizeof(double) * nmax, TRUE,
			"Hip_Allocate_Matrix::start" );
	hip_malloc( (void **) &qeq_gpu->Hdia_inv, sizeof(double) * nmax, TRUE,
			"Hip_Allocate_Matrix::start" );
	hip_malloc( (void **) &qeq_gpu->b_s, sizeof(double) * nmax, TRUE,
			"Hip_Allocate_Matrix::start" );
	hip_malloc( (void **) &qeq_gpu->b_t, sizeof(double) * nmax, TRUE,
			"Hip_Allocate_Matrix::start" );
	hip_malloc( (void **) &qeq_gpu->b_prc, sizeof(double) * nmax, TRUE,
			"Hip_Allocate_Matrix::start" );
	hip_malloc( (void **) &qeq_gpu->b_prm, sizeof(double) * nmax, TRUE,
			"Hip_Allocate_Matrix::start" );

	int size = nmax;
	if (dual_enabled)
	{
		size*= 2;
	}

	hip_malloc( (void **) &qeq_gpu->p, sizeof(double) * size, TRUE,
			"Hip_Allocate_Matrix::start" );
	hip_malloc( (void **) &qeq_gpu->q, sizeof(double) * size, TRUE,
			"Hip_Allocate_Matrix::start" );
	hip_malloc( (void **) &qeq_gpu->r, sizeof(double) * size, TRUE,
			"Hip_Allocate_Matrix::start" );
	hip_malloc( (void **) &qeq_gpu->d, sizeof(double) * size, TRUE,
			"Hip_Allocate_Matrix::start" );
}


void  HipInitStorageForFixQeq(fix_qeq_gpu *qeq_gpu, double *Hdia_inv, double *b_s,double *b_t,double *b_prc,double
                              *b_prm,double *s,double *t, int NN)
{
	sHipMemcpy( Hdia_inv, qeq_gpu->Hdia_inv, sizeof(double) * NN,
			hipMemcpyHostToDevice, __FILE__, __LINE__ );
	sHipMemcpy( b_s, qeq_gpu->b_s, sizeof(double) * NN,
			hipMemcpyHostToDevice, __FILE__, __LINE__ );
	sHipMemcpy( b_t, qeq_gpu->b_t, sizeof(double) * NN,
            hipMemcpyHostToDevice, __FILE__, __LINE__ );
	sHipMemcpy( b_prc, qeq_gpu->b_prc, sizeof(double) * NN,
            hipMemcpyHostToDevice, __FILE__, __LINE__ );
	sHipMemcpy( b_prm, qeq_gpu->b_prm, sizeof(double) * NN,
            hipMemcpyHostToDevice, __FILE__, __LINE__ );
	sHipMemcpy( s, qeq_gpu->s, sizeof(double) * NN,
            hipMemcpyHostToDevice, __FILE__, __LINE__ );
	sHipMemcpy(t, qeq_gpu->t, sizeof(double) * NN,
            hipMemcpyHostToDevice, __FILE__, __LINE__ );


}

void  Hip_Init_Fix_Atoms(reax_system *system,fix_qeq_gpu *qeq_gpu)

{
	hip_malloc( (void **) &qeq_gpu->d_fix_my_atoms, sizeof(reax_atom) * system->N, TRUE,
			"Hip_Allocate_Matrix::start" );
	sHipMemcpy(qeq_gpu->fix_my_atoms, qeq_gpu->d_fix_my_atoms, sizeof(reax_atom) * system->N,
               hipMemcpyHostToDevice, __FILE__, __LINE__);
}

void Hip_Free_Memory(fix_qeq_gpu *qeq_gpu)
{
	hip_free(qeq_gpu->d_fix_my_atoms, "fix_qeq_gpu::d_fix_my_atoms");
	hip_free(qeq_gpu->d_cm_entries, "fix_qeq_gpu::d_cm_entries");
	hip_free(qeq_gpu->d_max_cm_entries, "fix_qeq_gpu::d_max_cm_entries");
	hip_free(qeq_gpu->d_total_cm_entries, "fix_qeq_gpu::d_total_cm_entries");
	hip_free(qeq_gpu->H.start, "fix_qeq_gpu::H.start");
	hip_free(qeq_gpu->H.end, "fix_qeq_gpu::H.end");
	hip_free(qeq_gpu->H.j, "fix_qeq_gpu::H.j");
	hip_free(qeq_gpu->H.val, "fix_qeq_gpu::H.val");
}


HIP_DEVICE real Init_Charge_Matrix_Entry(real *workspace_Tap,
                                          int i, int j, real r_ij, real gamma)
                                          {
    real Tap,denom;

    Tap = workspace_Tap[7] * r_ij + workspace_Tap[6];
    Tap = Tap * r_ij + workspace_Tap[5];
    Tap = Tap * r_ij + workspace_Tap[4];
    Tap = Tap * r_ij + workspace_Tap[3];
    Tap = Tap * r_ij + workspace_Tap[2];
    Tap = Tap * r_ij + workspace_Tap[1];
    Tap = Tap * r_ij + workspace_Tap[0];

    denom = r_ij * r_ij * r_ij + gamma;
    denom = POW(denom, 1.0 / 3.0 );

    return Tap * EV_to_KCALpMOL / denom;
}


HIP_GLOBAL void k_init_cm_full_fs( reax_atom *my_atoms,
                                   reax_list far_nbr_list, sparse_matrix* H,
                                   control_params *control, int n, double *d_Tap, double *gamma,
                                   int *max_cm_entries, int *realloc_cm_entries)
                                   {



    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int cm_top;
    int num_cm_entries;
    real r_ij;
//    two_body_parameters *tbp;
    reax_atom *atom_i, *atom_j;
    far_neighbor_data *nbr_pj;
    double shld;
    double dx, dy, dz;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= H->n_max)
    {
        return;
    }

    cm_top = H->start[i];

    if (i < H->n) {

        atom_i = &my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index(i, &far_nbr_list );

        /* diagonal entry in the matrix */
        H->j[cm_top] = i;
        H->val[cm_top] = Init_Charge_Matrix_Entry(d_Tap,
                                                  i, i, 0.0, 0.0);
        ++cm_top;


        for ( pj = start_i; pj < end_i; ++pj )
        {

            if ( far_nbr_list.far_nbr_list.d[pj] <= control->nonb_cut)
            {
                //if ( i == 500)
                //printf("%d,%d,%f,%f\n", i, j, nbr_pj->d,nonb_cut);
                j = far_nbr_list.far_nbr_list.nbr[pj];
                atom_j = &my_atoms[j];
                type_j = atom_j->type;

                r_ij = far_nbr_list.far_nbr_list.d[pj];

                H->j[cm_top] = j;
                shld = pow( gamma[type_i] * gamma[type_j], -1.5);
                H->val[cm_top] = Init_Charge_Matrix_Entry(d_Tap, i, j, r_ij, shld);
                ++cm_top;

            }

        }
    }
    __syncthreads();


    H->end[i] = cm_top;
    num_cm_entries = cm_top - H->start[i];

    /* reallocation check */
    if ( num_cm_entries > max_cm_entries[i] )
    {
        *realloc_cm_entries = TRUE;
    }
}




void  Hip_Calculate_H_Matrix(reax_list **lists,  reax_system *system, fix_qeq_gpu *qeq_gpu,
                             control_params *control, int n)
{

	int blocks;
	blocks = (n) / DEF_BLOCK_SIZE +
			(((n % DEF_BLOCK_SIZE) == 0) ? 0 : 1);

	hipLaunchKernelGGL(k_init_dist, dim3(blocks), dim3(DEF_BLOCK_SIZE), 0, 0,
                       qeq_gpu->d_fix_my_atoms, *(lists[FAR_NBRS]), n );
	hipCheckError();
	hipDeviceSynchronize();

	//printf("nonb %f, %d\n",control->nonb_cut,n );


	//printf("Blocks %d , blocks size %d\n", blocks, DEF_BLOCK_SIZE);
	//printf("N %d, h n %d \n",system->N, qeq_gpu->H.n);

	hipLaunchKernelGGL(k_init_cm_full_fs , dim3(blocks), dim3(DEF_BLOCK_SIZE), 0, 0,  qeq_gpu->d_fix_my_atoms,
                       *(lists[FAR_NBRS]), &(qeq_gpu->H), (control_params *) control->d_control_params, n, qeq_gpu->d_Tap,qeq_gpu->gamma,
                       qeq_gpu->d_max_cm_entries,
                       qeq_gpu->d_realloc_cm_entries);
	hipDeviceSynchronize();


}

void Hip_Init_Taper(fix_qeq_gpu *qeq_gpu, double *Tap, int numTap)
{
	hip_malloc( (void **) &qeq_gpu->d_Tap, sizeof(double)*numTap, TRUE,
			"Hip_Allocate_Matrix::start");
	sHipMemcpy(Tap, qeq_gpu->d_Tap, sizeof(double) * numTap,
			hipMemcpyHostToDevice, __FILE__, __LINE__);

}




void Hip_Estimate_CMEntries_Storages( reax_system *system, control_params *control, reax_list **lists,
                                      fix_qeq_gpu *qeq_gpu,int n)
{
	int blocks;

	hip_malloc( (void **) &qeq_gpu->d_cm_entries,
			n * sizeof(int), TRUE, "system:d_cm_entries" );
	hipCheckError();

	hip_malloc( (void **) &qeq_gpu->d_max_cm_entries,
			n * sizeof(int), TRUE, "system:d_cm_entries" );
	hip_malloc( (void **) &qeq_gpu->d_total_cm_entries,
			n * sizeof(int), TRUE, "system:d_cm_entries" );


	blocks = n / DEF_BLOCK_SIZE +
			(((n % DEF_BLOCK_SIZE == 0)) ? 0 : 1);

	hipLaunchKernelGGL(k_estimate_storages_cm_full, dim3(blocks), dim3(DEF_BLOCK_SIZE), 0, 0,
                       (control_params *)control->d_control_params,
			*(lists[FAR_NBRS]), n, n,
			qeq_gpu->d_cm_entries, qeq_gpu->d_max_cm_entries);
	hipDeviceSynchronize();
	hipCheckError();


	//TB:: Should max_cm or cm entries be used for calculating total_cm_entries
	Hip_Reduction_Sum(qeq_gpu->d_max_cm_entries, qeq_gpu->d_total_cm_entries, n);
	sHipMemcpy( &system->total_cm_entries, qeq_gpu->d_total_cm_entries, sizeof(int),
			hipMemcpyDeviceToHost, __FILE__, __LINE__ );
}


HIP_GLOBAL void k_init_matvec_fix(fix_qeq_gpu d_qeq_gpu,int n, single_body_parameters
		*sbp,reax_atom *my_atoms)
{
	int i;
	int type_i;
	fix_qeq_gpu *qeq_gpu;
	qeq_gpu = &d_qeq_gpu;


	i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= n)
	{
		return;
	}
	reax_atom *atom;
	atom = &my_atoms[i];
	type_i = atom->type;


	qeq_gpu->Hdia_inv[i] = 1. / qeq_gpu->eta[type_i];
	qeq_gpu->b_s[i] = -qeq_gpu->chi[type_i];
	qeq_gpu->b_t[i] = -1.0;



	qeq_gpu->t[i] = qeq_gpu->t_hist[i][2] + 3 * ( qeq_gpu->t_hist[i][0] - qeq_gpu->t_hist[i][1]);
	/* cubic extrapolation for s & t from previous solutions */
	qeq_gpu->s[i] = 4*(qeq_gpu->s_hist[i][0]+qeq_gpu->s_hist[i][2])-(6*qeq_gpu->s_hist[i][1]+qeq_gpu->s_hist[i][3]);
}

void  Hip_Init_Matvec_Fix(int n, fix_qeq_gpu *qeq_gpu, reax_system *system)
{
	int blocks;

	blocks = n / DEF_BLOCK_SIZE
			+ (( n % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);

	hipLaunchKernelGGL(k_init_matvec_fix, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0, *(qeq_gpu),n,system->reax_param.d_sbp,qeq_gpu->d_fix_my_atoms);
	hipDeviceSynchronize();
	hipCheckError();
}

void  Hip_Copy_Pertype_Parameters_To_Device(double *chi,double *eta,double *gamma,int ntypes,fix_qeq_gpu *qeq_gpu)
{
	hip_malloc( (void **) &qeq_gpu->gamma, sizeof(double)*(ntypes+1), TRUE,
			"Hip_Allocate_Matrix::start");
	sHipMemcpy(gamma, qeq_gpu->gamma, sizeof(double) * (ntypes+1),
			hipMemcpyHostToDevice, __FILE__, __LINE__);
	hip_malloc( (void **) &qeq_gpu->chi, sizeof(double)*(ntypes+1), TRUE,
			"Hip_Allocate_Matrix::start");
	sHipMemcpy(chi, qeq_gpu->chi, sizeof(double) * (ntypes+1),
               hipMemcpyHostToDevice, __FILE__, __LINE__);
	hip_malloc( (void **) &qeq_gpu->eta, sizeof(double)*(ntypes+1), TRUE,
			"Hip_Allocate_Matrix::start");
	sHipMemcpy(eta, qeq_gpu->eta, sizeof(double) * (ntypes+1),
               hipMemcpyHostToDevice, __FILE__, __LINE__);

}

void  Hip_Copy_From_Device_Comm_Fix(double *buf, double *x, int n, int offset)
{
	sHipMemcpy(buf, x+offset, sizeof(double) * n,
               hipMemcpyDeviceToHost, __FILE__, __LINE__ );
}


HIP_GLOBAL void k_update_buf(double *dev_buf, double *x, int nn, int offset)
{
	int i, c, col;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i >= nn )
	{
		return;
	}


	x[i+offset] = dev_buf[i];
}


void  Hip_Copy_To_Device_Comm_Fix(double *buf,double *x,int nn,int offset)
{

	double *dev_buf;
	hip_malloc( (void **) &dev_buf, sizeof(double)*nn, TRUE,
			"Hip_Allocate_Matrix::start");
	sHipMemcpy(buf,dev_buf,sizeof(double)*nn,hipMemcpyHostToDevice, __FILE__, __LINE__);


	int blocks;

	blocks = nn / DEF_BLOCK_SIZE
			+ (( nn % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);
	//printf("Blocks %d \n",blocks);

	hipLaunchKernelGGL(k_update_buf, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0, dev_buf,x,nn, offset);
	hipDeviceSynchronize();

	hipFree(dev_buf);

	/*sHipMemCopy(buf, x+offset, sizeof(double) * nn,
				hipMemcpyHostToDevice, __FILE__, __LINE__ );*/

}


HIP_GLOBAL void k_update_q(double *temp_buf, double *q, int nn)
{
	int i, c, col;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i >= nn )
	{
		return;
	}


	q[i] = q[i] +  temp_buf[i];

	//printf("m: %d %f\n",i, q[i]);

}

void  Hip_UpdateQ_And_Copy_To_Device_Comm_Fix(double *buf,fix_qeq_gpu *qeq_gpu,int nn)
{
	double *temp_buf;
	hip_malloc( (void **) &temp_buf, sizeof(double)*nn, TRUE,
			"Hip_Allocate_Matrix::start");
	sHipMemcpy(buf, temp_buf, sizeof(double) * nn,
               hipMemcpyHostToDevice, __FILE__, __LINE__);

	int blocks;

	blocks = nn / DEF_BLOCK_SIZE
			+ (( nn % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);
	//printf("Blocks %d \n",blocks);


	hipLaunchKernelGGL(k_update_q, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0, temp_buf,qeq_gpu->q,nn);
	hipDeviceSynchronize();

}



HIP_GLOBAL void k_matvec_csr_fix( sparse_matrix* H, real *vec, real *results,
		int num_rows)
{

	int i, c, col;
	real results_row;
	real val;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i >= num_rows )
	{
		return;
	}



	results_row = results[i];


	int iter = 0;
	for ( c = H->start[i]; c < H->end[i]; c++ )
	{
		col = H->j[c];
		val = H->val[c];
		results_row += val * vec[col];
		iter++;


	}

	__syncthreads();


	results[i] = results_row;
}

HIP_GLOBAL void k_init_q(reax_atom *my_atoms, double *q, double *x,double *eta, int nn, int NN)
{

	int i;
	int type_i;
	reax_atom *atom;


	i = blockIdx.x * blockDim.x + threadIdx.x;

	if ( i >= NN)
	{
		return;
	}

	if (i < nn ) {
		atom = &my_atoms[i];
		type_i = atom->type;


		q[i] = eta[type_i] * x[i];

		//printf("i %d, eta %f, x %f, q%f\n ", i, eta[type_i],x[i],q[i]);
	}
	else
	{
		q[i] = 0.0;

	}



}


void Hip_Sparse_Matvec_Compute(sparse_matrix *H,double *x, double *q, double *eta, reax_atom *d_fix_my_atoms, int nn, int NN)
{

	int blocks;

	blocks = NN / DEF_BLOCK_SIZE
			+ (( NN % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);
	//printf("Blocks %d \n",blocks);


	hipLaunchKernelGGL(k_init_q, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0, d_fix_my_atoms,q,x,eta,nn,NN);
	hipDeviceSynchronize();



	blocks = nn / DEF_BLOCK_SIZE
			+ (( nn % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);
	//printf("Blocks %d \n",blocks);


	hipLaunchKernelGGL(k_matvec_csr_fix, dim3(blocks), dim3(DEF_BLOCK_SIZE), 0 , 0, H, x, q, nn);
	hipDeviceSynchronize();
	hipCheckError();

	//printf("\n\n");
}

void Hip_Vector_Sum_Fix( real *res, real a, real *x, real b, real *y, int count )
{
	//res = ax + by
	//use the cublas here
	int blocks;

	blocks = (count / DEF_BLOCK_SIZE)+ ((count % DEF_BLOCK_SIZE == 0) ? 0 : 1);

	hipLaunchKernelGGL(k_vector_sum, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  res, a, x, b, y, count );
	hipDeviceSynchronize();
	hipCheckError();
}

void Hip_CG_Preconditioner_Fix(real *res, real *a, real *b, int count)
{

	int blocks;

	blocks = (count / DEF_BLOCK_SIZE) + ((count % DEF_BLOCK_SIZE == 0) ? 0 : 1);


	hipLaunchKernelGGL(k_vector_mult, dim3(blocks), dim3(DEF_BLOCK_SIZE ), 0, 0,  res, a, b, count );
	hipDeviceSynchronize( );
	hipCheckError( );

}

void  Hip_Copy_Vector_To_Device(real *host_vector, real *device_vector, int nn)
{
	sHipMemcpy( host_vector, device_vector, sizeof(real) * nn,
                hipMemcpyHostToDevice, __FILE__, __LINE__ );

}


void  Hip_Copy_Vector_From_Device(real *host_vector, real *device_vector, int nn)
{
	sHipMemcpy( host_vector, device_vector, sizeof(real) * nn,
                hipMemcpyDeviceToHost, __FILE__, __LINE__);
}


int  compute_nearest_pow_2_fix( int blocks)
{

	int result = 0;
	result = (int) EXP2( CEIL( LOG2((double) blocks)));
	return result;
}





float  Hip_Calculate_Local_S_Sum(int nn,fix_qeq_gpu *qeq_gpu)
{
	int blocks;
	real *output;
	//hip malloc this
	hip_malloc((void **) &output, sizeof(real), TRUE,
			"Hip_Allocate_Matrix::start");
	double my_acc;


	blocks = nn / DEF_BLOCK_SIZE
			+ (( nn % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);


	Hip_Reduction_Sum(qeq_gpu->s, output, nn);

	my_acc = 0;

	sHipMemcpy( &my_acc, output,
                sizeof(real), hipMemcpyDeviceToHost, __FILE__, __LINE__ );

	return my_acc;
}

float  Hip_Calculate_Local_T_Sum(int nn,fix_qeq_gpu *qeq_gpu)
{
	int blocks;
	real *output;
	//hip malloc this
	hip_malloc((void **) &output, sizeof(real), TRUE,
			"Hip_Allocate_Matrix::start");
	double my_acc;


	blocks = nn / DEF_BLOCK_SIZE
			+ (( nn % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);

	Hip_Reduction_Sum(qeq_gpu->t, output, nn);



	my_acc = 0;

	sHipMemcpy( &my_acc, output,
                sizeof(real), hipMemcpyDeviceToHost, __FILE__, __LINE__ );



	return my_acc;

}

HIP_GLOBAL void k_update_q_and_backup_st(double *q, double *s, double *t, reax_atom *my_atoms,rvec4 *s_hist, rvec4 *t_hist, double u, int nn,reax_atom *sys_my_atoms)
{

	int i;
	int type_i;


	i = blockIdx.x * blockDim.x + threadIdx.x;

	reax_atom *atom;
	atom = &my_atoms[i];

	reax_atom *atom2;
	atom2 = &sys_my_atoms[i];


	if ( i >= nn)
	{
		return;
	}

	q[i]  = atom->q  = sys_my_atoms[i].q = s[i] - u*t[i];

	//printf("S[%d] %f,T[%d] %f,Q[%d] %f \n",i, s[i],i,t[i],i,sys_my_atoms[i].q);


	s_hist[i][3] = s_hist[i][2];
	s_hist[i][2] = s_hist[i][1];
	s_hist[i][1] = s_hist[i][0];
	s_hist[i][0] = s[i];


	t_hist[i][3] = t_hist[i][2];
	t_hist[i][2] = t_hist[i][1];
	t_hist[i][1] = t_hist[i][0];
	t_hist[i][0] = t[i];
}

void  Hip_Update_Q_And_Backup_ST(int nn, fix_qeq_gpu *qeq_gpu, double u,reax_system *system)
{
	int blocks;
	blocks = nn / DEF_BLOCK_SIZE
			+ (( nn % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);

	hipLaunchKernelGGL(k_update_q_and_backup_st, dim3(blocks), dim3(DEF_BLOCK_SIZE), 0, 0,   qeq_gpu->q,qeq_gpu->s,qeq_gpu->t,qeq_gpu->d_fix_my_atoms,qeq_gpu->s_hist,qeq_gpu->t_hist,u, nn, system->d_my_atoms);
	hipDeviceSynchronize();
	hipCheckError();

	sHipMemcpy(qeq_gpu->fix_my_atoms, qeq_gpu->d_fix_my_atoms, sizeof(reax_atom) * system->N,
               hipMemcpyDeviceToHost, __FILE__, __LINE__);
}

void  HipFreeFixQeqParams(fix_qeq_gpu *qeq_gpu)
{
	   hip_free(qeq_gpu->s, "S");
       hip_free(qeq_gpu->t, "T");
       hip_free(qeq_gpu->Hdia_inv, "Hdia");
       hip_free(qeq_gpu->b_s, "B_S");
       hip_free(qeq_gpu->b_t, "B_T");
       hip_free(qeq_gpu->b_prc, "PRC");
       hip_free(qeq_gpu->b_prm, "PRM");

       hip_free(qeq_gpu->p, "P");
       hip_free(qeq_gpu->q, "Q");
       hip_free(qeq_gpu->r, "R");
       hip_free(qeq_gpu->d, "D");
}

void HipFreeHMatrix(fix_qeq_gpu *qeq_gpu)
{
    Hip_Deallocate_Matrix(&qeq_gpu->H);
}


}
