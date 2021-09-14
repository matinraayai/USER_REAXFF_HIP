#ifndef __CUDA_FIX_QEQ_H_
#define __CUDA_FIX_QEQ_H_

#include "reaxff_types.h"


#ifdef __cplusplus
extern "C"  {
#endif

void  HipAllocateStorageForFixQeq(int nmax, int dual_enabled, fix_qeq_gpu *qeq_gpu);
void Hip_Allocate_Hist_ST(fix_qeq_gpu *qeq_gpu,int nmax);
void  HipInitStorageForFixQeq(fix_qeq_gpu *qeq_gpu, double *Hdia_inv, double *b_s,double *b_t,double *b_prc,double *b_prm,double *s,double *t, int N);
void  Hip_Calculate_H_Matrix(reax_list **gpu_lists,  reax_system *system,fix_qeq_gpu *qeq_gpu, control_params *control, int inum);
void  Hip_Init_Taper(fix_qeq_gpu *qeq_gpu,double *Tap, int numTap);
void  Hip_Init_Fix_Atoms(reax_system *system,fix_qeq_gpu *qeq_gpu);
void  Hip_Init_Matvec_Fix(int nn, fix_qeq_gpu *qeq_gpu, reax_system *system);
void  Hip_Copy_Pertype_Parameters_To_Device(double *chi,double *eta,double *gamma,int ntypes,fix_qeq_gpu *qeq_gpu);
void Hip_Copy_From_Device_Comm_Fix(double *buf, double *x, int n, int offset);
void Hip_Copy_To_Device_Comm_Fix(double *buf, double *x, int n, int offset);
void  Hip_Sparse_Matvec_Compute(sparse_matrix *H,double *x, double *q, double *eta, reax_atom *d_fix_my_atoms, int nn, int NN);
void Hip_Vector_Sum_Fix( real *, real, real *, real, real *, int );
void Hip_CG_Preconditioner_Fix( real *, real *, real *, int );
void  Hip_Copy_Vector_From_Device(real *host_vector, real *device_vector, int nn);
//void Hip_Calculate_Q(int nn,fix_qeq_gpu *qeq_gpu,int blocks);
//void  Hip_Parallel_Vector_Acc(int nn,double *x);
void  Hip_UpdateQ_And_Copy_To_Device_Comm_Fix(double *buf,fix_qeq_gpu *qeq_gpu,int n);
void  HipFreeFixQeqParams(fix_qeq_gpu *qeq_gpu);
void HipFreeHMatrix(fix_qeq_gpu *qeq_gpu);
void Hip_Free_Memory(fix_qeq_gpu *qeq_gpu);
void  Hip_UpdateQ_And_Copy_To_Device_Comm_Fix(double *buf,fix_qeq_gpu *qeq_gpu,int nn);
float  Hip_Calculate_Local_S_Sum(int nn,fix_qeq_gpu *qeq_gpu);
void  Hip_Calculate_H_Matrix(reax_list **lists,  reax_system *system, fix_qeq_gpu *qeq_gpu,
                             control_params *control, int n);
float  Hip_Calculate_Local_T_Sum(int nn,fix_qeq_gpu *qeq_gpu);
void  Hip_Update_Q_And_Backup_ST(int nn, fix_qeq_gpu *qeq_gpu, double u,reax_system *system);
void  Hip_Copy_Vector_To_Device(real *host_vector, real *device_vector, int nn);
void Hip_Estimate_CMEntries_Storages( reax_system *system, control_params *control, reax_list **lists,
                                      fix_qeq_gpu *qeq_gpu,int n);

#ifdef __cplusplus
}
#endif

#endif
