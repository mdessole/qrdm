#include <lapacke.h>
#include "lapacke_utils.h"


lapack_int dgeqrdm( int matrix_layout, lapack_int m, lapack_int n,
		      double* a, lapack_int lda, lapack_int* jpvt,
		      double* tau,  lapack_int* ncols, double *thres,
		      int nb)
{
lapack_int info = 0;

/* Call middle-level interface */
info = dgeqrdm_work( matrix_layout, m, n, a, lda, jpvt, tau, ncols, thres, nb );

return info;
}
