/**
 * @file DMQR.h
 * @author DM
 * @date 13/12/2020
 * @brief QRDM public interface
 */

#ifndef QRDM_H_
#define QRDM_H_

#include "stdint.h"
#include <lapacke.h>
#include <cblas.h>

typedef lapack_int integer; // int
typedef double lapack_double; // doublereal
typedef lapack_double doublereal;

lapack_int dgeqrdm( int matrix_layout, lapack_int m, lapack_int n,
		    double* a, lapack_int lda, lapack_int* jpvt,
		    double* tau, lapack_int* ncols, double *thres,
		    int DM_procedure);

//void modelInit(void);
//float modelSetExternalOutputs(uint8_t index);
//void modelOverwriteParameter(uint16_t index, float value);

			     
#endif /* QRDM_H_ */
