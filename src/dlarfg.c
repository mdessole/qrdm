/* dlarfp.f -- translated by f2c (version 20061008).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include "lapacke_utils.h"

typedef double doublereal;
typedef int integer;

double d_sign(doublereal *a, doublereal *b)
{
double x;
x = (*a >= 0 ? *a : - *a);
return( *b >= 0 ? x : -x);
}

/* Subroutine */ int dlarfg_mia(integer *n, doublereal *alpha, doublereal *x, 
				integer *incx, doublereal *tau, doublereal *thres,
				int *idx_house)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1;


    /* Local variables */
    integer j, knt;
    doublereal beta;
    //extern doublereal dnrm2_(integer *, doublereal *, integer *);
    //extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
    //integer *);
    doublereal xnorm;
    extern doublereal dlapy2_(doublereal *, doublereal *);
    //, dlamch_(char *);
    doublereal safmin, rsafmn;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLARFG generates a real elementary reflector H of order n, such */
/*  that */

/*        H * ( alpha ) = ( beta ),   H' * H = I. */
/*            (   x   )   (   0  ) */

/*  where alpha and beta are scalars, and x is */
/*  an (n-1)-element real vector.  H is represented in the form */

/*        H = I - tau * ( 1 ) * ( 1 v' ) , */
/*                      ( v ) */

/*  where tau is a real scalar and v is a real (n-1)-element */
/*  vector. */

/*  If the elements of x are all zero, then tau = 0 and H is taken to be */
/*  the unit matrix. */

/*  Otherwise  1 <= tau <= 2. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the elementary reflector. */

/*  ALPHA   (input/output) DOUBLE PRECISION */
/*          On entry, the value alpha. */
/*          On exit, it is overwritten with the value beta. */

/*  X       (input/output) DOUBLE PRECISION array, dimension */
/*                         (1+(N-2)*abs(INCX)) */
/*          On entry, the vector x. */
/*          On exit, it is overwritten with the vector v. */

/*  INCX    (input) INTEGER */
/*          The increment between elements of X. INCX > 0. */

/*  TAU     (output) DOUBLE PRECISION */
/*          The value tau. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --x;

    /* Function Body */
    if (*n <= 1) {
	*tau = 0.;
	return 0;
    }

    i__1 = *n - 1;
    xnorm = cblas_dnrm2(i__1, &x[1], *incx);

    
    if ((idx_house[0] > 1) && (xnorm < thres[0] )){ //  < thres[0]  //((*idx_house > 1) && ((xnorm/fabs(*diag_last)) <= 1.e-3 ))
      //printf("found %d\n", *idx_house);
      //      printf("1 if branch\n");
      return 1;
    }
    else if (xnorm == 0.) {
      //printf("2 if branch\n");
      /*        H  =  I */
      
      *tau = 0.;
    } else {
      //printf("3 if branch\n");
      
      /*        general case */
      
      d__1 = dlapy2_(alpha, &xnorm);
      beta = -d_sign(&d__1, alpha);
      safmin = LAPACKE_dlamch('S') / LAPACKE_dlamch('E');
      knt = 0;
      if (fabs(beta) < safmin) {
	
	/*           XNORM, BETA may be inaccurate; scale X and recompute them */
	
	rsafmn = 1. / safmin;
      L10:
	++knt;
	i__1 = *n - 1;
	cblas_dscal(i__1, rsafmn, &x[1], *incx);
	beta *= rsafmn;
	*alpha *= rsafmn;
	if (fabs(beta) < safmin) {
	  goto L10;
	}
	
	/*           New BETA is at most 1, at least SAFMIN */
	
	i__1 = *n - 1;
	xnorm = cblas_dnrm2(i__1, &x[1], *incx);
	d__1 = dlapy2_(alpha, &xnorm);
	beta = -d_sign(&d__1, alpha);
      }
      *tau = (beta - *alpha) / beta;
      i__1 = *n - 1;
      d__1 = 1. / (*alpha - beta);
      cblas_dscal(i__1, d__1, &x[1], *incx);
      
      /*        If ALPHA is subnormal, it may lose relative accuracy */
      
      i__1 = knt;
      for (j = 1; j <= i__1; ++j) {
	beta *= safmin;
	/* L20: */
      }
      *alpha = beta;
    }
    
    return 0;
    /*     End of DLARFP */
    
} /* dlarfp_ */
