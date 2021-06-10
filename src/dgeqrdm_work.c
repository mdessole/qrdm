#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include "lapacke_utils.h"
#include <stdio.h>

/* Table of constant values */

static int c__1 = 1;
static int c_n1 = -1;
static int c__3 = 3;
static int c__2 = 2;

#define min(a,b) ((a) <= (b) ? (a) : (b))
#define max(a,b) ((a) >= (b) ? (a) : (b))

#define LAPACK_ROW_MAJOR               101
#define LAPACK_COL_MAJOR               102

#define DEBUGMODE

typedef struct 
{ double val;
  int idx;
}wrapStruct;

int mark_argmin(int * vec, int *n){
  int j = 0;
  while(vec[j]==1 && j<n){
    j++;
  }
  return j;
}

void norm_update(int m, int n, int fjb,
		 double *a, int lda,
		 double *work, double *workOLD, double *maxnrm,
		 wrapStruct *tmpStr,
		 int * workint,
		 double tol3z){
  /* This function safely updates the square of the column norms
     after having triangularized a block of fjb columns. Norm update
     is performed according to
     ---
     "On the failure of rank revealing QR factorization software - a case study",
     Zlatko Drmac and Zvonimir Bujanovic,
     ACM transaction on Mathematical Software.
     ---
     Arguments
     m = nb of rows
     n = nb of columns
     fjb = nb of colums previously factorized
     a = m,n columnwise matrix
     work = previous column norms
     workOLD = last value of column norm computed from scratch
     maxnrm = maximum norm value (output, estimate of smaller singular value)
     tmpStr = auxil. struct used in DM, updated here for grouping together different for loops
     workint = auxil. memory used in DM, updated here for grouping together different for loops
     tol3z = sqrt(working machine epsilon)
  */
  int ii, jj, flag = 0;
  double temp, temp2, d__1, d__2;
  // update column norms
  *maxnrm = 0.0;
  
  if (0){ // norm computation from scratch
    for(jj = 0; jj<n; jj++){
      work[jj] = cblas_dnrm2(m, &a[jj*lda+fjb], c__1);
      
      if (*maxnrm < work[jj])
	*maxnrm = work[jj];
      if (1){
	tmpStr[jj].val = work[jj];
	tmpStr[jj].idx = jj;	
	workint[jj] = 0;
      }
    }
  }
  else{ // norm update
    for(jj = 0; jj<n; jj++){
      if (work[jj] != 0.){	
	d__1 = 0.0;
	for(ii=0; ii<fjb;ii++){
	  d__1 += a[jj*lda+ii]*a[jj*lda+ii];
	}
	temp = (sqrt(fabs(d__1))) / work[jj];

	d__1 = 0., d__2 = (temp + 1.) * (1. - temp);
	temp = max(d__1,d__2);
	/* Computing 2nd power */
	d__1 = work[jj] / workOLD[jj];
	temp2 = temp * (d__1 * d__1);
	if (temp2 <= tol3z) {
	  if (m > 0) {
	    work[jj] = cblas_dnrm2(m, &a[jj*lda+fjb], c__1);
	    workOLD[jj] = work[jj];
	  } else {
	    work[jj] = 0.;
	    workOLD[jj] = 0.;
	  }
	} else {
	  work[jj] *= sqrt(temp);
	}
		
	if (*maxnrm < work[jj])
	  *maxnrm = work[jj];
	
      }
      if (1){ // initialize auxiliary arrays
	tmpStr[jj].val = work[jj];
	tmpStr[jj].idx = jj;	
	workint[jj] = 0;
      }
    }
  }
  
  
  return;

}



int cmpStruct (const void * a, const void * b)
{

  wrapStruct *wrapStructA = (wrapStruct *)a;
  wrapStruct *wrapStructB = (wrapStruct *)b;

  double valA = wrapStructA->val;
  double valB = wrapStructB->val;

  if (valA > valB)
    return -1;
  else if (valA < valB)
    return 1;
  else
    return 0;
 
}



void apply_jpvt(int onebased,
		int m, int n,
		double *a, int lda,
		int *jpvt, int *jpvtdef,
		double *norms, int *mark, double *tmpcol ){
  /* this function permutes the columns of a accordingly to the order specified by the vector jpvt
     onebased = 1 if indices in jvpt start from 1, 0 otherwise (indices start from 0)
     m = the number of rows of a
     n = the number of columns, and the length of jvt
     a = matrix of size m*n
     lda = leading dimension of a  = nb of rows
     jpvt = output of DM = new ordering for the n columns of a 
     jpvtdef = current pivoting for the matrix a (and previous columns, if any)
     norms = vector of column norms
     mark = integer workspace
     tmpcol = double workspace = m doubles to perform the swap of two columns

     **auxiliary memory**
     n int (mark)
     m double (tmpcol)
  */

  
  int  marked = 0, i, j, tmp, tmp2;  
  double tmpnorm; 

  int ii;

  if (onebased){
    for (ii=0;ii<n;ii++){
      jpvt[ii] -= 1;
      mark[ii] = 0;
    }
  }
  else{
    for (ii=0;ii<n;ii++)
      mark[ii] = 0; 
  }
  
  int nscambi = 0;
  marked = 0;
  while (marked<n){
    i = mark_argmin(mark,n); /* Find column i to be moved */
    tmp = i;
    tmp2 = jpvtdef[i];
    tmpnorm = norms[i];
    /*Copy i-th column into tmpcol*/
    for (j=0;j<m;j++)
      tmpcol[j] = a[i*lda+j];
    while((jpvt[i]!=tmp) && (marked<n)){
      if (i!=jpvt[i]){
	/* Swap columns of indices i and jpvt[i] */
	for (j=0;j<m;j++)
	  a[i*lda+j] = a[jpvt[i]*lda+j];
	jpvtdef[i] = jpvtdef[jpvt[i]];
	norms[i]   = norms[jpvt[i]];
	nscambi +=1;
	i = jpvt[i];
      }
      mark[i]    = 1;
      marked++;
    }
    /*Copy i-th column from tmpcol*/
    for (j=0;j<m;j++)
      a[i*lda+j] = tmpcol[j];
    jpvtdef[i] = tmp2;
    norms[i] = tmpnorm;
    mark[i] = 1;
    marked++;
    nscambi +=1;
  }

  return;
}


void permute_marked(int m, int n, int ncols,
		    double *a, int lda,
		    int *jpvt, double *norms,
		    int *marked,
		    int *idxI, int nb_idxI,
		    int nz){
  int jb = 0, jblda;
  int jt = ncols-1, jtlda;
  int nscambi = 0;
  int i, j, jc, itmp, jclda;
  double dtmp;


  for (j=0;j<nz+nb_idxI;j++){
    jc = idxI[j];
    
    while ((jb<jt) && ((marked[jt] == 1) || (marked[jb]==2))){
      itmp = marked[jt];
      marked[jt] = marked[jb];
      marked[jb] = itmp;
      itmp = jpvt[jt];
      jpvt[jt] = jpvt[jb];
      jpvt[jb] = itmp;
      dtmp = norms[jt];
      norms[jt] = norms[jb];
      norms[jb] = dtmp;
      jblda = jb*lda;
      jtlda = jt*lda;
      for (i=0;i<m;i++){
	dtmp = a[jtlda];
	a[jtlda] = a[jblda];
	a[jblda] = dtmp;
	jblda++;
	jtlda++;
      }
      nscambi +=1;
      while((jb<ncols) && (marked[jb]!=0) && (marked[jb]!=2))
	jb+=1;
      while((jt>=0) && (marked[jt]!=0) && (marked[jt]!=1))
	jt-=1;
    }
    

    if (marked[jc]==1){
      while((jb<ncols) && (marked[jb]!=0) && (marked[jb]!=2))
	jb+=1;
      if ((jc<=jb) || (jc< nb_idxI))
	continue;
      if (marked[jb]==0){
	itmp = marked[jc];
	marked[jc] = marked[jb];
	marked[jb] = itmp;
	itmp = jpvt[jc];
	jpvt[jc] = jpvt[jb];
	jpvt[jb] = itmp;
	dtmp = norms[jc];
	norms[jc] = norms[jb];
	norms[jb] = dtmp;
	jblda = jb*lda;
	jclda = jc*lda;
	for (i=0;i<m;i++){
	  dtmp = a[jclda];
	  a[jclda] = a[jblda];
	  a[jblda] = dtmp;
	  jblda++;
	  jclda++;
	}
	nscambi +=1;
	jb+=1;
      }
    }else if (marked[jc]==2){
      while((jt>=0) && (marked[jt]!=0) && (marked[jt]!=1))
	jt-=1;
      if ((jc>=jt) || (jc >= ncols-nz))
	continue;
      if (marked[jt]==0){
	itmp = marked[jc];
	marked[jc] = marked[jt];
	marked[jt] = itmp;
	itmp = jpvt[jc];
	jpvt[jc] = jpvt[jt];
	jpvt[jt] = itmp;
	dtmp = norms[jc];
	norms[jc] = norms[jt];
	norms[jt] = dtmp;
	jclda = jc*lda;
	jtlda = jt*lda;
    	for (i=0;i<m;i++){
	  dtmp = a[jclda];
	  a[jclda] = a[jtlda];
	  a[jtlda] = dtmp;
	  jclda++;
	  jtlda++;
	}
	nscambi +=1;
	jt-=1;
      }	
    }
    
  }
  
  return;
}


void DM_perm(int mfull, int nfull, int ncols, int rank,
	     double* norms,
	     double *a, int lda,
	     int *jpvt, 
	     double tau_,
	     double delta, 
	     int k_max,
	     int *idxI, int *nb_idxI,
	     int *workint,
	     double *workdouble,
	     wrapStruct *tmpStr){
  /*
    This function computes a column ordering based on deviation maximization
    input:
    m = rows
    n = columns
    norms = vector containing the norm of each column
    a = matrix
    lda = leading dimension = nb of rows
    tau_ = threshold on the norms
    thres = threshold on the cosines
    k_max = maximum number of candidate columns 
    idxI = at top: columns to be factorised, at bottom: columns to be excluded 
    nb_idxI = nb of columns selected by DM to be triangularized
    workint = integer workspace
    workdouble = double workspace

    **auxiliary memory**
    wrapstruct (tmpStr) of size ncols 
    int (idxtmp, mark) of size(k_max)+(ncols)
    double (as, cosmat ) of size (m*nc)+(nc*nc)
  */

  int idxj,i,j,nc, ni, startpos1, startpos2;
  double *as, *cosmat;
  int *idxtmp, *mark;
  double one = 1.0, zero = 0.0, maxval, cc;
  int m = mfull - rank + 1;
  double dummy = 0.0;

  //mark = auxiliary vector of size ncols
  // at the end mark[i]=1 iff the i-th column is selected by DM
  mark = workint;
  // at the beginning it must be a zero valued vector, norm_update takes care of itInitialize auxiliary arrays 
  if (0){/*This is done in norm_update for efficiency */
    for (i=0;i<ncols;i++){
      tmpStr[i].val = norms[i];
      tmpStr[i].idx = i;
      mark[i] = 0;
    }
  }

  //idxtmp = auxiliary vector;
  idxtmp = &workint[ncols];

  //as =  auxiliary vector of size m*k_max, will contain the matrix a rescaled by the norms 
  as = workdouble;
  //cosmat = auxiliary vector of size k_max*k_max, will contain the cosine 
  cosmat = &workdouble[m*k_max];

  
  // the structure tmpStr is used as wrapper to find the ordering that sorts the partial column norms with quicksort 
  // the correct initialization of tmpStr is carried out by norm_update
  // quicksort call
  qsort(tmpStr, ncols, sizeof(wrapStruct),cmpStruct);
  
  /* Find the number of candidates based on the partial column norm */
  nc = 0;  
  maxval = tau_*tmpStr[0].val; // tau_*(maximum partial column norm)
  while((nc<ncols) && (nc<k_max) && (tmpStr[nc].val > maxval))
    nc++;

  /* Initialize with the index of the maximum partial column norm*/
  *nb_idxI = 1;
  int sel = 1;
  idxI[0] = tmpStr[0].idx;
  mark[tmpStr[0].idx] = 1;
  // Initialize ordered list of indices
  idxtmp[0] = 0;
  i = 1;
  
  
  if (nc>1){
    /* Rescale candidate columns by the inverse of the norm*/
    
    for(j=0;j<nc;j++){
      cc = 1.0/tmpStr[j].val;
      idxj = tmpStr[j].idx; 
      startpos1 = j*m;
      startpos2 = idxj*lda+rank;
      for(i=0;i<m;i++) {
	as[startpos1+i] = cc*a[startpos2+i];
      }
    }

    /* Compute the cosine matrix matrix cosmat = as.T*as */
    if (1){
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		  nc, nc, m, one, as, m, as, m,
		  zero, cosmat, nc);
      
      
    }
    else{
      /* Does not work as expected */
      cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
		  nc, m,
		  one, as, m,
		  zero, cosmat, nc);
    }

    
    while(i<nc){
      // Compute the maximum value of the cosine of the angle between candidate and previously selected columns
      if (mark[tmpStr[i].idx] == 0){
	maxval = 0.0; 
	for(j=0;j<*nb_idxI;j++){
	  if(maxval < fabs(cosmat[i*nc+idxtmp[j]]))
	    maxval = fabs(cosmat[i*nc+idxtmp[j]]); 
	}
	if ((maxval < delta) && (*nb_idxI < k_max)){ // if the cosine in small enough -> angle large enough
	// then add i to selected columns
	  idxtmp[sel] = i;
	  idxI[sel] = tmpStr[i].idx;
	  
	  mark[tmpStr[i].idx] = 1;
	  sel++;
	  nb_idxI[0] +=1;
	}
      }
      i++;
    }
  }
  
  /* Permute selected columns at top left positions*/
  permute_marked(mfull, nfull, ncols,
		 &a[1], lda,
		 jpvt, norms,
		 mark,
		 idxI, *nb_idxI,
		 0);
  
  return;
}

int dgeqrdm_work(int matrix_layout,
		 int m, int n, double *a, int lda,
		 int *jpvt, double *tau,
		 int *ncols,
		 double *thres,
		 int nb){
  
  /*  Purpose */
  /*  ======= */

  /*  DGEQRDM computes a QR factorization with Deviation Maximization pivoting of a */
  /*  matrix A:  A*P = Q*R  using Level 3 BLAS. */

  /*  Arguments */
  /*  ========= */

  /*  matrix_layout (input) INT */
  /*          101 if row major */
  /*          102 if column major */

  /*  M       (input) INT */
  /*          The number of rows of the matrix A. M >= 0. */

  /*  N       (input) INT */
  /*          The number of columns of the matrix A.  N >= 0. */

  /*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N) */
  /*          On entry, the M-by-N matrix A. */
  /*          On exit, the upper triangle of the array contains the */
  /*          min(M,N)-by-N upper trapezoidal matrix R; the elements below */
  /*          the diagonal, together with the array TAU, represent the */
  /*          orthogonal matrix Q as a product of min(M,N) elementary */
  /*          reflectors. */

  /*  LDA     (input) INT */
  /*          The leading dimension of the array A. LDA >= max(1,M). */

  /*  JPVT    (input/output) INT array, dimension (N) */
  /*          On entry, if JPVT[J]!=0, the J-th column of A is permuted */
  /*          to the front of A*P (a leading column); if JPVT(J)=0, */
  /*          the J-th column of A is a free column. */
  /*          On exit, if JPVT[J]=K, then the J-th column of A*P was the */
  /*          the K-th column of A. */

  /*  TAU     (output) DOUBLE PRECISION array, dimension (min(M,N)) */
  /*          The scalar factors of the elementary reflectors. */

  /*  NCOLS   (input/output) INT array, dimension (N) */
  /*          On entry, if NCOLS[0] = 1, the algorithm terminates when */
  /*          the stopping criterion is satisfied.  */
  /*          On exit, NCOLS[I] returns the number of factorized columns  */
  /*          at i-th iteration.  */

  /*  THRES   (input) DOUBLE PRECISION array, dimension (2) */
  /*          Deviation Maximization parameters:   */
  /*          THRES[0] contains the value of 0=<delta<=1*/
  /*          THRES[1] contains the value of 0=<tau_<=1*/

  /*  NB      (input) INT */
  /*          Blocksize. Maximum numer of candidate columns (lated denoted as K_MAX). */

  
  /*  INFO    (output) INT */
  /*          = 0: successful exit. */
  /*          < 0: if INFO = -i, the i-th argument had an illegal value. */

  /*  Further Details */
  /*  =============== */

  /*  The matrix Q is represented as a product of elementary reflectors */

  /*     Q = H(1) H(2) . . . H(k), where k = min(m,n). */

  /*  Each H(i) has the form */

  /*     H(i) = I - tau * v * v' */

  /*  where tau is a real/complex scalar, and v is a real/complex vector */
  /*  with v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in */
  /*  A(i+1:m,i), and tau in TAU(i). */

  /*  Based on */
  /*  "Deviation Maximization for rank-revealing QR factorizations" */
  /*  M. Dessole, F. Marcuzzi */
  /*  Dipartimento di Matematica "Tullio Levi Civita" */
  /*  University of Padova, Italy */
  
  /*  ===================================================================== */

  /*  Local variables */
  int flag_exit1=0;
  int enable_stop =0;
  if (ncols[0] == 1){
    enable_stop=1;
  }

  double delta,  tau_;
  delta = thres[0];
  tau_ = thres[1];
  int threschoice = 1; /* 1: Triagularize a column if it is still a candidate, 0: Traingularize if its partial norm is larger than 1e-14*/ 
  double maxnrm = 0.0, maxnrmA = 0.0;
  int ii, jj;
  int a_dim1, a_offset, i__1, i__2, i__3;

  int j, jb, na, sm, sn, nx, fjb, fjb_cmp, nfxd, ldt;
  int minmn;
  int sminmn;

  double eps = LAPACKE_dlamch('e');
  double tol3z = sqrt(eps);
  


  /* Parameter adjustments */
  /* Convert 0-based arrays into 1-based */
  a_dim1 = lda;
  a_offset = 1 + a_dim1;
  a -= a_offset;
  --jpvt;
  --tau;


  /* Function Body */
  
  /*     Test input arguments */
  /*     ==================== */

  int info = 0;
  if ((matrix_layout != LAPACK_COL_MAJOR) && (matrix_layout !=  LAPACK_ROW_MAJOR)){
    info = -1;
  } else if (m <= 0) {
    info = -2;
  } else if (n <= 0) {
    info = -3;
  } else if (lda < max(1,m)) {
    info = -5;
  } else if ((delta < 0.0) || (delta > 1.0)){
    info = -9;
  } else if ((tau_ < 0.0) || (tau_ > 1.0)){
    info = -9;
  } else if (nb <= 0){
    info = -10;
  }


  if (info != 0) {
    i__1 = -(info);
    xerbla_("DGEQRDM", &i__1);
    return -1;
  }
  
  /*   Double check. */
  minmn = min(m,n);
  if (minmn == 0) {
    i__1 = -(info);
    xerbla_("DGEQRDM", &i__1);
    return -1;
  }

  
  /*     Move initial columns up front. */

  nfxd = 1;
  i__1 = n;
  for (j = 1; j <= i__1; ++j) {
    if (jpvt[j] != 0) {
      if (j != nfxd) {
	cblas_dswap(m, &a[j * a_dim1 + 1], c__1, &a[nfxd * a_dim1 + 1], c__1);
	jpvt[j] = jpvt[nfxd];
	jpvt[nfxd] = j;
      } else {
	jpvt[j] = j;
      }
      ++nfxd;
    } else {
      jpvt[j] = j;
    }
  }
  --nfxd; 

  /*     Factorize fixed columns */
  /*     ======================= */

  /*     Compute the QR factorization of fixed columns and update */
  /*     remaining columns. */

  if (nfxd > 0) {
    na = min(m,nfxd);
    info = LAPACKE_dgeqrf(matrix_layout, m, na, &a[a_offset], lda, &tau[1]);
    if (info !=0){
      printf("LAPACK dgeqrf failed, info =%d \n", info);
      return info; 
    }
    
    if (na < n) {
      i__1 = n - na;
      info = LAPACKE_dormqr(matrix_layout, 'L', 'T', m, i__1, na, &a[a_offset], lda,
		     &tau[1], &a[(na + 1) * a_dim1 + 1], lda);
      if (info !=0){
	printf("LAPACK dormqr failed, info =%d \n", info);
	return info;
      }
    }
  }

  /*     Factorize free columns */
  /*     ====================== */
    
  if (nfxd < minmn) {
    
    sm = m - nfxd;
    sn = n - nfxd;
    sminmn = minmn - nfxd;

    ldt = nb;

    /* Allocate auxiliary memory */
   
    double *work;
    work = (double*) malloc(2*n*sizeof(double)); 
    --work;
  
    int * workint;
    workint = (int*) malloc((2*n+nb)*sizeof(int));
    int nc_x_block; 
    int ncmax = nb, k_max = nb;
    int it = -1;
    double * workdouble;
    workdouble = (double*) malloc((m*ncmax + ncmax*ncmax+n)*sizeof(double)); // ha inglobato workd_geqr2
    
    wrapStruct * tmpStr;
    tmpStr = (wrapStruct *) malloc(n*sizeof(wrapStruct));
    

    /*        Initialize partial column norms. The first N elements of work */
    /*        store the exact column norms. */
    i__1 = n;
  
    for (j = nfxd + 1; j <= i__1; ++j) {
      work[j] = cblas_dnrm2(sm, &a[nfxd + 1 + j * a_dim1], c__1);
      if (work[j]>maxnrm){
	maxnrm = work[j];
      }
      work[n + j] = work[j];
      tmpStr[j-1].val = work[j];
      tmpStr[j-1].idx = j-1;
      workint[n+j-1] = 0;
    }
    maxnrmA =maxnrm; // for the termination criterion

    // initialize indices
    j = nfxd + 1;      // 1-base column index
    i__1 = m - j + 1; // rows left to process
    i__2 = j - 1;  // 0-base column index
    i__3 = n - j + 1; // columns left to process
    j = nfxd + 1;
    
    /*   Compute factorization: while loop. */  
    while (j <= minmn) { 
      i__1 = m - j + 1;
      i__2 = j - 1;
      i__3 = n - j + 1;
      it = it+1;

	
      /* Deviation maximization with pivoting */
      if (maxnrm<=5*eps)
	k_max = 1; // if the partial column norms are too small, then switch to standard pivoting
      else
	k_max = min(min(nb,i__1),i__3);
	
      DM_perm(m, n, i__3, j,
	      &work[j],
	      &a[j * a_dim1], a_dim1,
	      &jpvt[j],
	      tau_,
	      delta,
	      k_max,
	      workint,
	      &fjb, 
	      &workint[n], workdouble, tmpStr);
		
      /* Compute Householder reflectors */
      /* H(j), H(j+1), . . ., H(j+fjb-1) */
      fjb_cmp = fjb;       	
      info = dgeqr2_mia(&i__1, &fjb,  
			&a[j * a_dim1 + j], &lda,
			&tau[j], &workdouble[m*ncmax + ncmax*ncmax],
			threschoice, &tau_, &fjb_cmp );
      
      ncols[it] = fjb_cmp;  // number of columns reduced to triangular form
	
      if (info !=0){
	printf("LAPACK dgeqr2 failed, info =%d \n", info);
	goto exit_level_0;
      }
	
      /* Form the triangular factor of the block reflector */
      /* H = H(j) H(j+1) . . . H(j+fjb-1) */
      
      info =  LAPACKE_dlarft(matrix_layout, 'f', 'c',
			     i__1, fjb_cmp, &a[j * a_dim1 + j],
			     lda, &tau[j], workdouble, 
			     ldt);
      if (info !=0){
	printf("LAPACK dlarft failed, info =%d \n", info);
	goto exit_level_0;
      }
      
      /* Apply H**T to A(j:m,j+fjb_cmp:n) from the left */
 
      info =  LAPACKE_dlarfb_mia(matrix_layout,
				 'l', 't', 'f',
				 'c', i__1, i__3-fjb,
				 fjb_cmp,  &a[j * a_dim1 + j], lda,
				 workdouble, ldt,  &a[(j+fjb) * a_dim1 + j],
				 lda );
      if ( info !=0){
	printf("LAPACK dlarfb failed, info =%d \n", info);
	goto exit_level_0;
      }
        
      /* Partial column norm upodate */
      norm_update(i__1-fjb_cmp, i__3-fjb_cmp, fjb_cmp,
		  &a[(j+fjb_cmp)*lda+j], lda,
		  &work[j+fjb_cmp], &work[n+j+fjb_cmp], &maxnrm,
		  tmpStr, &workint[n],
		  tol3z);
	
      j += fjb_cmp;
      /* Stop criterion */
      if (enable_stop == 1){
	if ((maxnrm*sqrt((i__3-fjb_cmp)) <= eps*maxnrmA*(n)))
	  goto exit_level_0;
      }
	
	
    }

    
  
    /* Use unblocked code to factor the last or only block. */
    /* NOT used here!!!!!!!!! */ 
    i__1 = n - j + 1;
    i__2 = j - 1;

    if (j <= minmn && m-i__2 > 1 && flag_exit1 == 0) {

      for(ii=0;ii<i__1;ii++)
	workint[ii] = 0;
      
      info = LAPACKE_dgeqpf(matrix_layout, m-i__2, i__1,
			    &a[j * a_dim1 + j], lda,
			    workint, &tau[j]);
      if (info !=0){
	printf("LAPACK dlarfb failed, info =%d \n", info);
	goto exit_level_0;
      }
      if (j>1){
	apply_jpvt(1, i__2, i__1, &a[j*a_dim1+1], lda, workint, &jpvt[j], &work[j], &workint[n], workdouble );
      }else{
	for(ii=0;ii<i__1;ii++)
	  jpvt[ii+1] = workint[ii];
      
      }
    }
    
  exit_level_0:
    free(tmpStr);
    free(workint);
    free(workdouble);
    work++;
    free(work);
    if (info !=0){
      return info;
    }
  }


  return info;

  /*     End of DGEQRDM */


} 
