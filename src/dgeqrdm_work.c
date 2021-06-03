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
  
  if (0){
    for(jj = 0; jj<n; jj++){
      work[jj] = cblas_dnrm2(m, &a[jj*lda+fjb], c__1);
      
      //printf("work[%d] = %f, nrm = %f \n", jj, work[jj], nrm*nrm );
      
      if (*maxnrm < work[jj])
	*maxnrm = work[jj];
      if (1){
	tmpStr[jj].val = work[jj];
	tmpStr[jj].idx = jj;	
	workint[jj] = 0;
      }
    }
  }
  else{
    /* for(jj = 0; jj<n; jj++){ */
    /*   for(ii=0; ii<fjb;ii++){ */
    /*     work[jj] -= a[jj*lda+ii]*a[jj*lda+ii]; // unstable! */
    /*     if (*maxnrm < work[jj]) */
    /*   	*maxnrm = work[jj]; */
    /*   } */
    /*   //nrm = cblas_dnrm2(m, &a[jj*lda+fjb], c__1); */
    /*   //printf("work[%d] = %f, nrm = %f \n", jj, work[jj], nrm*nrm ); */
    /* } */
  
    for(jj = 0; jj<n; jj++){
      if (work[jj] != 0.){	
	d__1 = 0.0;
	for(ii=0; ii<fjb;ii++){
	  d__1 += a[jj*lda+ii]*a[jj*lda+ii];
	}
	temp = (sqrt(fabs(d__1))) / work[jj];
	/* temp = 1.0-temp/work[jj]; */
	/* if (etmp<=eps) */
	/*   temp = 0.0; */
	/* temp2 = 0.05*tmp*(work[jj]/workOLD[jj]); */
	
	/* if (temp2<=eps){ */
	/*   if (m>0){ */
	/*     work[jj] = cblas_dnrm2(m, &a[jj*lda+fjb], c__1); */
	/*     work[jj] = work[jj]*work[jj]; */
	/*     workOLD[jj] = work[jj]; */
	/*   }else{ */
	/*     work[jj] = 0.0; */
	/*     workOLD[jj] = 0.0; */
	/*   } */
	/* } */
	/* else */
	/*   work[jj] = work[jj]*tmp; */


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
	

	//      printf("workint[%d] = %d\t", jj, workint[jj]);
	/* nrm = cblas_dnrm2(m, &a[jj*lda+fjb], c__1); */
	/* if (fabs(nrm*nrm - work[jj])>1e-13) */
	/* 	printf("\n\n!!!\n\n"); */
	//printf("work[%d] = %f, nrm = %f \n", jj, work[jj], nrm*nrm );
	
	if (*maxnrm < work[jj])
	  *maxnrm = work[jj];
	
      }
      // questo dev'essere fatto anche per le colonne di norma nulla
      // percio' e' dopo l'if
      if (1){
	tmpStr[jj].val = work[jj];
	tmpStr[jj].idx = jj;	
	workint[jj] = 0;
      }
    }
    //  printf("\n");
  }
  
  
  return;

}


void apply_jpvt(int onebased, int m, int n, double *a, int lda, int *jpvt, int *jpvtdef, double *norms, int *mark, double *tmpcol ){
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
  // se gli indici sono in base 1, riporta in base 0
  // inizializza workspace mark[i] = 0 for all i
  if (onebased){
    for (ii=0;ii<n;ii++){
      //idx[ii] = ii;
      jpvt[ii] -= 1;
      mark[ii] = 0;
    }
  }
  else{
    for (ii=0;ii<n;ii++)
      mark[ii] = 0; 
  }
  
  // applica il pivoting jpvt
  // alle colonne di a
  // al vettore delle norme delle colonne
  // al vettore che contiene il pivoting ti tutta la matrice
  // si basa sul fatto che ogni permutazione puo' essere scritta come un prodotto di cicli disgiunti
  // e ogni ciclo puo' essere scritto come prodotto di trasposizioni
  // (trasposizione = cicli di lunghezza 2 =  scambio di due colonne)
  // questo per usare solo m doubles come memoria ausiliaria
  int nscambi = 0;
  marked = 0;
  while (marked<n){
    i = mark_argmin(mark,n); // individua le colonne non ancora permutate
    tmp = i;
    tmp2 = jpvtdef[i];
    tmpnorm = norms[i];
    // copio colonna i in tmpcol
    for (j=0;j<m;j++)
      tmpcol[j] = a[i*lda+j];
    while((jpvt[i]!=tmp) && (marked<n)){
      if (i!=jpvt[i]){
	// scambio la colonna i e jpvt[i]
	for (j=0;j<m;j++)
	  a[i*lda+j] = a[jpvt[i]*lda+j];// qui j e' indice riga
	jpvtdef[i] = jpvtdef[jpvt[i]];
	norms[i]   = norms[jpvt[i]];
	nscambi +=1;
	i = jpvt[i];
      }
      mark[i]    = 1;
      marked++;
    }
    // copio colonna i da tmpcol
    for (j=0;j<m;j++)
      a[i*lda+j] = tmpcol[j];
    jpvtdef[i] = tmp2;
    norms[i] = tmpnorm;
    mark[i] = 1;
    marked++;
    nscambi +=1;
  }
  printf("nscambi = %d\n", nscambi);

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


void DM(int m, int ncols,
	double* norms,
	double *a, int lda,
	double tolnrm,
	double thresnrm,
	double threscos_low, double threscos_high,
	int nb_idxMax, int ncMax,
	int *idxI, int *nb_idxI,
	int *nz,
	int *workint, double *workdouble, wrapStruct *tmpStr){
  /*
    This function computes a column ordering based on deviation maximization
    input:
    m = rows
    n = columns
    norms = vector containing the norm of each column
    a = matrix
    lda = leading dimension = nb of rows
    thresnrm = threshold on the norms
    thres = threshold on the cosines
    nb_idxMax = maximum number of columns selected by DM 
    ncMax = maximum number of candidate columns >= nb_idxMax
    idxI = ordering chosen by DM for the ncols columns of a
    nb_idxI = nb of columns selected by DM to be triangularized
    workint = integer workspace
    workdouble = integer workspace

    **auxiliary memory**
    ncols wrapstruct 
    ncols+nb_idxMax int (idxtmp e mark)
    (m*nc)+(nc*nc) double (as, cosmat )
  */

  int idxj,i,j,nc, ni;
  // wrapStruct *tmpStr;
  double *as, *cosmat;
  int *idxtmp, *mark;
  double one = 1.0, zero = 0.0, maxval, cc;
  *nz = 0;

  //mark = auxiliary vector of size ncols
  // at the end mark[i]=1 iff the i-th column is selected by DM
  mark = workint; // at the beginning it must be a zero valued vector, norm_update takes care of it 
  //idxtmp = auxiliary vector;
  idxtmp = &workint[ncols];

  //as =  auxiliary vector of size m*ncMax, will contain the matrix a rescaled by the norms 
  as = workdouble;
  //cosmat = auxiliary vector of size ncMax*ncMax, will contain the cosine 
  cosmat = &workdouble[m*ncMax];
  
  // the structure tmpStr is used as wrapper to find the ordering ORD that sorts the vector norm with quicksort 
  // the correct initialization of tmpStr is carried out by norm_update
  // quicksort call
 
  qsort(tmpStr, ncols, sizeof(wrapStruct),cmpStruct);
  
  /* i=ncols-1; */
  /* printf("i = %d\t", i); */
  /* printf("tmpStr[i].val = %f\n", tmpStr[i].val); */
  /* while(i>=0 && tmpStr[i].val<tolnrm){ */
  /*   *nz++; */
  /*   idxI[ncols-*nz] = tmpStr[i].idx; */
  /*   mark[tmpStr[i].idx] = 1; */
  /*   i--; */
  /* } */
  /* printf("fine while\n"); */
  //  printf("norma più grande: %.15f, norma più piccola: %.15f\n", tmpStr[0].val, tmpStr[ncols-1].val);
  
  // find the number of candidates based of the norm vales
  nc = 0;
  maxval = thresnrm*tmpStr[0].val;
  while((nc<ncols) && (nc<ncMax) && (tmpStr[nc].val > maxval))
    nc++;

  // set as = a(:,ORD)*diag(c), c_j = cc
  for(j=0;j<nc;j++){
    cc = 1.0/sqrt(tmpStr[j].val);
    idxj = tmpStr[j].idx; 
    for(i=0;i<m;i++)
      as[j*m+i] = cc*a[idxj*lda+i];
  }
  // construct the symmetric matrix cosmat = as.T*as
  if (0){
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		nc, nc, m, one, as, m, as, m,
		zero, cosmat, nc);
  }
  else{
    // ATTENZIONE: qui l'output non e' simmetrico
    // (perche'? bo, forse dsyrk aggiunna solo una marte specifica, Upper o Lower Triang)
    // Qui solo la parte Triangolare superiore e' corretta
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
		nc, m,
		one, as, m,
		zero, cosmat, nc);
  }

  
  // DM procedure: first add column with largest norm
  *nb_idxI = 1;
  idxI[0] = tmpStr[0].idx;
  mark[tmpStr[0].idx] = 1;

  // select nb_idxI<=nc columns to be triangularized
  // and save their indices in idxI
  idxtmp[0] = 0;
  i = 1;// index in ORD of current candidate column
  while(i<nc){
    // maximum value of the cosine of the angle between candidate and previously selected coluns
    if (mark[tmpStr[i].idx] == 0){
      maxval = 0.0;
      for(j=0;j<*nb_idxI;j++){
	if(maxval < fabs(cosmat[i*nc+idxtmp[j]]))
	  // IMPORTANTE: guardare sempre l'elemento [idxtmp[j],i], NON quello [i, idxtmp[j]]
	  maxval = fabs(cosmat[i*nc+idxtmp[j]]); 
      }
      // if the cosine in small enough -> angle large enough
      // then add i to selected columns
      if ((maxval < threscos_low) && (*nb_idxI < nb_idxMax)){
	idxtmp[*nb_idxI] = i;
	idxI[*nb_idxI] = tmpStr[i].idx;
	mark[tmpStr[i].idx] = 1;
	nb_idxI[0] +=1;
      }
      else if (fabs(maxval-1)<threscos_high){
	nz[0]++;
	idxI[ncols-*nz] = tmpStr[i].idx;
	mark[tmpStr[i].idx] = 1;
	//printf("%f\t", maxval);
      }
    }
    i++;
  }

 
  // contruct ordering for the remaining ncols-nb_idxI columns of a:
  j = 0; 
  for(i=(*nb_idxI); i<ncols-*nz;i++){
    // the unselected columns with index >= nb_idxI stay in space
    if (mark[i] == 0){
      idxI[i] = i;
    }
    else if (mark[i] == 1){
      // otherwise swap the selected column with an unselected column of index < nb_idxI
      while(mark[j]==1){
	if (j+1 == (*nb_idxI))
	  j = ncols-*nz;
	else
	  j++;
      }
      idxI[i] = j;
      if (j+1 == (*nb_idxI))
	j = ncols-*nz;
      else
	j++;
    }
  }
  
  //  printf("aggiunte = %d\n", (*nb_idxI));
  /* for (i = 0; i<(*nb_idxI); i++){ */
  /*   printf("%d\t ",idxI[i]); */
  /* } */
  //printf("eliminate = %d\n", *nz);
  /* for (i = ncols-*nz; i<ncols; i++){ */
  /*   printf("%d\t ",idxI[i]); */
  /* } */
  /* printf("\n"); */
   
  
  // free(tmpStr);
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

  /* printf("fjb = %d, nz = %d.\n", nb_idxI, nz); */
  /* for (i=0;i<ncols;i++) */
  /*   printf("%d  ", marked[i]); */
  /* printf("\n\n"); */
  
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
    
    /* printf("jc = %d, marked[jc] = %d, jb = %d, jt = %d\n", jc, marked[jc], jb, jt); */

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
  
  
  /* for (i=0;i<ncols;i++) */
  /*   printf("%d  ", marked[i]); */
  /* printf("\nnscambi = %d su len(idxI) = %d\n", nscambi, nb_idxI+nz ); */
  return;
}


void DM_perm(int mfull, int nfull, int ncols, int rank,
	     double* norms,
	     double *a, int lda,
	     int *jpvt, 
	     double tolnrm,
	     double thresnrm,
	     double threscos_low, double threscos_high,
	     int nb_idxMax, int ncMax,
	     int *idxI, int *nb_idxI,
	     int *nz,
	     int *workint, double *workdouble, wrapStruct *tmpStr){
  /*
    This function computes a column ordering based on deviation maximization
    input:
    m = rows
    n = columns
    norms = vector containing the norm of each column
    a = matrix
    lda = leading dimension = nb of rows
    thresnrm = threshold on the norms
    thres = threshold on the cosines
    nb_idxMax = maximum number of columns selected by DM 
    ncMax = maximum number of candidate columns >= nb_idxMax
    idxI = at top: columns to be factorised, at bottom: columns to be excluded 
    nb_idxI = nb of columns selected by DM to be triangularized
    workint = integer workspace
    workdouble = integer workspace

    **auxiliary memory**
    ncols wrapstruct 
    ncols+nb_idxMax int (idxtmp e mark)
    (m*nc)+(nc*nc) double (as, cosmat )
  */

  int idxj,i,j,nc, ni, startpos1, startpos2;
  // wrapStruct *tmpStr;
  double *as, *cosmat;
  int *idxtmp, *mark;
  double one = 1.0, zero = 0.0, maxval, cc;
  *nz = 0;
  int m = mfull - rank + 1;
  double dummy = 0.0;

  //mark = auxiliary vector of size ncols
  // at the end mark[i]=1 iff the i-th column is selected by DM
  mark = workint;
  // at the beginning it must be a zero valued vector, norm_update takes care of it 
  if (0){//non lo facciamo perche' abbiamo accorpato il for in norm_update
    for (i=0;i<ncols;i++){
      tmpStr[i].val = norms[i];
      tmpStr[i].idx = i;
      mark[i] = 0;
    }
  }

  //idxtmp = auxiliary vector;
  idxtmp = &workint[ncols];

  //as =  auxiliary vector of size m*ncMax, will contain the matrix a rescaled by the norms 
  as = workdouble;
  //cosmat = auxiliary vector of size ncMax*ncMax, will contain the cosine 
  cosmat = &workdouble[m*ncMax];

  /* for (i=0;i<ncols;i++) */
  /*   printf("%f\t", tmpStr[i].val); */
  /* printf("\n"); */
  
  // the structure tmpStr is used as wrapper to find the ordering ORD that sorts the vector norm with quicksort 
  // the correct initialization of tmpStr is carried out by norm_update
  // quicksort call
  qsort(tmpStr, ncols, sizeof(wrapStruct),cmpStruct);
  
  // find the number of candidates based of the norm vales
  nc = 0;  
  maxval = thresnrm*tmpStr[0].val;
  while((nc<ncols) && (nc<ncMax) && (tmpStr[nc].val > maxval))
    nc++;

  //printf("Starting for...\n");
  // set as = a(:,ORD)*diag(c), c_j = cc
  //printf("nc = %d\n",nc); fflush();

  // DM procedure: first add column with largest norm
  *nb_idxI = 1;
  int sel = 1;
  idxI[0] = tmpStr[0].idx;
  mark[tmpStr[0].idx] = 1;
  
  if (nc>1){
    for(j=0;j<nc;j++){
      cc = 1.0/tmpStr[j].val;
      idxj = tmpStr[j].idx; 
      startpos1 = j*m;
      startpos2 = idxj*lda+rank;
      for(i=0;i<m;i++) {
	as[startpos1+i] = cc*a[startpos2+i];
      }
    }
    
    //printf("Starting dsyrk...\n");
    // construct the symmetric matrix cosmat = as.T*as
    if (1){
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		  nc, nc, m, one, as, m, as, m,
		  zero, cosmat, nc);
      
      
    }
    else{
      // ATTENZIONE: qui l'output non e' simmetrico
      // (perche'? bo, forse dsyrk aggiunna solo una marte specifica, Upper o Lower Triang)
      // Qui solo la parte Triangolare superiore e' corretta
      cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
		  nc, m,
		  one, as, m,
		  zero, cosmat, nc);
    }
    
    //printf("ended syrk...\n");
    
    // DM_procedure
    // select nb_idxI<=nc columns to be triangularized
    // and save their indices in idxI
    idxtmp[0] = 0;
    i = 1;// index in ORD of current candidate column
    // printf("Starting while...\n");  
    while(i<nc){
      // maximum value of the cosine of the angle between candidate and previously selected coluns
      if (mark[tmpStr[i].idx] == 0){
	maxval = 0.0; 
	for(j=0;j<*nb_idxI;j++){
	  if(maxval < fabs(cosmat[i*nc+idxtmp[j]]))
	    // IMPORTANTE: guardare sempre l'elemento [idxtmp[j],i], NON quello [i, idxtmp[j]]
	    maxval = fabs(cosmat[i*nc+idxtmp[j]]); 
	}
	// if the cosine in small enough -> angle large enough
	// then add i to selected columns
	if ((maxval < threscos_low) && (*nb_idxI < nb_idxMax)){
	  idxtmp[sel] = i;
	  idxI[sel] = tmpStr[i].idx;
	  
	  mark[tmpStr[i].idx] = 1;
	  sel++;
	  nb_idxI[0] +=1;
	}
	else if (0) { // (fabs(maxval-1)<threscos_high){  // qui con "if (0) {" si evita di scartare colonne
	  nz[0]++;
	  idxI[sel] = tmpStr[i].idx;
	  mark[tmpStr[i].idx] = 2;
	  sel++;
	  //printf("%f\t", maxval);
	}
      }
      i++;
    }
  }
  
  permute_marked(mfull, nfull, ncols,
		 &a[1], lda,
		 jpvt, norms,
		 mark,
		 idxI, *nb_idxI,
		 *nz);
  // printf("end DM...\n");
  
  return;
}

void DM_rect(int m, int ncols,
      double* norms,
	     double *a, int lda,
	     double tolnrm,
	     double thresnrm,
	     double threscos_low, double threscos_high,
	     int nb_idxMax, int ncMax,
	     int *idxI, int *nb_idxI,
	     int *nz,
	     int *workint, double *workdouble, wrapStruct *tmpStr){
  /*
    This function computes a column ordering based on deviation maximization
    input:
    m = rows
    n = columns
    norms = vector containing the norm of each column
    a = matrix
    lda = leading dimension = nb of rows
    thresnrm = threshold on the norms
    thres = threshold on the cosines
    nb_idxMax = maximum number of columns selected by DM 
    ncMax = maximum number of candidate columns >= nb_idxMax
    idxI = ordering chosen by DM for the ncols columns of a
    nb_idxI = nb of columns selected by DM to be triangularized
    workint = integer workspace
    workdouble = integer workspace

    **auxiliary memory**
    ncols wrapstruct 
    ncols+nb_idxMax int (idxtmp e mark)
    (m*ncols)+(ncols*nc) double (as, cosmat )
  */

  int idxj,i,j,l,nc, ni;
  // wrapStruct *tmpStr;
  double *as, *cosmat;
  int ldcosmat = ncols;
  int *idxtmp, *mark;
  double one = 1.0, zero = 0.0, maxval, cc;
  *nz = 0;

  //mark = auxiliary vector of size ncols
  // at the end mark[i]=1 iff the i-th column is selected by DM
  mark = workint; // at the beginning it must be a zero valued vector, norm_update takes care of it 
  //idxtmp = auxiliary vector;
  idxtmp = &workint[ncols];

  //as =  auxiliary vector of size m*ncMax, will contain the matrix a rescaled by the norms 
  as = workdouble;
  //cosmat = auxiliary vector of size ncMax*ncMax, will contain the cosine 
  cosmat = &workdouble[m*ncols];
  
  // the structure tmpStr is used as wrapper to find the ordering ORD that sorts the vector norm with quicksort 
  // the correct initialization of tmpStr is carried out by norm_update
  // quicksort call
  qsort(tmpStr, ncols, sizeof(wrapStruct),cmpStruct);

  // aggiungo le colonne nulle in coda al nuovo ordinamento
  /* i=ncols-1; */
  /* while(i>=0 && tmpStr[i].val<tolnrm){ */
  /*   *nz++; */
  /*   idxI[ncols-*nz] = tmpStr[i].idx; */
  /*   mark[tmpStr[i].idx] = 1; */
  /*   i--; */
  /* } */
  
  // find the number of candidates based of the norm vales
  nc = 0;
  maxval = thresnrm*tmpStr[0].val;
  while((nc<ncols) && (nc<ncMax) && (tmpStr[nc].val > maxval))
    nc++;

  // set as = a(:,ORD)*diag(c), c_j = cc
  for(j=0;j<(ncols-*nz);j++){
    cc = 1.0/sqrt(tmpStr[j].val);
    idxj = tmpStr[j].idx; 
    for(i=0;i<m;i++)
      as[j*m+i] = cc*a[idxj*lda+i];
  }
  
  // construct the symmetric matrix cosmat = as.T*as
  if (1){
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		(ncols-*nz), nc, m, one, as, m, as, m,
		zero, cosmat, ldcosmat);
  }
  else{
    // ATTENZIONE: qui l'output non e' simmetrico
    // (perche'? bo, forse dsyrk aggiunna solo una marte specifica, Upper o Lower Triang)
    // Qui solo la parte Triangolare superiore e' corretta
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
		nc, m,
		one, as, m,
		zero, cosmat, nc);
  }

  
  // DM procedure: first add column with largest norm
  *nb_idxI = 1;
  idxI[0] = tmpStr[0].idx;
  mark[tmpStr[0].idx] = 1;

  // select nb_idxI<=nc columns to be triangularized
  // and save their indices in idxI
  idxtmp[0] = 0;
  i = 1;// index in ORD of current candidate column
  while(i<nc){
    // maximum value of the cosine of the angle between candidate and previously selected coluns
    maxval = 0.0;
    if (mark[tmpStr[i].idx] == 0) {
      for(j=0;j<*nb_idxI;j++){
	if(maxval < fabs(cosmat[i*ldcosmat+idxtmp[j]])){
	  // IMPORTANTE: guardare sempre l'elemento [idxtmp[j],i], NON quello [i, idxtmp[j]]
	  maxval = fabs(cosmat[i*ldcosmat+idxtmp[j]]);
	}
      }
      // if the cosine in small enough -> angle large enough
      // then add i to selected columns
      if ((maxval < threscos_low) && (*nb_idxI < nb_idxMax)){
	idxtmp[*nb_idxI] = i;
	idxI[*nb_idxI] = tmpStr[i].idx;
	mark[tmpStr[i].idx] = 1;
	nb_idxI[0] +=1;
	for (l=0;l<ncols;l++){
	  // controllo la colonna i-esima
	  if ((mark[tmpStr[l].idx] == 0) && (1-fabs(cosmat[i*ldcosmat+l])<threscos_high)) {
	    nz[0]++;
	    idxI[ncols-*nz] = tmpStr[l].idx;
	    mark[tmpStr[l].idx] = 1;
	  }
	}
      }
      else if (fabs(maxval-1)<threscos_high){
	nz[0]++;
	idxI[ncols-*nz] = tmpStr[i].idx;
	mark[tmpStr[i].idx] = 1;
	//printf("%f\t", maxval);
      }
    }
    i++;
  }

 
  // contruct ordering for the remaining ncols-nb_idxI columns of a:
  j = 0; 
  for(i=(*nb_idxI); i<ncols-*nz;i++){
    // the unselected columns with index >= nb_idxI stay in space
    if (mark[i] == 0){
      idxI[i] = i;
    }
    else if (mark[i] == 1){
      // otherwise swap the selected column with an unselected column of index < nb_idxI
      while(mark[j]==1){
	if (j+1 == (*nb_idxI))
	  j = ncols-*nz;
	else
	  j++;
      }
      idxI[i] = j;
      if (j+1 == (*nb_idxI))
	j = ncols-*nz;
      else
	j++;
    }
  }

  return;
}



void DM_block(int m, int ncols,
	      double* norms,
	      double *a, int lda,
	      double tolnrm,
	      double thresnrm,
	      double threscos_low, double threscos_high,
	      int nb_idxMax, int ncMax,
	      int *idxI, int *nb_idxI,
	      int *nz,
	      int *workint, double *workdouble, wrapStruct *tmpStr, int nblocks){
  /*
    This function computes a column ordering based on deviation maximization
    input:
    m = rows
    n = columns
    norms = vector containing the norm of each column
    a = matrix
    lda = leading dimension = nb of rows
    thresnrm = threshold on the norms
    thres = threshold on the cosines
    nb_idxMax = maximum number of columns selected by DM 
    ncMax = maximum number of candidate columns >= nb_idxMax
    idxI = ordering chosen by DM for the ncols columns of a
    nb_idxI = nb of columns selected by DM to be triangularized
    workint = integer workspace
    workdouble = integer workspace

    **auxiliary memory**
    ncols wrapstruct 
    ncols+nb_idxMax int (idxtmp e mark)
    (m*ncols)+(ncols*int(nc/nblocks+1)) double (as, cosmat )
  */

  int idxj,i,j,l,k,nc,ib,ni;
  // wrapStruct *tmpStr;
  double *as, *cosmat;
  int ldcosmat = ncols;
  int beg_block, end_block, nc_x_block;
  int *idxtmp, *mark;
  double one = 1.0, zero = 0.0, maxval, cc;
  *nz = 0;

  //mark = auxiliary vector of size ncols
  // at the end mark[i]=1 iff the i-th column is selected by DM
  mark = workint; // at the beginning it must be a zero valued vector, norm_update takes care of it 
  //idxtmp = auxiliary vector;
  idxtmp = &workint[ncols];

  //as =  auxiliary vector of size m*ncMax, will contain the matrix a rescaled by the norms 
  as = workdouble;
  //cosmat = auxiliary vector of size ncMax*ncMax, will contain the cosine 
  cosmat = &workdouble[m*ncols];
  
  // the structure tmpStr is used as wrapper to find the ordering ORD that sorts the vector norm with quicksort 
  // the correct initialization of tmpStr is carried out by norm_update
  // quicksort call
  qsort(tmpStr, ncols, sizeof(wrapStruct),cmpStruct);

  // aggiungo le colonne nulle in coda al nuovo ordinamento
  /* i=ncols-1; */
  /* while(i>=0 && tmpStr[i].val<tolnrm){ */
  /*   *nz++; */
  /*   idxI[ncols-*nz] = tmpStr[i].idx; */
  /*   mark[tmpStr[i].idx] = 1; */
  /*   i--; */
  /* } */
  
  // find the number of candidates based of the norm vales
  nc = 0;
  maxval = thresnrm*tmpStr[0].val;
  while((nc<ncols) && (nc<ncMax) && (tmpStr[nc].val > maxval))
    nc++;
  
  nc_x_block = floor(nc/nblocks+1);
    
  // set as = a(:,ORD)*diag(c), c_j = cc
  for(j=0;j<(ncols-*nz);j++){
    cc = 1.0/sqrt(tmpStr[j].val);
    idxj = tmpStr[j].idx; 
    for(i=0;i<m;i++)
      as[j*m+i] = cc*a[idxj*lda+i];
  }
  

  // DM procedure: first add column with largest norm
  *nb_idxI = 1;
  idxI[0] = tmpStr[0].idx;
  mark[tmpStr[0].idx] = 1;

  // select nb_idxI<=nc columns to be triangularized
  // and save their indices in idxI
  idxtmp[0] = 0;
  i = 1;// index in ORD of current candidate column
  for (k=0;k<nblocks;k++){
    beg_block = k*nc_x_block +1;
    end_block = min(beg_block + nc_x_block,nc);
    //printf("nc_x_block = %d, k =%d, end_block = %d beg_block = %d\n", nc_x_block, k,  end_block,beg_block);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		(ncols-*nz), end_block-beg_block, m, one, as, m, &as[beg_block*m], m,
		zero, cosmat, ldcosmat);
    
    for(i=beg_block;i<end_block;i++){
      // maximum value of the cosine of the angle between candidate and previously selected coluns
      maxval = 0.0;
      if (mark[tmpStr[i].idx] == 0) {
	ib = i-beg_block;
	for(j=0;j<*nb_idxI;j++){
	  if(maxval < fabs(cosmat[ib*ldcosmat+idxtmp[j]])){
	    // IMPORTANTE: guardare sempre l'elemento [idxtmp[j],i], NON quello [i, idxtmp[j]]
	    maxval = fabs(cosmat[ib*ldcosmat+idxtmp[j]]);
	  }
	}
	// if the cosine in small enough -> angle large enough
	// then add i to selected columns
	if ((maxval < threscos_low) && (*nb_idxI < nb_idxMax)){
	  idxtmp[*nb_idxI] = i;
	  idxI[*nb_idxI] = tmpStr[i].idx;
	  mark[tmpStr[i].idx] = 1;
	  nb_idxI[0] +=1;
	  for (l=0;l<ncols;l++){
	    // controllo la colonna i-esima
	    if ((mark[tmpStr[l].idx] == 0) && (1-fabs(cosmat[ib*ldcosmat+l])<threscos_high)) {
	      nz[0]++;
	      idxI[ncols-*nz] = tmpStr[l].idx;
	      mark[tmpStr[l].idx] = 1;
	    }
	  }
	}
	else if (fabs(maxval-1)<threscos_high){
	  nz[0]++;
	  idxI[ncols-*nz] = tmpStr[i].idx;
	  mark[tmpStr[i].idx] = 1;
	  //printf("%f\t", maxval);
	}
      }
    }
  }

 
  // contruct ordering for the remaining ncols-nb_idxI columns of a:
  j = 0; 
  for(i=(*nb_idxI); i<ncols-*nz;i++){
    // the unselected columns with index >= nb_idxI stay in space
    if (mark[i] == 0){
      idxI[i] = i;
    }
    else if (mark[i] == 1){
      // otherwise swap the selected column with an unselected column of index < nb_idxI
      while(mark[j]==1){
	if (j+1 == (*nb_idxI))
	  j = ncols-*nz;
	else
	  j++;
      }
      idxI[i] = j;
      if (j+1 == (*nb_idxI))
	j = ncols-*nz;
      else
	j++;
    }
  }

  return;
}

void DM_block_perm(int mfull, int ncols, int rank,
		   double* norms,
		   double *a, int lda,
		   int *jpvt,
		   double tolnrm,
		   double thresnrm,
		   double threscos_low, double threscos_high,
		   int nb_idxMax, int ncMax,
		   int *idxI, int *nb_idxI,
		   int *nz,
		   int *workint, double *workdouble, wrapStruct *tmpStr, int nblocks){
  /*
    This function computes a column ordering based on deviation maximization
    input:
    m = rows
    n = columns
    norms = vector containing the norm of each column
    a = matrix
    lda = leading dimension = nb of rows
    thresnrm = threshold on the norms
    thres = threshold on the cosines
    nb_idxMax = maximum number of columns selected by DM 
    ncMax = maximum number of candidate columns >= nb_idxMax
    idxI = ordering chosen by DM for the ncols columns of a
    nb_idxI = nb of columns selected by DM to be triangularized
    workint = integer workspace
    workdouble = integer workspace

    **auxiliary memory**
    ncols wrapstruct 
    ncols+nb_idxMax int (idxtmp e mark)
    (m*ncols)+(ncols*int(nc/nblocks+1)) double (as, cosmat )
  */

  int idxj,i,j,l,k,nc,ib,ni;
  // wrapStruct *tmpStr;
  double *as, *cosmat;
  int ldcosmat = ncols;
  int beg_block, end_block, nc_x_block;
  int *idxtmp, *mark;
  double one = 1.0, zero = 0.0, maxval, cc;
  int m = mfull - rank +1;
  *nz = 0;

  //mark = auxiliary vector of size ncols
  // at the end mark[i]=1 iff the i-th column is selected by DM
  mark = workint; // at the beginning it must be a zero valued vector, norm_update takes care of it 
  //idxtmp = auxiliary vector;
  idxtmp = &workint[ncols];

  //as =  auxiliary vector of size m*ncMax, will contain the matrix a rescaled by the norms 
  as = workdouble;
  //cosmat = auxiliary vector of size ncMax*ncMax, will contain the cosine 
  cosmat = &workdouble[m*ncols];
  
  // the structure tmpStr is used as wrapper to find the ordering ORD that sorts the vector norm with quicksort 
  // the correct initialization of tmpStr is carried out by norm_update
  // quicksort call
  qsort(tmpStr, ncols, sizeof(wrapStruct),cmpStruct);

  // aggiungo le colonne nulle in coda al nuovo ordinamento
  /* i=ncols-1; */
  /* while(i>=0 && tmpStr[i].val<tolnrm){ */
  /*   *nz++; */
  /*   idxI[ncols-*nz] = tmpStr[i].idx; */
  /*   mark[tmpStr[i].idx] = 1; */
  /*   i--; */
  /* } */
  
  // find the number of candidates based of the norm vales
  nc = 0;
  maxval = thresnrm*tmpStr[0].val;
  while((nc<ncols) && (nc<ncMax) && (tmpStr[nc].val > maxval))
    nc++;
  
  nc_x_block = floor(nc/nblocks+1);
    
  // set as = a(:,ORD)*diag(c), c_j = cc
  for(j=0;j<ncols;j++){
    cc = 1.0/sqrt(tmpStr[j].val);
    idxj = tmpStr[j].idx; 
    for(i=0;i<m;i++)
      as[j*m+i] = cc*a[idxj*lda+rank+i];
  }
  

  // DM procedure: first add column with largest norm
  int sel = 1;
  *nb_idxI = 1;
  idxI[0] = tmpStr[0].idx;
  mark[tmpStr[0].idx] = 1;

  // select nb_idxI<=nc columns to be triangularized
  // and save their indices in idxI
  idxtmp[0] = 0;
  i = 1;// index in ORD of current candidate column
  for (k=0;k<nblocks;k++){
    beg_block = k*nc_x_block +1;
    end_block = min(beg_block + nc_x_block,nc);
    //printf("nc_x_block = %d, k =%d, end_block = %d beg_block = %d\n", nc_x_block, k,  end_block,beg_block);
    //printf("parameter 4:  ncols = %d, beg_block = %d, end_block = %d\n",  ncols, beg_block,end_block);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		ncols, end_block-beg_block, m, one, as, m, &as[beg_block*m], m,
		zero, cosmat, ldcosmat);
    
    for(i=beg_block;i<end_block;i++){
      // maximum value of the cosine of the angle between candidate and previously selected coluns
      maxval = 0.0;
      if (mark[tmpStr[i].idx] == 0) {
	ib = i-beg_block;
	for(j=0;j<*nb_idxI;j++){
	  if(maxval < fabs(cosmat[ib*ldcosmat+idxtmp[j]])){
	    // IMPORTANTE: guardare sempre l'elemento [idxtmp[j],i], NON quello [i, idxtmp[j]]
	    maxval = fabs(cosmat[ib*ldcosmat+idxtmp[j]]);
	  }
	}
	// if the cosine in small enough -> angle large enough
	// then add i to selected columns
	if ((maxval < threscos_low) && (*nb_idxI < nb_idxMax)){
	  idxtmp[*nb_idxI] = i;
	  idxI[sel] = tmpStr[i].idx;
	  mark[tmpStr[i].idx] = 1;
	  sel++;
	  nb_idxI[0] +=1;
	  for (l=0;l<ncols;l++){
	    // controllo la colonna i-esima
	    if (0) { //  ((mark[tmpStr[l].idx] == 0) && (1-fabs(cosmat[ib*ldcosmat+l])<threscos_high)) {
	      nz[0]++;
	      idxI[sel] = tmpStr[l].idx;
	      mark[tmpStr[l].idx] = 2;
	      sel++;
	    }
	  }
	}
	else if (0) { //  (fabs(maxval-1)<threscos_high){
	  nz[0]++;
	  idxI[sel] = tmpStr[i].idx;
	  mark[tmpStr[i].idx] = 2;
	  sel++;
	  //printf("%f\t", maxval);
	}
      }
    }
  }

 
  permute_marked(mfull, 0, ncols,
		 &a[1], lda,
		 jpvt,  norms,
		 mark,
		 idxI, *nb_idxI,
		 *nz);
  return;
}


int dgeqrdm_work(int matrix_layout, int m, int n, double *a, int lda,
		 int *jpvt, double *tau,  int *ncols, double *thres,
 		 int DM_procedure)
{

  int flag_exit1=0,flag_exit2=0;
  int enable_stop =0;
  if (ncols[0] == 1){
    enable_stop=1;
  }

  double threscos_low, threscos_high, thresnrm;
  threscos_low = thres[0];
  threscos_high = thres[1];
  int threschoice;
  if (threscos_high <= 1e-15)
    threschoice = 0;
  else
    threschoice = 1;
  
  thresnrm = thres[2];
  double maxnrm = 0.0, maxnrmA = 0.0, tolnrm = 1e-10;

  int tmpint, ii, jj;
  /* System generated locals */
  int a_dim1, a_offset, i__1, i__2, i__3;

  /* Local variables */
  int j, jb, na, nb, sm, sn, nx, fjb, fjb_cmp, nfxd, ldt;
  int nbmin, minmn;
  int minws;
  int topbmn, sminmn;

  double eps = LAPACKE_dlamch('e');
  double tol3z = sqrt(eps);
  //printf("eps = %.16f\n", eps);
  
  /*  -- LAPACK routine (version 3.2) -- */
  /*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
  /*     November 2006 */

  /*     .. Scalar Arguments .. */
  /*     .. */
  /*     .. Array Arguments .. */
  /*     .. */

  /*  Purpose */
  /*  ======= */

  /*  DGEQP3 computes a QR factorization with column pivoting of a */
  /*  matrix A:  A*P = Q*R  using Level 3 BLAS. */

  /*  Arguments */
  /*  ========= */

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
  /*          On entry, if JPVT(J).ne.0, the J-th column of A is permuted */
  /*          to the front of A*P (a leading column); if JPVT(J)=0, */
  /*          the J-th column of A is a free column. */
  /*          On exit, if JPVT(J)=K, then the J-th column of A*P was the */
  /*          the K-th column of A. */

  /*  TAU     (output) DOUBLE PRECISION array, dimension (min(M,N)) */
  /*          The scalar factors of the elementary reflectors. */

  /*  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK)) */
  /*          On exit, if INFO=0, WORK(1) returns the optimal LWORK. */

  /*  LWORK   (input) INT */
  /*          The dimension of the array WORK. LWORK >= 3*N+1. */
  /*          For optimal performance LWORK >= 2*N+( N+1 )*NB, where NB */
  /*          is the optimal blocksize. */

  /*          If LWORK = -1, then a workspace query is assumed; the routine */
  /*          only calculates the optimal size of the WORK array, returns */
  /*          this value as the first entry of the WORK array, and no error */
  /*          message related to LWORK is issued by XERBLA. */

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

  /*  Based on contributions by */
  /*    G. Quintana-Orti, Depto. de Informatica, Universidad Jaime I, Spain */
  /*    X. Sun, Computer Science Dept., Duke University, USA */

  /*  ===================================================================== */

  /*     .. Parameters .. */
  /*     .. */
  /*     .. Local Scalars .. */
  /*     .. */
  /*     .. External Subroutines .. */
  /*     .. */
  /*     .. External Functions .. */
  /*     .. */
  /*     .. Intrinsic Functions .. */
  /*     .. */
  /*     .. Executable Statements .. */

  /*     Test input arguments */
  /*     ==================== */



  /* Parameter adjustments */
  /* quest'operazione converte */ 
  /* a da array 0-base indexed in array 1-base indexed*/
  a_dim1 = lda;
  a_offset = 1 + a_dim1;
  a -= a_offset;
  --jpvt;
  --tau;

  
  /* Function Body */
  int info = 0;
  
  if (m <= 0) {
    info = -2;
  } else if (n <= 0) {
    info = -3;
  } else if (lda < max(1,m)) {
    info = -5;
  } else if ((matrix_layout != LAPACK_COL_MAJOR) && (matrix_layout !=  LAPACK_ROW_MAJOR)){
    info = -1;
  }


  if (info != 0) {
    i__1 = -(info);
    xerbla_("DGEDMQR", &i__1);
    printf("returning info!=0\n");
    return -1;
  }

  
  /*     Quick return if possible. */
  minmn = min(m,n);
  if (minmn == 0) {
    printf("returning minmn==0\n");
    return -1;
  }


  
  /*     Move initial columns up front. */

  nfxd = 1;
  i__1 = n;
  for (j = 1; j <= i__1; ++j) {
    if (jpvt[j] != 0) {
      if (j != nfxd) {
	//void cblas_dswap(const int N, double * x, const int incx, double * y, const int incy)
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
  --nfxd; //nfxd>=0 e' il numero di colonne spostate dal ciclo qui sopra


  /*     Factorize fixed columns */
  /*     ======================= */

  /*     Compute the QR factorization of fixed columns and update */
  /*     remaining columns. */

  if (nfxd > 0) {
    na = min(m,nfxd);
    /* CC      CALL DGEQR2( M, NA, A, LDA, TAU, WORK, INFO ) */
    /* dgeqrf_(m, &na, &a[a_offset], lda, &tau[1], &work[1], lwork, info); */
    /* lapack_int LAPACKE_dgeqrf( int matrix_layout, lapack_int m, lapack_int n, */
    /*                        double* a, lapack_int lda, double* tau );  */

    info = LAPACKE_dgeqrf(matrix_layout, m, na, &a[a_offset], lda, &tau[1]);
    if (info !=0){
      printf("LAPACK dgeqrf failed, info =%d \n", info);
      return info; 
    }
    
    if (na < n) {
      /* CC         CALL DORM2R( 'Left', 'Transpose', M, N-NA, NA, A, LDA, */
      /* CC  $                   TAU, A( 1, NA+1 ), LDA, WORK, INFO ) */
      i__1 = n - na;
      // update trailing matrix
      /* dormqr_("Left", "Transpose", m, &i__1, &na, &a[a_offset], lda, & */
      /* 	      tau[1], &a[(na + 1) * a_dim1 + 1], lda, &work[1], lwork,  */
      /* 	      info); */
      /* lapack_int LAPACKE_dormqr( int matrix_layout, char side, char trans, */
      /*                      lapack_int m, lapack_int n, lapack_int k, */
      /*                      const double* a, lapack_int lda, const double* tau, */
      /*                      double* c, lapack_int ldc ); */
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

    /*        Determine the block size. */
    nb = 64; //64; //32;
      
    ldt = nb;
    nbmin = 2;
    nx = 0; //nx determina quando passare dal codice blas3 a quello blas2
    

    /* Allocate auxiliary memory */
    //double * t; // triangular factor for compact WY representation of Householder QR
    //t = (double*) malloc((nb+1)*(nb+1)*sizeof(double));


    double *work;
    work = (double*) malloc(2*n*sizeof(double)); 
    --work;
  
    int * workint;
    workint = (int*) malloc((2*n+nb)*sizeof(int));
    double * workdouble;
    int nc_x_block; 
    int nz  = 0, nzold = 0, ncmax = nb, kdm = nb;
    int it = -1;
    if ( DM_procedure == 1 ){
      // DM rettangolare
      workdouble = (double*) malloc((m*n + n*ncmax)*sizeof(double)); 
    }else if ( DM_procedure >= 2 ){
      // DM rettangolare a blocchi
      // DM_procedure = nb_blocks
      nc_x_block = floor(ncmax/DM_procedure+1);
      workdouble = (double*) malloc((m*n + n*nc_x_block)*sizeof(double)); 
    }else{
      workdouble = (double*) malloc((m*ncmax + ncmax*ncmax+n)*sizeof(double)); // ha inglobato workd_geqr2
    }
    /* double* workd_geqr2; */
    /* workd_geqr2 = (double*) malloc((2*n+n*n)*sizeof(double));  */
    
    wrapStruct * tmpStr;
    tmpStr = (wrapStruct *) malloc(n*sizeof(wrapStruct));

    

   
    /*        Initialize partial column norms. The first N elements of work */
    /*        store the exact column norms. */
    i__1 = n;
  
    for (j = nfxd + 1; j <= i__1; ++j) {
      //work[j] = dnrm2_(&sm, &a[nfxd + 1 + j * a_dim1], &c__1);
      //double cblas_dnrm2(const int N, const double * x, const int incx)
      work[j] = cblas_dnrm2(sm, &a[nfxd + 1 + j * a_dim1], c__1);
      if (work[j]>maxnrm){
	maxnrm = work[j];
      }
      work[n + j] = work[j];
      tmpStr[j-1].val = work[j];
      tmpStr[j-1].idx = j-1;
      workint[n+j-1] = 0;
      //printf("%f\t", work[j]);
    }
    maxnrmA =maxnrm;//*sqrt((double)n);

    // MIE MODIFICHE 
    //sminmn = 33;
    nx = 0;
    j = nfxd;
    while(j+nb <= sminmn){
      nx += nb;
     j+=nb;
    }
    nx = minmn - nx;

    j = nfxd + 1;
      
    i__1 = m - j + 1;
    i__2 = j - 1;
    i__3 = n - j + 1;

    //stop MIE MODIFICHE
    if (1) {//(nb >= nbmin && nb <= sminmn && nx < sminmn)
      /*           Use blocked code initially. */
      j = nfxd + 1; //indice colonna iniziale
      /*           Compute factorization: while loop. */
      
      topbmn = minmn - nx;
      while (j <= minmn) { //  ( (minmn - j + 1) >= nb){// topbmn
	//printf("j=%d\n",j);
	i__1 = m - j + 1;
	i__2 = j - 1;
	i__3 = n - j + 1;
	it = it+1;

	
	//	printf("chiamo DM\n");
	if (1){
	  // chiamata a DM dove fjb e' il numero di colonne processate nell'iterazione corrente
	  
	  /* DM(int m, int ncols, */
	  /* double* norms, */
	  /* double *a, int lda, */
	  /* double tolnrm, */
	  /* double thresnrm, */
	  /* double threscos_low, double threscos_high, */
	  /* int nb_idxMax, int ncMax, */
	  /* int *idxI, */
	  /* int *nb_idxI, int *nz, */
	  /* int *workint, double *workdouble) */

	  // limitiamo ncmax perche' non ha senso prendere piu' colonne eccedendo m o n
	  // in quanto rango <= min(m,n)
	  if (maxnrm<=5*eps)
	    kdm = 1;
	  else
	    kdm = min(min(nb,i__1),i__3);
	  
	  nzold = nz;
	  //printf("kdm = %d\n", kdm);
	  //printf("Starting DM\n");
	  if  ( DM_procedure == 1 ){
	    DM_rect(i__1, i__3 - nzold,
		    &work[j],
		    &a[j * a_dim1 + j], a_dim1,
		    tolnrm, thresnrm,
		    threscos_low, threscos_high,
		    kdm, ncmax,
		    workint,
		    &fjb, &nz,
		    &workint[n], workdouble, tmpStr);
	    /* permute columns */
	    //apply_jpvt(int onebased, int m, int n, double *a, int lda, int *jpvt, int *jpvtdef, double *norms )
	    apply_jpvt(0, m, i__3 - nzold,
		       &a[j*lda+1], lda,
		       workint, &jpvt[j], &work[j], &workint[n], workdouble );
	    
	  }else if (DM_procedure>=2){
	    if (0){
	      DM_block(i__1, i__3 - nzold,
		       &work[j],
		       &a[j * a_dim1 + j], a_dim1,
		       tolnrm, thresnrm,
		       threscos_low, threscos_high,
		       kdm, ncmax,
		       workint,
		       &fjb, &nz,
		       &workint[n], workdouble, tmpStr,DM_procedure);
	      /* permute columns */
	      //apply_jpvt(int onebased, int m, int n, double *a, int lda, int *jpvt, int *jpvtdef, double *norms )
	      apply_jpvt(0, m, i__3 - nzold,
			 &a[j*lda+1], lda,
			 workint, &jpvt[j], &work[j], &workint[n], workdouble );
	    }else{
	      DM_block_perm(m, i__3 - nzold, j,
			    &work[j],
			    &a[j * a_dim1], a_dim1,
			    &jpvt[j],
			    tolnrm, thresnrm,
			    threscos_low, threscos_high,
			    kdm, ncmax,
			    workint,
			    &fjb, &nz,
			    &workint[n], workdouble, tmpStr,DM_procedure);
	     }
	  }else {
	    if (0){
	      DM(i__1, i__3 - nzold,
		 &work[j],
		 &a[j * a_dim1 + j], a_dim1,
		 tolnrm, thresnrm,
		 threscos_low, threscos_high,
		 kdm, ncmax,
		 workint,
		 &fjb, &nz,
		 &workint[n], workdouble, tmpStr);
	      /* permute columns */
	      //apply_jpvt(int onebased, int m, int n, double *a, int lda, int *jpvt, int *jpvtdef, double *norms )
	      apply_jpvt(0, m, i__3 - nzold,
			 &a[j*lda+1], lda,
			 workint, &jpvt[j], &work[j], &workint[n], workdouble );
	    }else{
	      DM_perm(m, n, i__3 - nzold, j,
		      &work[j],
		      &a[j * a_dim1], a_dim1,
		      &jpvt[j],
		      tolnrm, thresnrm,
		      threscos_low, threscos_high,
		      kdm, ncmax,
		      workint,
		      &fjb, &nz,
		      &workint[n], workdouble, tmpStr);
	      
	    }
	  }
	  //printf("it = %d, fjb = %d, nz = %d, ztot = %d, rank = %d, ncmax = %d \n", it,fjb, nz, nz+nzold, j+fjb, ncmax);

	  nz += nzold;
	}
	else
	  fjb = nb;
	//	printf("chiamo dgeqr2_mia\n");
	fjb_cmp = fjb; //  in uscita da sgeqr2 conterra' le colonne effettivamente triangolarizzate
	

	
	info = dgeqr2_mia(&i__1, &fjb,  
			  &a[j * a_dim1 + j], &lda,
			  &tau[j], &workdouble[m*ncmax + ncmax*ncmax],
			  threschoice, &thresnrm, &fjb_cmp );

	ncols[it] = fjb_cmp;	
	//printf("j = %d, fjb = %d, fattorizzate: %d\n", j, fjb, fjb_cmp);
	
	if (info !=0){
	  printf("LAPACK dgeqr2 failed, info =%d \n", info);
	  goto exit_level_0;
	}
	/* *              Form the triangular factor of the block reflector */
	/* *              H = H(j) H(j+1) . . . H(j+fjb-1) */
	// valutare se sostituire con LAPACKE_dlarft_work
	//printf("chiamo dlartf\n");
	
	info =  LAPACKE_dlarft(matrix_layout, 'f', 'c',
			       i__1, fjb_cmp, &a[j * a_dim1 + j],
			       lda, &tau[j], workdouble, //t
			       ldt);
	if (info !=0){
	  printf("LAPACK dlarft failed, info =%d \n", info);
	  goto exit_level_0;
	}
	
	/* *              Apply H**T to A(i:m,i+ib:n) from the left */
	/* LAPACKE_dlarfb( int matrix_layout, char side, char trans, char direct, */
	/*                    char storev, lapack_int m, lapack_int n, */
	/*                    lapack_int k, const double* v, lapack_int ldv, */
	/*                    const double* t, lapack_int ldt, double* c, */
	/*                    lapack_int ldc ) */
	// CALL DLARFB( 'Left', 'Transpose', 'Forward',
	//                'Columnwise', M-I+1, N-I-IB+1, IB,
	//                A( I, I ), LDA, WORK, LDWORK, A( I, I+IB ),
	//                LDA, WORK( IB+1 ), LDWORK )
	//printf("chiamo dlarfb\n");



	/* lapack_int LAPACKE_dlarfb( int matrix_layout, char side, char trans, char direct, */
	/* 			   char storev, lapack_int m, lapack_int n, */
	/* 			   lapack_int k, const double* v, lapack_int ldv, */
	/* 			   const double* t, lapack_int ldt, double* c, */
	/* 			   lapack_int ldc ) */

	/*   if( LAPACKE_dge_nancheck( matrix_layout, i__1, k, */
        /*                               &v[k*lrv], ldv ) ) */

	info =  LAPACKE_dlarfb_mia(matrix_layout,
				   'l', 't', 'f',
				   'c', i__1, i__3-fjb-nz,
				   fjb_cmp,  &a[j * a_dim1 + j], lda,
				   workdouble, ldt,  &a[(j+fjb) * a_dim1 + j],
				   lda );
	if ( info !=0){
	  printf("LAPACK dlarfb failed, info =%d \n", info);
	  goto exit_level_0;
	}
        
	// aggiorna le norme
	norm_update(i__1-fjb_cmp, i__3-fjb_cmp-nz, fjb_cmp,
		    &a[(j+fjb_cmp)*lda+j], lda,
		    &work[j+fjb_cmp], &work[n+j+fjb_cmp], &maxnrm,
		    tmpStr, &workint[n],
		    tol3z);
	/* printf("diagonal values j = %d, j+fjb = %d:\n", j, j+fjb); */
	/* for (ii=j;ii<j+fjb;ii++) */
	/*   printf("%f\t",a[ii * a_dim1 + ii]); */
	/* printf("\n"); */
	/* printf("jpvt:\n"); */
	/* for (ii=1;ii<=n;ii++) */
	/*   printf("%d\t",jpvt[ii]); */
	/* printf("\n"); */
	
	j += fjb_cmp;
	//printf("j = %d, fjb_cmp = %d\n", j, fjb_cmp);
	/* if (flag_exit2 == 0){ */
	/*   if ((maxnrm*sqrt((i__3-fjb_cmp)) <= eps*maxnrmA*sqrt(n))) { */
	/*     // NB: maxnrmA=sqrt(n)*max||a_i|| */
	/*     flag_exit2 = 1; */
	/*     ncols[n] = j-1; */
	/*   }  */
	/* } */
	if (flag_exit1 == 0){
      if (0) {
          printf("j = %d\n",j);
          printf("maxnrm = %g\n",maxnrm);
          printf("maxnrm*sqrt((i__3-fjb_cmp)) = %g\n",maxnrm*sqrt((i__3-fjb_cmp)));
          printf("eps*maxnrmA*(n) = %g\n",eps*maxnrmA*(n));
      }
	  if ((maxnrm*sqrt((i__3-fjb_cmp)) <= eps*maxnrmA*(n))) {
	    // NB: maxnrmA=sqrt(n)*max||a_i||
	    flag_exit1 = 1;
	    ncols[n+1] = j-1;
	    if (enable_stop == 1){
	      goto exit_level_0;
	    }
	  }
	}
      }
    } else {
      j = nfxd + 1;
    }
    
  
    /*        Use unblocked code to factor the last or only block. */
    //*ncols = j-1; // usato per comunicare quante colonne sono state processate in questa prima fase

      
    i__1 = n - j - nz + 1;
    i__2 = j - 1;

    if (j <= minmn && m-i__2 > 1 && flag_exit1 == 0) {

      printf("Chiamo dgeqpf x ultimo blocco j %d minmn %d \n", j, minmn);
      // il wrapper lapacke a dlaqp2 non c'e'. possiamo usare dgeqpf. la differenza e' che dlaqp2 usava le norme
      // precedentemente calcolate in work, mentre dgeqpf si alloca memoria aggiuntiva per ricalcolarsele da capo
      /* dlaqp2_(m, &i__1, &i__2, &a[j * a_dim1 + 1], lda, &jpvt[j], &tau[ */
      /*         j], &work[j], &work[n + j], &work[(*n << 1) + 1]); */
      /* dlaqp2_(integer *m, integer *n, integer *offset,  */
      /* 	 doublereal *a, integer *lda, integer *jpvt, doublereal *tau,  */
      /* 	 doublereal *vn1, doublereal *vn2, doublereal *work) */
      /* LAPACKE_dgeqpf(matrix_layout, m, n, */
      /* 		double* a, lapack_int lda, lapack_int* jpvt, */
      /* 	        double* tau ) */
      
      

      //prova
      for(ii=0;ii<i__1;ii++)
	workint[ii] = 0;
      
      info = LAPACKE_dgeqpf(matrix_layout, m-i__2, i__1,
			    &a[j * a_dim1 + j], lda,
			    workint, &tau[j]);//intwork
      if (info !=0){
	printf("LAPACK dlarfb failed, info =%d \n", info);
	goto exit_level_0;
      }
      if (j>1){
	//apply_jpvt(int onebased, int m, int n, double *a, int lda, int *jpvt, int *jpvtdef, double *norms )  
	apply_jpvt(1, i__2, i__1, &a[j*a_dim1+1], lda, workint, &jpvt[j], &work[j], &workint[n], workdouble ); //intwork
      }else{
	for(ii=0;ii<i__1;ii++)
	  jpvt[ii+1] = workint[ii];
      
      }
    }
    //#endif
    //free(t);
  exit_level_0:
    free(tmpStr);
    free(workint);
    free(workdouble);
    //free(workd_geqr2);
    work++;
    free(work);
    if (info !=0){
      return info;
    }
  }


  return info;

  /*     End of DGEQP3 */


} /* dgeqp3_ */
