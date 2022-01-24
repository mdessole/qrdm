import numpy as np
import QRDM
import math


def rank_from_sv(s, tol = 5.e-9):
    ''' it computes numerical rank up to the tolerance tol
    from singular values or
    the diagonal values of the R factor od a RRQR.
    It also works with unordered values'''

    null_sv = np.where(np.absolute(s)<tol)[0]

    if len(null_sv>0):
        return np.amin(null_sv)+1
    else:
        return len(s)

    
def checkQR(matrix_layout, m, n, A, M, tau, jpvt, k = 0, check_err = True, verbose = False):
    '''
    Check accuracy of QR decomposition    
    Given: 
    M = Q| R_11 R_12 |
         |  0   R_22 |
    Returns:
    R  = | R_11 R_12 |
         |  0   R_22 |
    where R_11 is upper triagular of order k
    ------------------------------
    
    Outputs:
    Q factor
    R factor of the QR factorization
    if check_err == True:
    err absolute error ||A[:,jpvt] - Q*R||
    errQ absolute error ||I - Q*Q.T||
    
    Inputs:
    A original matrix
    M, tau contains the QR decomposizion of a matrix m x n
    jpvt column ordering
    k order of the R_11 factor
    check_err if True, absolute errors are computed
    verbose if True, absolute errors are printed
    '''
    
    if (m >= n):
        mm = m
        nn = n
    elif (m < n):
        mm = m
        nn = m
    #end
    
        
    Q = np.eye(m)
    min_mn = min(mm,nn)
    if (k == 0) or (k > min_mn):
        k = min_mn
    #end
    lda = m
    ldc = m
    
    pvt = jpvt-1

    out = QRDM.DORMQR(matrix_layout, mm,nn,k, A,lda,tau, Q, ldc)

    if (matrix_layout == 101):
        if k == n:
            R = np.triu(A)
        else:
            R = np.zeros_like(A)
            R[:,:k] = np.triu(A[:,:k])
            R[:,k:] = A[:,k:].copy()
        #endif
    elif (matrix_layout == 102):
        if k == n:
            R = np.triu(A.T)
        else:
            R = np.zeros_like(A.T)
            R[:,:k] = np.triu(A[:k,:].T)
            R[:,k:] = A[k:,:].T
        #endif
        M = M.T
        Q = Q.T
    #endif
    errQ = 0
    err = 0

    if check_err:
        errQ = np.linalg.norm(np.eye(m) - Q.T@Q)
        if verbose:
            print("||Q.T*Q - I|| = ", errQ)
        if k<min_mn:
            R_22 = R[k:,k:]
            if verbose:
                print("||R_22||_2 = ", np.linalg.norm(R_22))
        #endif
        err = 0
        err = np.linalg.norm(M[:,pvt]- Q@R)
        if verbose:
            print("||A[:,p] - Q*R|| = ",err)
        #end
    return Q,R,err,errQ

def getR(matrix_layout, m, n, A, k = 0):
    
    '''
    R factor of the QR factorization
    Given:
    A = Q| R_11 R_12 |
         |  0   R_22 |
    Returns:
    R  = | R_11 R_12 |
         |  0   R_22 |
    where R_11 is upper triagular of order k
    ------------------------------
    Inputs:
    M contains the QR decomposizion of a matrix m x n
    k <= min(m,n)
    '''
    
    if (m >= n):
        mm = m
        nn = n
    elif (m < n):
        mm = m
        nn = m
    #end
    min_mn = min(mm,nn) 
    if (k == 0) or (k>min_mn):
        k = min_mn
    #end
  
    if (matrix_layout == 101):
        if k == n:
            R = np.triu(A)
        else:
            R = np.zeros_like(A)
            R[:,:k] = np.triu(A[:,:k])
            R[:,k:] = A[:,k:].copy()
        #endif
    elif (matrix_layout == 102):
        if k == n:
            R = np.triu(A.T)
        else:
            R = np.zeros_like(A.T)
            R[:,:k] = np.triu(A[:k,:].T)
            R[:,k:] = A[k:,:].T
        #endif
    #endif
    
    return R

def low_rank(matrix_layout, m, n, M, A, tau, jpvt, k ):
    '''
    Rank-k approximation given by the RRQR factorization
    Given:
    A Pi = Q| R_11 R_12 |
            |  0   R_22 |
    Returns:
    A_k  = Q| R_11 R_12 |
            |  0    0   |
    ------------------------------
    Inputs:
    M, tau is the QR decomposizion 
    A is the transposed matrix
    '''
    ## Definire M, N e K
    if (m >= n):
        mm = m
        nn = n
    elif (m < n):
        mm = m
        nn = m
    #end
    min_mn = min(mm,nn) 
    if (k == 0) or (k>min_mn):
        k = min_mn
    #end

    if (matrix_layout == 102):
        if k == n:
            R = np.tril(M)
        else:
            R = np.zeros_like(M)
            R[:k,:] = np.tril(M[:k,:])
            R[k:,:k] = M[k:,:k]
        #endif
    #endif
    
    Alow = R.copy()
        
    lda = m
    ldc = m
    
    out = QRDM.DORMQR(matrix_layout, m,n,k, M,lda,tau, Alow, ldc)
    
    return Alow.T

