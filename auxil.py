import numpy as np
import QRDM
import QR
import math

def dormqr():
    k=0
    Q = np.eye(m)
    if k == 0:
        k = min(m,n)
    lda = m
    ldc = m
    pvt = jpvt-1
    lenZ = 0

    if (matrix_layout == 101):
        if k == n:
            R = np.triu(A)
        else:
            R = np.zeros_like(A)
            R[:,:k] = np.triu(A[:,:k])
            R[:,k:] = A[:,k:].copy()
        #endif
    elif (matrix_layout == 102):
        if k == min(m,n):
            R = np.triu(A.T)
            H = np.tril(A.T,k=-1)[:,:min(k,n)]
        else:
            R = np.zeros_like(A.T)
            R[:,:k] = np.triu(A[:k,:].T)
            H = np.tril(A[:k,:].T,k=-1)
            R[:,k:] = A[k:,:].T
        #endif
        M = M.T
        Q = Q.T
    #endif

    for j in range(k):
        v = np.reshape(H[j:,j],(m-j,1))
        v[0] = 1.0
        beta = tau[j]
        Qj = np.eye(m)
        Qj[j:,j:] -= beta*v@v.T
        Q= Q@Qj
    #end

    print('R.shape, Q.shape, M.shape = ', R.shape, Q.shape, M.shape)
    print("||Q.T*Q - I|| = ", np.linalg.norm(np.eye(m) - Q@Q.T))
    print("||A[:,p] - Q*R|| = ", np.linalg.norm(M[:,pvt[:n-lenZ]]- Q@(R[:,:n-lenZ])))


    return

def rank_from_sv(s, tol = 5.e-9):
    # it computes numerical rank from singular values or
    # the diagonal values of a triangular matrix.
    # it also works with unordered values
    null_sv = np.where(np.absolute(s)<tol)[0]

    if len(null_sv>0):
        return np.amin(null_sv)
    else:
        return len(s)

def checkQ(matrix_layout, m, n, A, M, tau, jpvt, verbose = False, k = 0, check_err = True):
    ## Definire M, N e K
    if (m >= n):
        mm = m
        nn = n
    elif (m < n):
        mm = m
        nn = m
    #end
    
        
    Q = np.eye(m)
    if k == 0:
        k = min(mm,nn)
    lda = m
    ldc = m
    
    pvt = jpvt-1
    #print("chiamo dormqr m, n,k = ", mm,nn,k)

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
        if k<min(m,n):
            R_22 = R[k:,k:]
            if verbose:
                print("||R_22||_2 = ", np.linalg.norm(R_22))
        #endif
        err = 0
        err = np.linalg.norm(M[:,pvt]- Q@R)
        if verbose:
            print("||A[:,p] - Q*R|| = ",err)
        #print(np.linalg.norm(M[:,pvt] - Q@R, axis = 0))
        #print(np.linalg.norm(M[:,pvt] - Q@R, axis = 1))
    return Q,R,err,errQ

def getR(matrix_layout, m, n, A, k = 0):
    ## Definire M, N e K
    if (m >= n):
        mm = m
        nn = n
    elif (m < n):
        mm = m
        nn = m
    #end
    if k == 0:
        k = min(mm,nn)
  
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


def house(x, tol = 1e-17):
    m = x.size
    x = x.reshape((m,1))
    sigma = x[1:].T@x[1:]
    v = np.ones((m,1))
    v[1:] = x[1:].copy()
    x0 = x[0]
    if sigma <= tol:
        if x0 >= tol:
            beta = np.array([0])
        else:
            beta = np.array([-2])
        #end
    else:
        mu = math.sqrt(x0**2 + sigma)
        if x0 <= tol:
            v0 = x[0] -mu
        else:
            v0 = -1*sigma/(x0+mu)
        #end
        beta = 2*v0**2/(sigma + v0**2)
        v[1:] /= v0
    return v, beta.flatten()

def house_qr(A, inplace = False, tol = 1e-17):
    if A.ndim <2:
        m = A.shape[0]
        n = 1
        A = A.reshape((m,n))
    else:
        m,n = A.shape
    if inplace:
        R = A
    else:
        R = A.copy()
    #endif
    
    Q = np.eye(m)
    
    for j in range(n):
        v,beta = house(R[j:,j], tol = tol)
        R[j:,j:] -= beta*(v@(v.T@R[j:,j:]))
        if (j+1<m):
            R[j+1:,j] = 0
        #end
        Qj = np.eye(m)
        Qj[j:,j:] -= beta*v@v.T
        Q= Q@Qj
    #endfor
    return Q,R

def Householder(A,verbose=False):
    # INPUT: A - np.array 2D
    #l = A.shape[0]
    m = A.shape[0] 
    n = A.shape[1]
    V = np.zeros([m, n])
    tau = np.zeros([n])
    for k in range(0,n): # itero sulle colonne
        x = np.atleast_2d(A[k:m,k]).T
        e_1 = np.zeros((m-k,1)); e_1[0] = 1.0;
        segno = 1 if float(x[0]) >= 0 else -1
        
        alpha = x[0]
        beta = (-segno)*np.linalg.norm(x,2)
        tau[k] = (beta - alpha)/beta
        
        v_k = segno*np.linalg.norm(x,2)*e_1 + x; # v_k e' stato cambiato di segno per maggior efficienza del calcolo
        v_k = v_k / v_k[0]#np.linalg.norm(v_k,2);
        A[k:m,k:n] = A[k:m,k:n] - (1/(np.linalg.norm(v_k)**2))*2*v_k@v_k.T@A[k:m,k:n];
        if verbose: print('A = ',A)
        V[k:m,k] = np.squeeze(v_k)
    #endfor
    return V,A,tau


# In[11]:


def block_householder(A, verbose = False):
    '''
    calcola la rappresentazione a blocchi Q = I - VTV^T
    '''

    #AA = A.copy()
    
    Vh, R, tau = Householder(A)
    
    m,n = Vh.shape
    Th = Vh.T@Vh

    if (verbose):
        print('T hat')
        print(Th)
    
    T = np.zeros((n,n))
    T[0,0] = tau[0]
    for i in range(1,n):
        T[:i,i] = -tau[i]*np.triu(T[:i,:i])@Th[:i,i]
        T[i,i] = tau[i]
    #endfor

    # A = QR = (np.eye(m)-(Vh@T)@Vh.T) @ R
    # print(np.linalg.norm(AA- ((np.eye(m)-(Vh@T)@Vh.T)@R)) )
    
    return Vh, T

def DM(norms, R, k, r, Z = [], CS = [], fullCS = False, rectCS = False,
       thres = 0.2222, thres_high = 0.9,
       thresnrm  = 0.8, tol = 1e-17, markZ = 1, verbose = False):
    lenZold = len(Z)
    ncols = norms.size
    
    marked = np.zeros((ncols,), dtype = np.int32)
    
    idx = np.argsort(norms, axis = None)
    idx = idx[::-1]
    normsI = norms[idx]
    #print('norma minima = ', np.amin(norms))

    thres_loc = thresnrm*thresnrm*norms[idx[0]]
    C = np.where(normsI>thres_loc)[0]
    #C = np.where(norms>thres_loc)[0]

    nz = 0
    
    if (len(C)> k):
        C = C[:k]
    if verbose:
        print('candidati = ',C.size, normsI[C], idx[C])
        #print(CS[I[C[:4]],:][:,I[C[:4]]])
    nc = C.size
    #print('numero candidati =', nc)
    
    if fullCS:
        cosmat = CS[:,idx[C]][idx[C],:]
        #print(cosmat)
    else:
        #costruisco matrice dei coseni
        #le colonne candidate sono gli indici colonna     
        cols = R[:,idx[:nc]]/np.sqrt(normsI[:nc])
        if rectCS:
            rows = R[:,idx]/np.sqrt(normsI)
            
        else:
            rows = cols.copy()
        #end
        cosmat = np.absolute(rows.T@cols) #cosmat = cols.T@cols
        small_angles = np.absolute(cosmat -1)
    #end

        
    
    # to be permuted ad top
    idxI  = [idx[0]]
    idxtmp = [0]
    add = 1
    # to be permuted at bottom
    Zn = []
    #CZ = np.where()

    
    marked[idx[0]] = 1
    for i in range(1,nc):
        if (marked[idx[i]] == 0):
            maxcos = np.amax(cosmat[idxtmp, i])
            if (maxcos < thres) and (add < k): # and (abs(wzI[i] -nrm)>tol):
                idxtmp.append(i)
                idxI.append(idx[i])
                marked[idx[i]] = 1
                add = add+1
                if rectCS:
                    LD = np.where(small_angles[:,i]<=thres_high)[0]
                    lenLD = len(LD)
                    if (lenLD>0):
                        for d in LD:
                            if (marked[idx[d]] == 0):
                                Zn.append(idx[d])
                                marked[idx[d]] = markZ
                                nz += 1
                            #end
                        #end
                    #end
                #end
            #end
            if (abs(maxcos-1) <= thres_high):
                Zn.append(idx[i])
                marked[idx[i]] = markZ
                nz += 1
            #end
        #end
        #print('Zn', Zn)
    #end
    Zn.reverse()

    return idxI, marked, Zn

def deviation_maximization_perm(norms, jpvt, R, k, r, end, 
                                Z = [], CS = [], fullCS = False, rectCS = False,
                                thres = 0.2222, thres_high = 0.9, 
                                thresnrm  = 0.8, tol = 1e-17, verbose = False, it = -1):
    # thres e' la tolleranza sul coseno
    # no scusa, viene usata... si e' la tolleranza sulla magnitudo del vettore

    markZ = 2
    ncols = end-r
    idxI, marked, Zn =  DM(norms[r:end], R[r:,r:end], k, r,
                           markZ = markZ,
                           Z = Z, CS = [], fullCS = False, rectCS = rectCS,
                           thres = thres, thres_high = thres_high,
                           thresnrm  = thresnrm, tol = tol, verbose = verbose)

    lenZn = len(Zn)
    add = len(idxI)
    
    nscambi=0
    #print('p', marked)
 
    # NB: le prime "r" vengono tolte e quindi "jb" e "jt" sono indici in [0:ncols]
    jb = 0 
    jt = ncols-1
    listtomove = idxI+Zn
    if 1:
        for ic in listtomove:
            while (jb < jt and (marked[jt]==1 or marked[jb]==2)):
                marked[jt], marked[jb] = marked[jb], marked[jt]
                tmp = R[:,jt+r].copy()
                R[:,jt+r] = R[:,r+jb].copy()
                R[:,r+jb] = tmp.copy()
                jpvt[jt+r], jpvt[r+jb] = jpvt[r+jb], jpvt[jt+r]
                norms[jt+r], norms[r+jb] = norms[r+jb], norms[jt+r]
                nscambi +=1
                while(jb<ncols and marked[jb]!=0 and marked[jb]!=2):
                    jb+=1
                #end
                while(jt>=0 and marked[jt]!=0 and marked[jt]!=1):
                    jt-=1
                #end
            #end
            if (marked[ic]==1):
                while(jb<ncols and marked[jb]!=0 and marked[jb]!=2):
                    jb+=1
                #end
                if ic<=jb or ic< add:
                    continue
                #end
                if (marked[jb]==0):
                    marked[ic], marked[jb] = marked[jb], marked[ic]
                    tmp = R[:,ic+r].copy()
                    R[:,ic+r] = R[:,r+jb].copy()
                    R[:,r+jb] = tmp.copy()
                    jpvt[ic+r], jpvt[r+jb] = jpvt[r+jb], jpvt[ic+r]
                    norms[ic+r], norms[r+jb] = norms[r+jb], norms[ic+r]
                    nscambi +=1
                    jb+=1
                #end
            elif (marked[ic]==2):#move bottom
                while(jt>=0 and marked[jt]!=0 and marked[jt]!=1):
                    jt-=1
                #end
                if ic>=jt or ic > ncols-lenZn-1:
                    continue
                #end
                if (marked[jt]==0):
                    marked[ic], marked[jt] = marked[jt], marked[ic]
                    tmp = R[:,ic+r].copy()
                    R[:,ic+r] = R[:,r+jt].copy()
                    R[:,r+jt] = tmp.copy()
                    jpvt[ic+r], jpvt[r+jt] = jpvt[r+jt], jpvt[ic+r]
                    norms[ic+r], norms[r+jt] = norms[r+jt], norms[ic+r]
                    nscambi +=1
                    jt-=1
                #end
            #end
        #end
    else:
        # NB: le prime "r" vengono tolte e quindi "jb" e "jt" sono indici in [0:ncols]
        jb = 0 
        jt = ncols-1
        listtomove = idxI+Zn
        ic_range = range(add,ncols-lenZn)
        print("max(listtomove) = ",np.max(listtomove))
        # MODIFICA #1 (anche il "while")
        print("len(ic_range) = ",len(ic_range),"len(listtomove) = ",len(listtomove))
        indic = 0; 
        if len(ic_range) > 0: 
            ic = ic_range[indic]
        else:
            ic = -1
        #endif
        while indic < len(ic_range) or jb < add-1 or jt > ncols-lenZn-1:
            if indic < len(ic_range): ic = ic_range[indic] 
            #print("indic = ",indic," , jb = ",jb," , jt = ",jt)
            while (jb < jt and (marked[jt]==1 or marked[jb]==2)):
                marked[jt], marked[jb] = marked[jb], marked[jt]
                tmp = R[:,jt+r].copy()
                R[:,jt+r] = R[:,r+jb].copy()
                R[:,r+jb] = tmp.copy()
                jpvt[jt+r], jpvt[r+jb] = jpvt[r+jb], jpvt[jt+r]
                norms[jt+r], norms[r+jb] = norms[r+jb], norms[jt+r]
                nscambi +=1
                while(jb<ncols and marked[jb]!=0 and marked[jb]!=2):
                    jb+=1
                #end
                while(jt>=0 and marked[jt]!=0 and marked[jt]!=1):
                    jt-=1
                #end
            #end
            if ic > -1:
                if (marked[ic]==1):
                    if ic<=jb:
                        if indic < len(ic_range): indic = indic + 1
                        continue
                    #end
                    if (marked[jb]==0):
                        marked[ic], marked[jb] = marked[jb], marked[ic]
                        tmp = R[:,ic+r].copy()
                        R[:,ic+r] = R[:,r+jb].copy()
                        R[:,r+jb] = tmp.copy()
                        jpvt[ic+r], jpvt[r+jb] = jpvt[r+jb], jpvt[ic+r]
                        norms[ic+r], norms[r+jb] = norms[r+jb], norms[ic+r]
                        nscambi +=1
                        jb+=1
                    #end
                elif (marked[ic]==2):#move bottom
                    if ic>=jt:
                        if indic < len(ic_range): indic = indic + 1
                        continue
                    #end
                    if (marked[jt]==0):
                        marked[ic], marked[jt] = marked[jt], marked[ic]
                        tmp = R[:,ic+r].copy()
                        R[:,ic+r] = R[:,r+jt].copy()
                        R[:,r+jt] = tmp.copy()
                        jpvt[ic+r], jpvt[r+jt] = jpvt[r+jt], jpvt[ic+r]
                        norms[ic+r], norms[r+jt] = norms[r+jt], norms[ic+r]
                        nscambi +=1
                        jt-=1
                    #end
                while(jb<ncols and marked[jb]!=0 and marked[jb]!=2):
                    jb+=1
                #end
                while(jt>=0 and marked[jt]!=0 and marked[jt]!=1):
                    jt-=1
                #end
                if indic < len(ic_range): indic = indic + 1
            #endif
        #end
    #end
    #print('scambi effettuati = ', nscambi,' su len(listtomove) = ',len(listtomove))

    for i in range(lenZn):
        Zn[i] +=r
    for i in range(add):
        idxI[i] +=r
    #
    #print('d', marked)#,(norms[r:r+add]))
    
    return norms, jpvt, R, idxI, Zn

def deviation_maximization(norms, R, k, r, end, Z = [], CS = [], fullCS = False, rectCS = False,
                           thres = 0.2222, thres_high = 0.9,
                           thresnrm  = 0.8, tol = 1e-17, verbose = False, it = -1):
    # thres e' la tolleranza sul coseno
    # no scusa, viene usata... si e' la tolleranza sulla magnitudo del vettore
   
    ncols = end-r
    if fullCS:
        idxI, marked, Zn =  DM(norms[r:end], R[r:,r:end], k, r, 
                               Z = Z, CS = CS[r:end,:][:,r:end], fullCS =  fullCS, rectCS = rectCS,
                               thres = thres, thres_high = thres_high,
                               thresnrm  = thresnrm, tol = tol, verbose = verbose)
    else:
        idxI, marked, Zn =  DM(norms[r:end], R[r:,r:end], k, r, 
                               Z = Z, CS = [], fullCS =  fullCS, rectCS = rectCS,
                               thres = thres, thres_high = thres_high,
                               thresnrm  = thresnrm, tol = tol, verbose = verbose)
    #end
    piv = np.zeros((ncols,), dtype = np.int32)

    lenZn = len(Zn)
    add = len(idxI)
    
    piv[:add] = idxI.copy()
    if lenZn>0:
        piv[-lenZn:] = Zn.copy()
    #end
        
    j = 0
    for i in range(add,ncols-lenZn): 
        if (marked[i] == 0):
            piv[i] = i
        elif (marked[i] == 1):
            while (marked[j] ==1):
                if (j+1 == add):
                    j = ncols-lenZn
                else:
                    j +=1
                #endif
            #endwhile
            piv[i] = j
            if (j+1 == add):
                j = ncols-lenZn
            else:
                j +=1
            #endif
        #endif
    #endfor

    piv += r
    for i in range(lenZn):
        Zn[i] +=r
    for i in range(add):
        idxI[i] +=r
    #
    
    return idxI, piv, Zn

def apply_perm(r, ncols, T, Z, jpvt, nrms, R, CS =[], fullCS = False):
    lenT = len(T)
    lenZ = len(Z)

    Ztmp = Z[::-1]
    

    for j in range(lenT):
        print('scambio r+j =', r+j, ' e T[j] =', T[j])
        if (T[j]<lenT):
            continue
        else:
            R[:,T[j]], R[:,r+j]   = R[:,r+j], R[:,T[j]]
            jpvt[T[j]], jpvt[r+j] = jpvt[r+j], jpvt[T[j]]
            nrms[T[j]], nrms[r+j] = nrms[r+j], nrms[T[j]]
            if fullCS:
                CS[:,T[j]], CS[:,r+j]   = CS[:,r+j], CS[:,T[j]]
                CS[T[j],:], CS[r+j,:]   = CS[r+j,:], CS[T[j],:]
            #end
        #end       
    #end

    for j in range(lenZ):
        print('scambio r+ncols-j =', r+ncols - (j+1), ' e Z[j] =', Ztmp[j])
        if (Ztmp[j]>=(ncols-lenZ)):
            jpvt[r+ncols - (j+1)], jpvt[Ztmp[j]] = jpvt[Ztmp[j]], jpvt[r+ncols - (j+1)]
        else:
            R[:,r+ncols - (j+1)] = R[:,Ztmp[j]].copy()
            jpvt[r+ncols - (j+1)], jpvt[Ztmp[j]] = jpvt[Ztmp[j]], jpvt[r+ncols - (j+1)]
            nrms[r+j] = nrms[Ztmp[j]]
            if fullCS:
                CS[:,r+ncols - (j+1)] = CS[:,Ztmp[j]].copy()
                CS[r+ncols - (j+1),:] = CS[Ztmp[j],:].copy()
            #end
        #end
    #end
    
    
    return jpvt, nrms, R, CS

def apply_permold(r, ppiv, piv, c, R, CS =[], fullCS = False):
    npiv = r+ppiv.size
    
    tmp = piv[ppiv].copy()
    piv[r:npiv] = tmp.copy()
    #
    tmp = R[:,ppiv].copy()
    R[:,r:npiv] = tmp.copy()
    #
    tmp = c[ppiv].copy()
    c[r:npiv] = tmp.copy()
    #
    if fullCS:
        tmp = CS[:,ppiv].copy()
        CS[:,r:npiv] = tmp.copy()
        #
        tmp = CS[ppiv,:].copy()
        CS[r:npiv,:] = tmp.copy()
    #endif
    
    return piv, c, R, CS


def DMrrqr(A, kDM = 2, thres = 0.2222, thres_high = 0.9, thresnrm = 0.5, 
           inplace = False, tol = 1e-17, verbose = False, pivot = True,
           fullfact = True, fullCS = False, rectCS = False):
    if inplace:
        R = A
    else:
        R = A.copy()
    #endif
    m,n = A.shape
    
    # normalize columns of R
    col_nrm = np.linalg.norm(R, axis = 0)
    if fullCS:
        CS = R / col_nrm
        CS = CS.T@CS
    else:
        CS = []
    #endif

    
    Q = np.eye(m)
    piv = np.arange(n,dtype = 'int')
    c = np.square(col_nrm)
    thresnrm = thresnrm
    #OR:
    #c = np.zeros(n,)
    #for j in range(n):
    #    c[j] = A[:,j].T@A[:,j]
    #end
    
    r = -1
    l = 1
    z = 0
    tau = 1.0
    it = -1
    min_mn = min(m,n)
    max_mn = max(m,n)
    Z = []
    end = n
    while (tau>tol) and (r+l+z<max_mn):# and it<1:
        it += 1
        r = r+l
        if verbose:
            print('\n\niterazione:', it)
        #compute (block) pivot
        if pivot:
            K, ppiv, Zn = deviation_maximization(c, R, kDM, r, n-z,
                                                 CS = CS,
                                                 thres = thres, 
                                                 thres_high = thres_high,
                                                 thresnrm = thresnrm, 
                                                 verbose = verbose,
                                                 rectCS = rectCS, it = it)
            
            #piv, c, R, CS = apply_perm(r, ppiv.size, K, Zn, piv, c, R, CS =[], fullCS = False)
            piv, c, R, CS = apply_permold(r, ppiv, piv, c, R, CS = CS, fullCS = fullCS)
        else:
            c, piv, R, K, Zn = deviation_maximization_perm(c, piv, R, kDM, r, n-z,
                                                           CS = [],
                                                           thres = thres, 
                                                           thres_high = thres_high,
                                                           thresnrm = thresnrm, 
                                                           verbose = verbose,
                                                           rectCS = rectCS, it = it)
        #end   
        #print(piv)
        Z = Zn+Z
        # devono essere ordinati, se no python si incasina nello scambio
        l = len(K)
        z = len(Z)
        if not fullfact:
            end = n-z
        if (r + l > min_mn):          
            K = K[:min_mn-r]
            l = len(K) 
        #end
        
        print('it = ',it,', fjb = ', len(K), ', nz = ', len(Zn), 'ztot = ', z, 'rank =', r + len(K))
        #print('K,Zn = ',K,Zn)
        #print('np.linalg.matrix_rank(R[r:,r:r+l]) = ', np.linalg.matrix_rank(R[r:,r:r+l]),len(K))
        if verbose:
            print('r = ', r, 'l = ', l, 'K = ',K)
            #print('cosines : \n', CS[K,:][:,K])
            
        
        #print('rango(R[r:, r:r+l])', np.linalg.matrix_rank(R[r:, r:r+l]), 'over', l, 'columns' )
        Qhat, R[r:,r:r+l] = QR.qr(R[r:,r:r+l].copy(), tol = 1e-14)  
        #print('||Qhat.T@Qhat - I|| = ', np.linalg.norm(Qhat.T@Qhat - np.eye(Qhat.shape[0])))
        if (r+l<n):
            R[r:,r+l:end] = Qhat.T@R[r:,r+l:end]
        # update Q
        Qr = np.eye(m)
        Qr[r:,r:] = Qhat
        Q = Q@Qr
        #print('||Qhat@Qhat.T - I|| =', np.linalg.norm(Qhat@Qhat.T - np.eye(Qhat.shape[0])))
        #print()
        
        if False:
            # update Square Norms (c)
            D = np.diag(np.sqrt(c[r+l:end]))
            for i in range(r+l,end):
                for k in range(r,r+l):
                    c[i] -=R[k,i]**2
                #end
            #end

            c[r:r+l] = -np.inf

            # udate Cosine Matrix (CS) if norms are consistent (i.e. >0)
            if (tau > tol) and fullCS:
                CS[r+l:end,r+l:end] = D@CS[r+l:end,r+l:end]@D                
                up = R[r:r+l,r+l:end].copy()
                CS[r+l:end,r+l:end] -= up.T@up
                D = np.diag(np.reciprocal(np.sqrt(c[r+l:end])))
                CS[r+l:end,r+l:end] = D@CS[r+l:end,r+l:end]@D
            #end
        else:
            # compute from scratch
            c[r:r+l] = -np.inf
            c[r+l:end] = np.linalg.norm(R[r+l:,r+l:end], axis=0)
            if fullCS:
                CSn = R[r+l:,r+l:end] / c[r+l:end]
                c[r+l:end] = np.square(c[r+l:end])
                #print((CSn.T@CSn).shape, CS[r+l:end,r+l:end].shape  )
                CS[r+l:end,r+l:end] = CSn.T@CSn
            else:
                c[r+l:end] = np.square(c[r+l:end])
            #end
        #endif
        if (end - (r+l)>0):
            tau = np.amax(c[r+l:end])
        else:
            tau = 0.0
        #endif
        #print(it, r+l, end, tau)
    #end

    
    return Q,R,piv, it, Z
