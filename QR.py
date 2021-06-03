import numpy as np
import math

def house(x, tol = 1e-16):
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
            if m>1:
                beta = np.array([-2])
            else:
                beta = np.array([2])
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

def qr(A, inplace = False, tol = 1e-16):
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
    min_mn = min(m,n)
    for j in range(min_mn):
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

def rrqr(A, inplace = False, tol = 1e-16):
    if inplace:
        R = A
    else:
        R = A.copy()
    #endif
    m,n = A.shape
    
    Q = np.eye(m)
    c = np.zeros(n,)
    piv = np.arange(n,dtype = 'int')
    for j in range(n):
        c[j] = A[:,j].T@A[:,j]
    #end
    r = -1
    tau = 1.0
    it = 0
    while (tau>tol) and (r+1<n):
        it += 1
        r = r+1
        #print(r)
        #compute pivot
        k = np.argmax(c)
        #exchange columns and related variables
        tmp = piv[r]
        piv[r] = piv[k]
        piv[k] = tmp 
        #
        tmp = R[:,r].copy()
        R[:,r] = R[:,k].copy()
        R[:,k] = tmp
        #
        c[k] = c[r]
        c[r] = -np.inf
        # compute and apply householder reflector
        v,beta = house(R[r:,r])
        R[r:,r:] -= beta*(v@(v.T@R[r:,r:]))
        # update Q
        Qr = np.eye(m)
        Qr[r:,r:] -= beta*v@v.T
        Q = Q@Qr
        if (r+1 < m):
            R[r+1:,r] = 0
            #R[r+1:r] = 0
        #end
        #print('r', r, 'k', k)
        for i in range(r+1,n):
            #print(i, 'c[i]', c[i], 'R[r,i]**2', R[r,i]**2, 'c[i]-R[r,i]**2 ',c[i]-R[r,i]**2 )
            c[i] -=R[r,i]**2
        #end
        tau = np.amax(c)
        #print(c)
    #end
    
    return Q,R,piv, it


def deviation_maximization_vold(Cnorm, wz, k, normalized = False, thres = 0.2222, thres_w = 0.8, tol = 1e-16):
    # thres e' la tolleranza sul coseno
    # no scusa, viene usata... si e' la tolleranza sulla magnitudo del vettore
    
    #Qui wz deve contenere le norme delle colonne, ad esempio, 
    #ma quelle gia' trattare devono essere messe a -inf
    # inoltre finora ho notato che alcune sottocolonne di R diventano 
    # linearmente dipendenti nel caso rank deficient, quindi nella deviation maximization 
    # aggiungo solo sottocolonne che abbiano norme diverse, per evitare di aggiungere 
    # la stessa colonna due volte (ho visto che capita spesso)
    
    #Deviation Maximization
    if not normalized:
        col_nrm = np.linalg.norm(Cnorm, axis = 0)
        if (np.linalg.norm(col_nrm - np.ones(Cnorm.shape[1])) > 0.0):
            Cnorm = Cnorm / col_nrm
        #endif
    #endif
    

    I = np.argsort(wz, axis = None)
    I = I[::-1]
    wzI = wz[I]
    #print(I)
    #print(wz)
    #print(wzI)
    thres_wloc = thres_w*wz[I[0]]
    C = np.where(wzI>thres_wloc)[0]     


    n = C.size
    T   = [I[0]]
    nrm = wzI[0]
    add = 1
    
    for i in range(1,n):
        c = C[i]
        #print((Cnorm[:,c].T).shape, Cnorm[:,S].shape)
        cosines = (Cnorm[:,I[c]].T)@Cnorm[:,T]

        if (np.amax(np.absolute(cosines)) < thres) and (abs(wzI[i] -nrm)>tol):      
            T.append(I[c])
            add = add+1
            nrm = wzI[i]
        #end
        if (add >= k):
            break    
        #end
    #end

    return T

def DMrrqr_vold(A, kDM = 2, inplace = False, tol = 1e-16):
    if inplace:
        R = A
    else:
        R = A.copy()
    #endif
    m,n = A.shape
    
    # normalize columns of R
    col_nrm = np.linalg.norm(R, axis = 0)
    R_nrm = R / col_nrm
    CS = R_nrm.T@R_nrm
    
    Q = np.eye(m)
    c = np.zeros(n,)
    piv = np.arange(n,dtype = 'int')
    for j in range(n):
        c[j] = A[:,j].T@A[:,j]
    #end
    r = -1
    l = 1
    tau = 1.0
    it = 0
    min_mn = min(m,n)
    while (tau>tol) and (r+l<min_mn):
        it += 1
        r = r+l
        #compute pivot
        K = deviation_maximization(R_nrm, c, kDM, normalized = True, thres_w = 0.5)
        # devono essere ordinati, se no python si incasina nello scambio
        l = len(K)
        if (r + l > m):          
            K = K[:m-r]
            l = len(K) 
        #end
        idxold = piv[r:r+l].copy()
        idx = list(range(r,r+l))
        for i,item in enumerate(idxold):
            if item in K:
                K.remove(item)
                idx.remove(i+r)
            #end
        #end
        idxold = idx.copy()
        for item in idxold:
            Kold = K.copy()
            if item in Kold:
                K.remove(item)
                idx.remove(item)
            #end
        #end
        if (len(K)) > 0:
            tmp = piv[idx].copy()
            piv[idx] = piv[K].copy()
            piv[K] = tmp.copy()
            #
            tmp = R_nrm[:,idx].copy()
            R_nrm[:,idx] = R_nrm[:,K].copy()
            R_nrm[:,K] = tmp.copy()
            #
            tmp = R[:,idx].copy()
            R[:,idx] = R[:,K].copy()
            R[:,K] = tmp.copy()
            #
            tmp = c[idx].copy()
            c[idx] = c[K].copy()
            c[K] = tmp.copy()
        #end
        Qhat, R[r:,r:r+l] = qr(R[r:,r:r+l].copy(), tol = tol)
        if (r+l<n):
            R[r:,r+l:] = Qhat.T@R[r:,r+l:]
        # update Q
        Qr = np.eye(m)
        Qr[r:,r:] = Qhat
        Q = Q@Qr
        for i in range(r+l,n):
            for j in range(r,r+l):
                c[i] -=R[j,i]**2
        #end
        c[r:r+l] = -np.inf
        tau = np.amax(c)
    #end

    
    return Q,R,piv, it

def deviation_maximization(CS, wz, k, r, thres = 0.2222, thresnrm  = 0.8, tol = 1e-16, verbose = False):
    # thres e' la tolleranza sul coseno
    # no scusa, viene usata... si e' la tolleranza sulla magnitudo del vettore
    
    #Qui wz deve contenere le norme delle colonne, ad esempio, 
    #ma quelle gia' trattare devono essere messe a -inf
    # inoltre finora ho notato che alcune sottocolonne di R diventano 
    # linearmente dipendenti nel caso rank deficient, quindi nella deviation maximization 
    # aggiungo solo sottocolonne che abbiano norme diverse, per evitare di aggiungere 
    # la stessa colonna due volte (ho visto che capita spesso)
    
    marked = np.zeros((CS.shape[0],), dtype = np.int32)

    I = np.argsort(wz, axis = None)
    I = I[::-1]
    wzI = wz[I]
    #print(I)
    #print(wz)
    #print(wzI)
    thres_wloc = thresnrm*wz[I[0]]
    C = np.where(wzI>thres_wloc)[0]     
    if (len(C)> k*2):
        C = C[:k*2]
    if verbose:
        print('candidati = ',C.size, wzI[C[:4]], I[C[:4]])
        print(CS[I[C[:4]],:][:,I[C[:4]]])
    
    n = C.size
    #nrm = wzI[0]

    # to be permuted ad top
    T   = [I[0]]
    add = 1

    marked[I[0]] = 1
    
    for i in range(1,n):
        c = C[i]
        #print((Cnorm[:,c].T).shape, Cnorm[:,S].shape)
        cosines = np.absolute(CS[I[c],T])
        #print('cosines =', cosines)
        if (np.amax(cosines) < thres): # and (abs(wzI[i] -nrm)>tol):      
            T.append(I[c])
            marked[I[c]] = 1
            add = add+1
        #end
        if (add >= k):
            break    
        #end
    #end
    
    lenT = len(T)
    n = CS.shape[0]
    piv = np.zeros((n-r), dtype = np.int32)
    piv[:lenT] = T.copy()
  
    j = r
    for i, idx in enumerate(range(r+lenT,n)): #-lenZ
        if (marked[idx] == 0):
            piv[lenT+i] = idx
        elif(marked[idx] == 1):
            while((marked[j] ==1) and(j<r+lenT)):
                j +=1
            #endwhile
            piv[lenT+i] = j
            j +=1
        #endif
    #endfor

    return T, piv

def DMrrqr(A, kDM = 2, thres = 0.2222, thresnrm = 0.5, inplace = False, tol = 1e-17, verbose = False):
    if inplace:
        R = A
    else:
        R = A.copy()
    #endif
    m,n = A.shape
    
    # normalize columns of R
    col_nrm = np.linalg.norm(R, axis = 0)
    CS = R / col_nrm
    CS = CS.T@CS
    
    Q = np.eye(m)
    piv = np.arange(n,dtype = 'int')
    c = np.square(col_nrm)
    #OR:
    #c = np.zeros(n,)
    #for j in range(n):
    #    c[j] = A[:,j].T@A[:,j]
    #end
    
    r = -1
    l = 1
    tau = 1.0
    it = 0
    min_mn = min(m,n)
    while (tau>tol) and (r+l<min_mn):
        it += 1
        r = r+l
        #compute (block) pivot 
        K, ppiv = deviation_maximization(CS, c, kDM, r, thres = thres, thresnrm = thresnrm) #, verbose = verbose)
        # devono essere ordinati, se no python si incasina nello scambio
        l = len(K)
        if (r + l > m):          
            K = K[:m-r]
            l = len(K) 
        #end
        if verbose:
            print('r = ', r, 'l = ', l, 'K = ',K)
            #print('cosines : \n', CS[K,:][:,K])
        
        idx = list(range(r,r+l))
        
        if (0):
            idxold = piv[r:r+l].copy()
            for i,item in enumerate(idxold):
                if item in K:
                    K.remove(item)
                    idx.remove(i+r)
                #end
            #end
            idxold = idx.copy()
            for item in idxold:
                Kold = K.copy()
                if item in Kold:
                    K.remove(item)
                    idx.remove(item)
                #end
            #end

            if (len(K)) > 0:
                tmp = piv[idx].copy()
                piv[idx] = piv[K].copy()
                piv[K] = tmp.copy()
                #
                tmp = R[:,idx].copy()
                R[:,idx] = R[:,K].copy()
                R[:,K] = tmp.copy()
                #
                tmp = c[idx].copy()
                c[idx] = c[K].copy()
                c[K] = tmp.copy()
                #
                tmp = CS[idx,:].copy()
                CS[idx,:] = CS[K,:].copy()
                CS[K,:] = tmp.copy()
                #
                tmp = CS[:,idx].copy()
                CS[:,idx] = CS[:,K].copy()
                CS[:,K] = tmp.copy()
            #end
        elif (1):
            tmp = piv[ppiv].copy()
            piv[r:] = tmp.copy()
            #
            tmp = R[:,ppiv].copy()
            R[:,r:] = tmp.copy()
            #
            tmp = c[ppiv].copy()
            c[r:] = tmp.copy()
            #
            tmp = CS[:,ppiv].copy()
            CS[:,r:] = tmp.copy()
            #
            tmp = CS[ppiv,:].copy()
            CS[r:,:] = tmp.copy()
        else:
            for i, idxi in enumerate(idx):
                Ki=K[i]
                if (idxi!=Ki):
                    #print('scambio', idxi, ' e',Ki, 'piv[',idxi,']=', piv[idxi])
                    tmp = piv[idxi].copy()
                    piv[idxi] = piv[Ki].copy()
                    piv[Ki] = tmp.copy()
                    #
                    tmp = R[:,idxi].copy()
                    R[:,idxi] = R[:,Ki].copy()
                    R[:,Ki] = tmp.copy()
                    #
                    tmp = c[idxi].copy()
                    c[idxi] = c[Ki].copy()
                    c[Ki] = tmp.copy()
                    #
                    tmp = CS[idxi,:].copy()
                    CS[idxi,:] = CS[Ki,:].copy()
                    CS[Ki,:] = tmp.copy()
                    #
                    tmp = CS[:,idxi].copy()
                    CS[:,idxi] = CS[:,Ki].copy()
                    CS[:,Ki] = tmp.copy()
        if verbose:
            print(piv)
            
        
        Qhat, R[r:,r:r+l] = qr(R[r:,r:r+l].copy(), tol = 1e-14)
        #print(r,l,np.linalg.norm(Qhat.T@Qhat - np.eye(Qhat.shape[0])))
        if (r+l<n):
            R[r:,r+l:] = Qhat.T@R[r:,r+l:]
        # update Q
        Qr = np.eye(m)
        Qr[r:,r:] = Qhat
        Q = Q@Qr
        
        # update Square Norms (c)
        D = np.diag(np.sqrt(c[r+l:]))
        for i in range(r+l,n):
            for k in range(r,r+l):
                c[i] -=R[k,i]**2
            #end
        #end
        
        c[r:r+l] = -np.inf
        tau = np.amax(c)
        # udate Cosine Matrix (CS) if norms are consistent (i.e. >0)
        if tau > tol:
            CS[r+l:,r+l:] = D@CS[r+l:,r+l:]@D
            for i in range(r+l,n):
                for j in range(i,n):
                    for k in range(r,r+l):
                        CS[i,j] -= R[k,i]*R[k,j]
                    #end
                    CS[j,i] = CS[i,j]
                #end
            #end
            D = np.diag(np.reciprocal(np.sqrt(c[r+l:])))
            CS[r+l:,r+l:] = D@CS[r+l:,r+l:]@D
        #end
        
        # CHECK sulla matrice CS 
        # ricalcolo i coseni da capo per 
        # controllare che l'update sia eseguito
        #correttamente
        if False: 
            CS2 = np.eye(n)
            for i in range(r+l,n):
                zi = R[r+l:,i]
                nrm_zi = np.linalg.norm(zi)
                for j in range(i+1,n):
                    zj = R[r+l:,j]
                    nrm_zj = np.linalg.norm(zj)
                    CS2[i,j] = np.dot(zi,zj)/(nrm_zi*nrm_zj)
                    CS2[j,i] = CS2[i,j]
                #
            #
            print(np.allclose(CS[r+l:,r+l:],CS2[r+l:,r+l:]))   
            #print(CS[r+l:,r+l:],'\n',CS2[r+l:,r+l:],'\n\n')
        #end
    #end

    
    return Q,R,piv, it
