import numpy as np
from scipy.sparse import dok_matrix
from copy import deepcopy
import matplotlib.pyplot as plt





def SparseMatrixMultiply(M, v):
    '''
      y = SparseMatrixMultiply(G, v)
      
      Multiplies a vector (x) by a sparse matrix M,
      such that y = M @ v .
      
      Inputs:
        M is an N x M dictionary-of-keys (dok) sparse matrix
        v is a vector of size M
      
      Output:
        y is a vector of size N
    '''
    # obtain indices of non-zero elements in M , nonzero returns a tuple
    rows,cols = M.nonzero()

    Nrows,Ncols = np.shape(M)

    y = np.zeros(Ncols)
    
    for i in range(len(rows)):
      y[rows[i]] += M[rows[i],cols[i]] * v[cols[i]]

    
    return y



def PageRank(G, alpha):
    '''
     p, iters = PageRank(G, alpha)

     Computes the Google Page-rank for the network in the adjacency matrix G.
     
     Note: This function never forms a full RxR matrix, where R is the number
           of node in the network.

     Input
       G     is an RxR adjacency matrix, G[i,j] = 1 iff node j projects to node i
             Note: G must be a dictionary-of-keys (dok) sparse matrix
       alpha is a scalar between 0 and 1

     Output
       p     is a probability vector containing the Page-rank of each node
       iters is the number of iterations used to achieve a change tolerance
             of 1e-8 (changes to elements of p are all smaller than 1e-8)
    '''
    R = np.shape(G)[0]
    rows,cols = G.nonzero()
    iters = 0
    Nrows,Ncols = np.shape(G)

    u,c = np.unique(cols,return_counts=True)

    for i in range(len(u)):
         # G[:, u[i]][:] = [x / c[i] for x in G[:, u[i]]]
         G[:, u[i]] = G[:, u[i]] / c[i] # since numpy array
    #print(G)
    d = np.zeros(Ncols)
    e = np.ones(Ncols)

    for i in range(Ncols):
          if np.sum(G[:, i]) == 0:
                d[i] = 1
   # print(d)

    #P =   G + (1 / Ncols) * np.multiply(e, np.transpose(d)) 

    # Don't form the entire since it is too large
    #M = alpha * P + (1 - alpha) * (1 / Ncols) * np.multiply(e, np.transpose(e))

    converge = False
    p = e / Ncols

    #print(p)

    while(converge == False):
          p_new = alpha * (SparseMatrixMultiply(G,p) + (1 / Ncols) * np.dot(np.transpose(d),p) * e) + (1 - alpha) * e / Ncols

         # print(p_new)
          #print(G)
          #print(p)
          #print(SparseMatrixMultiply(G,p))
          #print(SparseMatrixMultiply(G,np.transpose(p)))
         #print(d)
         # print(p)
         # print(np.dot(d,p))
         #SparseMatrixMultiply(alpha * P + (1 - alpha) * (1 / Ncols) * np.multiply(e, np.transpose(e)),p)
         #derived formula to avoid computing full matrix during algorithm 


          #print(M)
          delta = np.linalg.norm(p_new - p, ord = np.inf)
          if delta < 1e-8:
            converge = True 
          iters += 1
          p = p_new

    return p, iters







# A = dok_matrix((4,4), dtype=np.float32)
# A[0,0] = 1.
# A[0,1] = 1
# A[1,2] = 1.
# A[3,0] = 1.
# A[3,2] = 1

# print(PageRank(A,0.85))