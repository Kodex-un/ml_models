import numpy as np

def left_shift(x):
    dims = x.shape
    x = np.pad(x, [(0,0), (1,0)])
    x = x.reshape(dims[1]+1, dims[0])
    x = x[1:,:]
    x = x.reshape(*dims)
    return x.reshape(*dims)

def right_shift(x):
    dims = x.shape
    x = np.pad(x, [(0,0), (0,1)])
    x = x.reshape(dims[1]+1, dims[0])
    x = x[:-1,:]
    x = x.reshape(*dims)
    return x

def generate_samples(qlen, klen, d_model):
    q = np.random.randn(qlen, d_model)
    U = np.random.randn(klen, d_model)

    offset = klen-qlen # if klen>qlen, start qlen at offset position
    M = np.zeros((qlen, klen))
    for i in range(qlen):
        for j in range(klen):
            M[i,j] = q[i]@U[abs(i-j+offset)] # this is a vector dot product
    return M, q, U

def rel_position(q, U, direction='left'):
    if direction == 'left':
        return left_shift(q@np.flipud(U).T) # remember we need to reverse positions (up/down flip)
    
    if direction == 'right':
        m, n = q.shape[0], U.shape[0]
        if m < n:
            return np.pad(right_shift(q@U[:m-n].T), [(0,0), (n-m,0)])
        return right_shift(q@U.T)

    if direction=='both':
        m, n = q.shape[0], U.shape[0]        
        lower_mask, upper_mask = np.tril(np.ones((n,n)))[n-m:], np.triu(np.ones((n,n)), k=1)[n-m:]
        
        print(f'q: {q.shape}, U: {U.shape}')
        lower_diag = left_shift(q@np.flipud(U).T)
        print(f'lower_diag: {lower_diag.shape}')

        if m < n:
            upper_diag = right_shift(q@U[:m-n].T)
            upper_diag = np.pad(upper_diag, [(0,0), (n-m,0)])
        else:
            upper_diag = right_shift(q@U.T)
            
        return upper_diag*upper_mask + lower_diag*lower_mask # these zero out the garbage parts
    
    else:
        raise ValueError("Incorrect direction entry.")

qlen = 3
klen = 5
d_model = 12
M, q, U = generate_samples(qlen, klen, d_model)
pred = rel_position(q, U, 'both')
print("Correct answer:\n", np.round(M, 2))
print("Shifted algorithm:\n", np.round(pred, 2))
print("Match?", np.allclose(M, pred))
