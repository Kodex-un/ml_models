import numpy as np
import tensorflow as tf

def left_shift(x):
    dims = x.shape
    x = tf.pad(x, [(0,0), (1,0)])
    x = tf.reshape(x, [dims[1]+1, dims[0]])
    x = x[1:, :]
    x = tf.reshape(x, dims)
    return x

def right_shift(x):
    dims = x.shape
    x = tf.pad(x, [(0,0), (0,1)])
    x = tf.reshape(x, [dims[1]+1, dims[0]])
    x = x[:-1, :]
    x = tf.reshape(x, dims)
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

        ones = tf.ones((n,n), dtype=q.dtype)
        lower_mask = tf.linalg.band_part(ones, -1, 0)[n-m:]
        upper_mask = (tf.linalg.band_part(ones, 0, -1) - tf.linalg.band_part(ones, 0, 0))[n-m:]

        U_rev = tf.reverse(U, [0])
        U_rev = tf.transpose(U_rev)
        lower_diag = left_shift(q@U_rev)

        if m < n:
            U_part = U[:m-n]
            U_part = tf.transpose(U_part)
            upper_diag = right_shift(q@U_part)
            upper_diag = tf.pad(upper_diag, [(0, 0), (n-m, 0)])
        else:
            U_t = tf.transpose(U)
            upper_diag = right_shift(q@U_t)
            
        return upper_diag*upper_mask + lower_diag*lower_mask # these zero out the garbage parts
    
    else:
        raise ValueError("Incorrect direction entry.")

qlen = 3
klen = 5
d_model = 12
M, q, U = generate_samples(qlen, klen, d_model)
pred = rel_position(q, U, 'both').numpy()
print(pred.shape)
print("Correct answer:\n", np.round(M, 2))
print("Shifted algorithm:\n", np.round(pred, 2))
print("Match?", np.allclose(M, pred))
