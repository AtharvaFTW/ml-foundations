import numpy as np

def scaled_dot_product_attention(Q,K,V):
    """
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        output: shape (seq_len, d_v)
        weights: shape (seq_len, seq_len)
    """
    
    raw_scores = np.matmul(Q, K.T)/np.sqrt(Q.shape[-1])
    weights = softmax(raw_scores)
    output = np.matmul(weights,V)

    return output, weights


def softmax(x):
    """
    Apply softmax along the last axis.
    """

    exp = np.exp(x - np.max(x))
    return exp/np.sum(exp, axis=-1, keepdims= True)


if __name__ == "__main__":
    seq_len, d_k, d_v = 4, 8, 8
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)

    output, weights = scaled_dot_product_attention(Q,K,V)

    print("Output Shape", output.shape)
    print("Weights Shape", weights.shape)
    print(f"Weights sum to 1? {np.allclose(weights.sum(axis= -1) , 1.0)}")