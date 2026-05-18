import torch
import time

# ===============================
# 1. Dense Attention
# ===============================
def dense_attention(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)

# ===============================
# 2. Sparse Attention (slow - loop)
# ===============================
def sparse_attention_loop(Q, K, V, window_size=16):
    n, d = Q.shape
    output = torch.zeros_like(Q)

    for i in range(n):
        left = max(0, i - window_size)
        right = min(n, i + window_size)

        scores = torch.matmul(Q[i], K[left:right].transpose(0, 1))
        weights = torch.softmax(scores, dim=-1)
        output[i] = torch.matmul(weights, V[left:right])

    return output

# ===============================
# 3. Sparse Attention (FAST - vectorized)
# ===============================
def sparse_attention_mask(Q, K, V, window_size=16):
    n, d = Q.shape

    # full attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (n, n)

    # build mask
    mask = torch.ones_like(scores) * float('-inf')

    for i in range(n):
        left = max(0, i - window_size)
        right = min(n, i + window_size)
        mask[i, left:right] = 0  # allow local window

    # apply mask
    masked_scores = scores + mask

    weights = torch.softmax(masked_scores, dim=-1)
    return torch.matmul(weights, V)

# ===============================
# 4. Experiment
# ===============================
def run_experiment(n=128, d=64, runs=5):
    torch.manual_seed(0)

    Q = torch.randn(n, d)
    K = torch.randn(n, d)
    V = torch.randn(n, d)

    # warmup
    dense_attention(Q, K, V)
    sparse_attention_loop(Q, K, V)
    sparse_attention_mask(Q, K, V)

    # ---- Dense ----
    dense_times = []
    for _ in range(runs):
        start = time.time()
        dense_out = dense_attention(Q, K, V)
        dense_times.append(time.time() - start)

    # ---- Sparse (loop) ----
    loop_times = []
    for _ in range(runs):
        start = time.time()
        loop_out = sparse_attention_loop(Q, K, V)
        loop_times.append(time.time() - start)

    # ---- Sparse (mask optimized) ----
    mask_times = []
    for _ in range(runs):
        start = time.time()
        mask_out = sparse_attention_mask(Q, K, V)
        mask_times.append(time.time() - start)

    # ---- Differences ----
    diff_loop = torch.norm(dense_out - loop_out).item()
    diff_mask = torch.norm(dense_out - mask_out).item()

    # ---- Print ----
    print("===== Preliminary Results =====")
    print(f"Input size: n={n}, d={d}")

    print("\n--- Time ---")
    print(f"Dense: {sum(dense_times)/runs:.6f} s")
    print(f"Sparse (loop): {sum(loop_times)/runs:.6f} s")
    print(f"Sparse (mask optimized): {sum(mask_times)/runs:.6f} s")

    print("\n--- Difference (vs Dense) ---")
    print(f"Loop version: {diff_loop:.6f}")
    print(f"Mask version: {diff_mask:.6f}")

    print("\n--- Speedup ---")
    print(f"Loop vs Dense: {(sum(dense_times)/runs)/(sum(loop_times)/runs):.2f}x")
    print(f"Mask vs Dense: {(sum(dense_times)/runs)/(sum(mask_times)/runs):.2f}x")

# ===============================
# Run
# ===============================
if __name__ == "__main__":
    run_experiment()