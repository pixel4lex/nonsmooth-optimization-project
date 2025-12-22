#
# PSD case (X* = U*U*^T, symmetric Gaussian A_i, sparse/outlier corruption)
# This script follows the PSD setup + SubGM experiments in the paper:
#   - measurements:  y_i = <A_i, X*> + s_i
#   - objective:     f(U) = (1/m) || y - A(UU^T) ||_1
#   - subgradient:   d(U) = (2/m) A^*(Sign(A(UU^T) - y)) U   (PSD/symmetric A_i case)
#   - init:          truncated spectral method (Algorithm 3.1)
#


import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Problem size and parameters (same scale as paper’s experiments)
# ---------------------------
n = 50
r = 5
m = 5 * n * r
p = 0.3
max_iter = 3000


# --------------------------------------
# Ground truth: low-rank PSD matrix X* = U*U*^T
# (in PSD case the factorization is unique up to right orthogonal transform)
# --------------------------------------
U_star = np.random.randn(n, r)
X_star = U_star @ U_star.T


# --------------------------------------
# Symmetric Gaussian matrices A_i (PSD case in the paper)
# We generate A_i symmetric with i.i.d. N(0,1) entries on/above diagonal.
# --------------------------------------
G = np.random.randn(m, n, n)
A = np.triu(G)
A = A + np.transpose(A, (0, 2, 1))
A[:, np.arange(n), np.arange(n)] *= 0.5  # keep diagonal N(0,1)


def apply_A_from_X(X):
    # Forward map: (A(X))_i = <A_i, X>
    return np.tensordot(A, X, axes=([1, 2], [0, 1]))


def apply_A_from_U(U):
    return apply_A_from_X(U @ U.T)


def apply_A_star(v):
    # Adjoint: A^*(v) = sum_i v_i A_i
    return np.tensordot(v, A, axes=(0, 0))


# --------------------------------------
# Outlier corruption: sparse vector s* with p-fraction nonzeros (as in experiments)
# --------------------------------------
y_clean = apply_A_from_X(X_star)
num_outliers = int(p * m)
s_star = np.zeros(m)
idx = np.random.choice(m, num_outliers, replace=False)
s_star[idx] = 10 * np.random.randn(num_outliers)  # N(0, 10^2)
y = y_clean + s_star


# Optimal objective value f* = (1/m)||s*||_1 (used for Polyak step; known here in simulation)
f_star = np.mean(np.abs(s_star))


# --------------------------------------
# Truncated spectral initialization (Algorithm 3.1)
# Split measurements in half, use median-based truncation, then rank-r SVD of E.
# This robustifies the initializer against outliers.
# --------------------------------------
beta = 3.0
m1 = m // 2
y1, y2 = y[:m1], y[m1:]
tau = beta * np.median(np.abs(y2))
mask = np.abs(y1) <= tau

E = (y1[mask, None, None] * A[:m1][mask]).sum(axis=0) / m1

Ue, Se, Vte = np.linalg.svd(E, full_matrices=False)
Ur = Ue[:, :r]
Sr = Se[:r]
U0 = Ur * np.sqrt(Sr)  # PSD init: U0 = P * Π^{1/2}


# --------------------------------------
# Geometry: distance to solution set 𝒰 = {U*R : R ∈ O_r}
# (measures recovery up to orthogonal ambiguity in PSD factor)
# --------------------------------------
def dist_to_true(U):
    M = U_star.T @ U
    Uu, _, Vv = np.linalg.svd(M)
    R = Uu @ Vv
    return np.linalg.norm(U - U_star @ R, 'fro')


# --------------------------------------
# Objective + PSD subgradient (paper’s PSD case)
# f(U) = (1/m)|| y - A(UU^T) ||_1
# d(U) = (2/m) A^*(Sign(A(UU^T) - y)) U
# --------------------------------------
def f_and_subgrad(U):
    y_hat = apply_A_from_U(U)
    residual = y_hat - y
    sign = np.sign(residual)          # take Sign(0)=0
    Gm = apply_A_star(sign)           # A^*(Sign(.))
    D = (2.0 / m) * (Gm @ U)
    f_val = np.mean(np.abs(residual))
    return f_val, D


# quick check: f(U*) should match f* in simulation (up to numerical noise)
dist0 = dist_to_true(U0)
f_U0, _ = f_and_subgrad(U0)
f_Ustar, _ = f_and_subgrad(U_star)
print("dist(U0, U*) =", dist0)
print("f(U0) =", f_U0, "f(U*) =", f_Ustar, "f* =", f_star)


# --------------------------------------
# Subgradient method (SubGM): U_{k+1} = U_k - μ_k d_k
# We test different step schedules (constant-normalized, geometric, Polyak).
# --------------------------------------
def run_subgm(U_init, max_iter, step_rule, tol=1e-8):
    U = U_init.copy()
    dists = []
    for k in range(max_iter):
        f_val, D = f_and_subgrad(U)
        dists.append(dist_to_true(U))

        norm_D = np.linalg.norm(D, "fro")
        mu = step_rule(k, f_val, D)
        if norm_D < tol or mu == 0:
            break

        U = U - mu * D
    return np.array(dists)


# -------------------------------------------------
# (a) Constant normalized step length (stable practical variant)
# Update is U <- U - (η/||D||) D, so ||ΔU|| ≈ η each iteration.
# -------------------------------------------------
etas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
results_const = {}
for eta in etas:
    results_const[eta] = run_subgm(
        U0, max_iter,
        step_rule=lambda k, f, D, eta=eta: eta / (np.linalg.norm(D, "fro") + 1e-12)
    )
    print(f"[const] eta={eta}, final dist={results_const[eta][-1]:.12f}")

plt.figure()
for eta, dists in sorted(results_const.items()):
    plt.semilogy(dists, label=fr"$\eta={eta}$")
plt.xlabel("Number of iterations")
plt.ylabel(r"dist($U_k$, $\mathcal{U}$)")
plt.title("SubGM with constant normalized step sizes")
plt.legend()
plt.tight_layout()
plt.savefig("subgm_psd_const.png")
plt.show()


# ---------------------------------------------------------
# (b) Geometric steps (paper’s main schedule): μ_k = μ_0 ρ^k
# ---------------------------------------------------------
rho_list = [0.9, 0.93, 0.96, 0.97, 0.98, 0.99]
mu0 = 1.0
results_geom = {}
for rho in rho_list:
    results_geom[rho] = run_subgm(
        U0, max_iter,
        step_rule=lambda k, f, D, mu0=mu0, rho=rho: mu0 * (rho ** k)
    )
    print(f"[geom] rho={rho}, final dist={results_geom[rho][-1]:.12f}")

plt.figure()
for rho, dists in sorted(results_geom.items()):
    plt.semilogy(dists, label=fr"$\mu_0={mu0}, \rho={rho}$")
plt.xlabel("Number of iterations")
plt.ylabel(r"dist($U_k$, $\mathcal{U}$)")
plt.title(r"SubGM with geometrically decaying step sizes, $\mu_k = \mu_0 \rho^k$")
plt.legend()
plt.tight_layout()
plt.savefig("subgm_psd_geom.png")
plt.show()


# ---------------------------------------------------------
# (c) Polyak step (uses known f* in simulation): μ_k = (f(U_k)-f*) / ||d_k||^2
# ---------------------------------------------------------
def polyak_step_rule(k, f_val, D):
    if f_val <= f_star:
        return 0.0
    return (f_val - f_star) / (np.linalg.norm(D, "fro")**2 + 1e-12)

dists_polyak = run_subgm(U0, max_iter, polyak_step_rule)
print(f"[polyak] final dist={dists_polyak[-1]:.12f}")


# ---------------------------------------------------------
# Combined comparison plot (pick a few representative settings)
# ---------------------------------------------------------
plt.figure()

for rho in [0.93, 0.96, 0.97]:
    plt.semilogy(results_geom[rho], label=fr"geom: $\mu_0={mu0}, \rho={rho}$")

for eta in [0.05, 0.1, 0.15]:
    plt.semilogy(results_const[eta], label=fr"const: $\eta={eta}$")

plt.semilogy(dists_polyak, label="polyak step")
plt.xlabel("Number of iterations")
plt.ylabel(r"dist($U_k$, $\mathcal{U}$)")
plt.title("SubGM: comparing various types of step sizes")
plt.legend()
plt.tight_layout()
plt.savefig("subgm_psd_compar.png")
plt.show()