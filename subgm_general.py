#
# General case: recover rank-r matrix X* from linear measurements y = A(X*) + s*
# with sparse outliers s* (fraction p < 1/2). We optimize the nonsmooth objective
#   g(U,V) = (1/m)||y - A(UV^T)||_1 + λ ||U^T U - V^T V||_F
# where the balancing regularizer removes the (U,V) ↦ (UT, V T^{-T}) ambiguity.
# (This is the paper’s general-case formulation.)
#


import numpy as np
import matplotlib.pyplot as plt


# --- Choose δ (theory-driven upper bound depends on p), then set λ from δ and p.
# In experiments, we pick δ conservatively as a fraction of the admissible bound,
# then λ is determined by the paper’s ζ(δ,p) rule (once δ is fixed, λ is fixed).
def choose_delta(p):
    # Theory: for a given outlier fraction p, δ must satisfy
    # δ < sqrt(2/pi) * (1 - 2p) / (3 - 2p)
    # for sharpness to hold.
    # In practice we don't know δ, so we pick a value
    # inside this admissible interval and use it to set λ.
    sqrt2pi = np.sqrt(2 / np.pi)
    return ((1 - 2 * p) * sqrt2pi / (3 - 2 * p)) * 0.5 # we stay in the middle

def choose_lambda(p, delta):
    # As article suggest, optimal λ must be ζ / 2
    sqrt2pi = np.sqrt(2 / np.pi)
    zeta = 2*(1-p)*(sqrt2pi - delta) - (sqrt2pi + delta)
    return 0.5 * zeta


# ---------------------------
# Problem size and parameters
# ---------------------------
n = 50
r = 5
m = 20*n*r
p = 0.3
max_iter = 3000
lam = choose_lambda(p, choose_delta(p))


# --------------------------------------
# Ground truth generation:
# Construct a "balanced" factorization X* = U* V*^T with U*^T U* = V*^T V*,
# matching the paper’s general-case setup so W* = [U*; V*] is well-defined.
# --------------------------------------
Phi, _ = np.linalg.qr(np.random.randn(n, r))
Psi, _ = np.linalg.qr(np.random.randn(n, r))
sign_vals = np.abs(np.random.randn(r)) + 1.0
Sigma_sqrt = np.diag(np.sqrt(sign_vals))

U_star = Phi @ Sigma_sqrt
V_star = Psi @ Sigma_sqrt
X_star = U_star @ V_star.T

W_star = np.vstack([U_star, V_star])  # W* = [U*; V*]


# --------------------------------------
# Measurement operator:
# Use i.i.d. Gaussian matrices A_i (no symmetry in general case).
# --------------------------------------
A = np.random.randn(m, n, n)


def apply_A_from_X(X):
    return np.tensordot(A, X, axes=([1, 2], [0, 1]))

def apply_A_from_UV(U, V):
    return apply_A_from_X(U @ V.T)

def apply_A_star(v):
    return np.tensordot(v, A, axes=(0, 0))


# --------------------------------------
# Outlier corruption:
# Corrupt a p-fraction of measurements with large noise (sparse outliers).
# --------------------------------------
y_clean = apply_A_from_X(X_star)
num_outliers = int(p * m)

s_star = np.zeros(m)
idx = np.random.choice(m, num_outliers, replace=False)
s_star[idx] = 10 * np.random.randn(num_outliers)

y = y_clean + s_star


g_star = np.mean(np.abs(s_star))


# --------------------------------------
# Truncated spectral initialization (Algorithm 3.1):
# Split measurements in half; threshold y1 by β·median(|y2|) to remove outlier influence,
# build E = average of y_i A_i over untrimmed indices, then rank-r SVD:
# U0 = P Π^{1/2}, V0 = Q Π^{1/2}.
# --------------------------------------
beta = 3.0
m1 = m // 2
y1 = y[:m1]
y2 = y[m1:]

tau = beta * np.median(np.abs(y2))
mask = np.abs(y1) <= tau

E = (y1[mask, None, None] * A[:m1][mask]).sum(axis=0) / m1

P, S, Qt = np.linalg.svd(E, full_matrices=False)
Pr = P[:, :r]
Sr = S[:r]
Qr = Qt.T[:, :r]

sqrtSr = np.sqrt(np.maximum(Sr, 0.0))
U0 = Pr * sqrtSr
V0 = Qr * sqrtSr

W0 = np.vstack([U0, V0])


# --------------------------------------
# Distance metric:
# dist(W, 𝓦) = min_{R orthogonal} ||W - W*R||_F accounts for rotation ambiguity.
# --------------------------------------
def dist_to_true(W):
    M = W_star.T @ W
    Uu, _, Vv = np.linalg.svd(M)
    R = Uu @ Vv
    return np.linalg.norm(W - W_star @ R, 'fro')

def split_W(W):
    return W[:n, :], W[n:, :]


# --------------------------------------
# Objective + PSD subgradient (paper’s general case)
# --------------------------------------
def g_and_subgrad(W):
    U, V = split_W(W)

    # data term
    y_hat = apply_A_from_UV(U, V)
    residual = y_hat - y
    sign = np.sign(residual)

    G = apply_A_star(sign)

    DU_data = (1.0 / m) * (G @ V)
    DV_data = (1.0 / m) * (G.T @ U)

    # balancing regularizer
    Sbal = U.T @ U - V.T @ V
    normS = np.linalg.norm(Sbal, "fro")

    if normS > 0:
        Psi = Sbal / normS
        DU_reg = 2.0 * lam * (U @ Psi)
        DV_reg = -2.0 * lam * (V @ Psi)
    else:
        DU_reg = np.zeros_like(U)
        DV_reg = np.zeros_like(V)

    DU = DU_data + DU_reg
    DV = DV_data + DV_reg

    D = np.vstack([DU, DV])
    g_val = np.mean(np.abs(residual)) + lam * normS
    return g_val, D


# quick check: f(U*) should match f* in simulation (up to numerical noise)
dist0 = dist_to_true(W0)
g_W0, _ = g_and_subgrad(W0)
g_Wstar, _ = g_and_subgrad(W_star)
print("dist(W0, W*) =", dist0)
print("g(W0) =", g_W0, "g(W*) =", g_Wstar, "g* =", g_star)


# --------------------------------------
# Subgradient method (SubGM):
# Iterate W_{k+1} = W_k - μ_k D_k with D_k ∈ ∂g(W_k).
# --------------------------------------
def run_subgm(U_init, max_iter, step_rule, record_obj=False, tol=1e-8):
    W = U_init.copy()
    dists = []
    objs = [] if record_obj else None

    for k in range(max_iter):
        g_val, D = g_and_subgrad(W)
        if record_obj:
            objs.append(g_val)
        dists.append(dist_to_true(W))

        norm_D = np.linalg.norm(D, "fro")
        if norm_D < tol:
            break

        step = step_rule(k, g_val, D)
        if step == 0:
            break

        W = W - step * D

    return np.array(dists), (np.array(objs) if record_obj else None)


# -------------------------------------------------
# (a) Constant normalized step length (stable practical variant)
# Update is U <- U - (η/||D||) D, so ||ΔU|| ≈ η each iteration.
# -------------------------------------------------
etas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
results_const = {}

for eta in etas:
    dists, _ = run_subgm(W0, max_iter,
                         step_rule=lambda k, f, D, eta=eta: eta / (np.linalg.norm(D, 'fro') + 1e-12))
    results_const[eta] = dists
    print(f"[const] eta={eta}, final dist={dists[-1]:.12f}")

# Plot (a)
plt.figure()
for eta, dists in sorted(results_const.items()):
    plt.semilogy(dists, label=f"eta={eta}")
plt.xlabel("Number of iterations")
plt.ylabel(r"dist($W_k$, $\mathcal{W}$)")
plt.title("SubGM with constant normalized step sizes")
plt.legend()
plt.tight_layout()
plt.savefig('subgm_general_const.png')
plt.show()


# ---------------------------------------------------------
# (b) Geometric steps (paper’s main schedule): μ_k = μ_0 ρ^k
# ---------------------------------------------------------
rho_list = [0.9, 0.93, 0.96, 0.97, 0.98, 0.99]
mu0 = .2
results_geom = {}

for rho in rho_list:
    def step_rule(k, f, D, mu0=mu0, rho=rho):
        return mu0 * (rho ** k)
    dists, _ = run_subgm(W0, max_iter, step_rule)
    results_geom[rho] = dists
    print(f"[geom] rho={rho}, final dist={dists[-1]:.12f}")

plt.figure()
for rho, dists in sorted(results_geom.items()):
    plt.semilogy(dists, label=f"mu0={mu0}, rho={rho}")
plt.xlabel("Number of iterations")
plt.ylabel(r"dist($W_k$, $\mathcal{W}$)")
plt.title(r"SubGM with geometrically decaying step sizes, $\mu_k = \mu_0 \rho^k$")
plt.legend()
plt.tight_layout()
plt.savefig('subgm_general_geom.png')
plt.show()


# ---------------------------------------------------------
# (c) Polyak step (uses known f* in simulation): μ_k = (f(U_k)-f*) / ||d_k||^2
# ---------------------------------------------------------
def polyak_step_rule(k, f_val, D):
    if f_val <= g_star:
        return 0.0
    return (f_val - g_star) / (np.linalg.norm(D, "fro")**2 + 1e-12)

dists_polyak, _ = run_subgm(W0, max_iter, polyak_step_rule)
print(f"[polyak] final dist={dists_polyak[-1]:.12f}")


# ---------------------------------------------------------
# Combined comparison plot (pick a few representative settings)
# ---------------------------------------------------------
plt.figure()
for rho in [0.97, 0.98, 0.99]:
    plt.semilogy(results_geom[rho], label=fr"geom: $\mu_0={mu0}, \rho={rho}$")

for eta in [0.05, 0.1, 0.15]:
    plt.semilogy(results_const[eta], label=fr"const: $\eta={eta}$")
plt.semilogy(dists_polyak, label="polyak step")
plt.xlabel("Number of iterations")
plt.ylabel(r"dist($W_k$, $\mathcal{W}$)")
plt.title("SubGM: comparing various types of step sizes")
plt.legend()
plt.tight_layout()
plt.savefig('subgm_general_compar.png')
plt.show()