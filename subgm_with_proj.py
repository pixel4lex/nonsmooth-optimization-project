import re

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (7, 4)
plt.rcParams["axes.grid"] = True


def set_seed(seed=0):
    np.random.seed(seed)


def sign0(x, eps=1e-12):
    """
    sign(x) с 'нулём' в малой окрестности 0:
    +1 если x>eps, -1 если x<-eps, 0 иначе.
    (корректный выбор субградиента для |x| в точке 0)
    """
    s = np.zeros_like(x, dtype=float)
    s[x > eps] = 1.0
    s[x < -eps] = -1.0
    return s


# =========================================================
# Example 2.1 — Phase retrieval
#   g(x) = (1/m) * sum_i | <a_i, x>^2 - b_i |
#   Dist(x, X*) / ||x̄|| = min(||x-x̄||, ||x+x̄||) / ||x̄||   (знак неидентифицируем)
# =========================================================
def generate_phase(d, m, noisy=False, outlier_prob=0.1):
    A = np.random.randn(m, d)
    x_true = np.random.randn(d)
    clean = (A @ x_true) ** 2

    if not noisy:
        b = clean
    else:
        # выбросы: b_i = (1-z_i)<a_i,x̄>^2 + z_i|ζ_i|,  ζ_i ~ N(0,100)
        z = np.random.binomial(1, outlier_prob, size=m)
        zeta = np.random.normal(0.0, 10.0, size=m)  # std=10 -> N(0,100)
        b = (1 - z) * clean + z * np.abs(zeta)

    return A.astype(float), b.astype(float), x_true.astype(float)


def phase_objective(A, b, x):
    u = A @ x
    r = u**2 - b
    return float(np.mean(np.abs(r)))


def phase_subgrad(A, b, x):

    # Обозначим u_i=<a_i,x>, r_i=u_i^2-b_i.
    # Тогда один субградиент:
    # ζ(x) = (2/m) * sum_i u_i * sign(r_i) * a_i

    m = A.shape[0]
    u = A @ x
    r = u**2 - b
    s = sign0(r)
    return (2.0 / m) * (A.T @ (u * s))


def dist_phase(x, x_true):

    # Dist(x, X*)/||x̄|| как в статье (учёт x ~ -x):
    # Dist = min(||x-x̄||, ||x+x̄||) / ||x̄||

    denom = np.linalg.norm(x_true) + 1e-16
    return float(min(np.linalg.norm(x - x_true), np.linalg.norm(x + x_true)) / denom)


def spectral_init_phase(A, b, n_iter=50):

    # Спектральная инициализация (power iteration) для
    # M = (1/m) * sum_i b_i a_i a_i^T
    # без явного построения M:
    # v <- A^T ( b ⊙ (A v) )

    m, d = A.shape
    v = np.random.randn(d)
    v /= np.linalg.norm(v) + 1e-16

    for _ in range(n_iter):
        v = A.T @ (b * (A @ v))
        v /= (np.linalg.norm(v) + 1e-16)

    scale = np.sqrt(np.mean(b))
    return (scale * v).astype(float)


# =========================================================
# Example 2.2 — Covariance estimation
#   g(X) = (1/m) * sum_{i=1}^{m/2} | <XX^T, a_{2i}a_{2i}^T - a_{2i-1}a_{2i-1}^T> - (b_{2i}-b_{2i-1}) |
#   Dist(X, X*)/||X̄||_F = min_{Q^TQ=I} ||XQ - X̄||_F / ||X̄||_F  (неоднозначность по правому ортогональному множителю)
# =========================================================
def generate_covariance(d, r, m, noisy=False, outlier_prob=0.1):
    assert m % 2 == 0, "m must be even"
    A = np.random.randn(m, d)
    X_true = np.random.randn(d, r)

    clean = np.sum((A @ X_true) ** 2, axis=1)  # ||X̄^T a_i||^2

    if not noisy:
        b = clean
    else:
        z = np.random.binomial(1, outlier_prob, size=m)
        zeta = np.random.normal(0.0, 10.0, size=m)  # std=10 -> N(0,100)
        b = (1 - z) * clean + z * np.abs(zeta)

    return A.astype(float), b.astype(float), X_true.astype(float)


def covariance_objective(A, b, X):
    m, d = A.shape
    A_odd, A_even = A[0::2], A[1::2]
    b_odd, b_even = b[0::2], b[1::2]

    n_odd = np.sum((A_odd @ X) ** 2, axis=1)
    n_even = np.sum((A_even @ X) ** 2, axis=1)

    d_i = b_even - b_odd
    r_i = (n_even - n_odd) - d_i

    return float(np.sum(np.abs(r_i)) / m)


def covariance_subgrad(A, b, X):

    # Используем тождество:
    #   <XX^T, aa^T> = a^T XX^T a = ||X^T a||^2
    # Тогда для пар (2i-1, 2i) получаем r_i(X) и sign(r_i),
    # а субградиент реализуем без явного хранения матриц C_i.

    m, d = A.shape
    A_odd, A_even = A[0::2], A[1::2]
    b_odd, b_even = b[0::2], b[1::2]

    AX_odd = A_odd @ X
    AX_even = A_even @ X

    n_odd = np.sum(AX_odd ** 2, axis=1)
    n_even = np.sum(AX_even ** 2, axis=1)

    d_i = b_even - b_odd
    r_i = (n_even - n_odd) - d_i

    s = sign0(r_i)

    term_even = A_even.T @ (s[:, None] * AX_even)
    term_odd = A_odd.T @ (s[:, None] * AX_odd)

    return (2.0 / m) * (term_even - term_odd)


def dist_procrustes(X, X_true):

    # Dist(X, X*)/||X̄||_F как в статье (Прокруст):
    #   Dist = min_{Q^TQ=I} ||XQ - X̄||_F / ||X̄||_F
    # Решение через SVD матрицы M = X^T X̄.

    denom = np.linalg.norm(X_true, ord="fro") + 1e-16
    M = X.T @ X_true
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    Q = U @ Vt
    return float(np.linalg.norm(X @ Q - X_true, ord="fro") / denom)


def spectral_init_covariance(A, b, r, n_iter=30):

    # Спектральная инициализация (subspace / QR iteration) для топ-r подпространства
    #   M = (1/m) * sum_i b_i a_i a_i^T

    m, d = A.shape
    Q = np.random.randn(d, r)
    Q, _ = np.linalg.qr(Q)

    for _ in range(n_iter):
        Z = A.T @ (b[:, None] * (A @ Q))
        Q, _ = np.linalg.qr(Z)

    AQ = A @ Q
    T = (AQ.T @ (b[:, None] * AQ)) / m
    T = (T + T.T) / 2.0

    S, W = np.linalg.eigh(T)
    idx = np.argsort(S)[::-1]
    S = np.maximum(S[idx], 0.0)
    W = W[:, idx]
    U = Q @ W

    X0 = U * np.sqrt(S[None, :] + 1e-16)
    return X0.astype(float)


# =========================================================
# Step rules (Algorithms 1–3)
# =========================================================
def step_polyak(x, g_val, z, k, g_star=0.0):
    # α_k = (g(x_k)-g*) / ||ζ_k||^2,  x_{k+1} = x_k - α_k ζ_k
    nz2 = np.linalg.norm(z) ** 2
    if nz2 == 0:
        return x
    alpha = max(g_val - g_star, 0.0) / nz2
    return x - alpha * z


def step_constant(x, g_val, z, k, alpha=1.0):
    # x_{k+1} = x_k - α * ζ_k / ||ζ_k||
    nz = np.linalg.norm(z)
    if nz == 0:
        return x
    return x - (alpha / nz) * z


def step_geometric(x, g_val, z, k, lam=1.0, q=0.99):
    # α_k = λ q^k,  x_{k+1} = x_k - α_k * ζ_k / ||ζ_k||
    nz = np.linalg.norm(z)
    if nz == 0:
        return x
    alpha_k = lam * (q ** k)
    return x - (alpha_k / nz) * z


# =========================================================
# Universal loop: store g(x_k) but PLOT ONLY Dist (как просили)
# =========================================================
def run_subgradient(x0, objective, subgrad, dist_fn, K, step_fn):
    x = x0.copy()
    g_hist = np.zeros(K)
    dist_hist = np.zeros(K)

    for k in range(K):
        g = objective(x)
        z = subgrad(x)
        g_hist[k] = g
        dist_hist[k] = dist_fn(x)
        x = step_fn(x, g, z, k)

    return g_hist, dist_hist

def sanitize_filename(s):
    return re.sub(r'[\\/:*?"<>|]', '_', s).strip()

def plot_dist(curves, title, logy=True):
    safe_title = sanitize_filename(title)

    plt.figure()
    for name, y in curves.items():
        plt.plot(np.maximum(y, 1e-16), label=name)  # защита для log-scale
    plt.title(title)
    plt.xlabel("Number of iterations")
    plt.ylabel("dist")
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{safe_title}.png')
    plt.show()


def pick_top_k(hist_dict, k=3):
    # выбираем параметры с наименьшим финальным Dist
    keys = list(hist_dict.keys())
    keys = sorted(keys, key=lambda p: float(np.nan_to_num(hist_dict[p][1][-1], nan=np.inf, posinf=np.inf)))
    return keys[:k]


ALPHAS_CONST = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]      # constant (6 кривых)
QS_GEOM = [0.9, 0.93, 0.96, 0.97, 0.98, 0.99]         # geometric (6 кривых)


# =========================================================
# Config
# =========================================================
use_paper = False
if use_paper:
    phase_cfg = dict(d=1000, m=3000, K=6000)
    cov_cfg = dict(d=1000, r=3, m=10000, K=2000)
else:
    phase_cfg = dict(d=300, m=900, K=1500)
    cov_cfg = dict(d=300, r=3, m=3000, K=1000)


# =========================================================
# (A) NOISY — CONSTANT step
#   2 графика: Phase noisy (6 кривых) + Cov noisy (6 кривых)
# =========================================================
# --- Phase noisy ---
d_p, m_p, K_p = phase_cfg["d"], phase_cfg["m"], phase_cfg["K"]
set_seed(2)
A_pn, b_pn, x_true_pn = generate_phase(d_p, m_p, noisy=True)
x0_pn = spectral_init_phase(A_pn, b_pn)

objective_pn = lambda x: phase_objective(A_pn, b_pn, x)
subgrad_pn = lambda x: phase_subgrad(A_pn, b_pn, x)
distfn_pn = lambda x: dist_phase(x, x_true_pn)

const_phase_noisy = {}
for a in ALPHAS_CONST:
    const_phase_noisy[a] = run_subgradient(
        x0_pn, objective_pn, subgrad_pn, distfn_pn, K_p,
        step_fn=lambda x, g, z, k, aa=a: step_constant(x, g, z, k, alpha=aa),
    )

plot_dist(
    {f"α={a}": const_phase_noisy[a][1] for a in ALPHAS_CONST},
    "NOISY — Phase retrieval — Constant step: Dist(k) ",
    logy=True,
)

# --- Covariance noisy ---
d_c, r_c, m_c, K_c = cov_cfg["d"], cov_cfg["r"], cov_cfg["m"], cov_cfg["K"]
set_seed(4)
A_cn, b_cn, X_true_cn = generate_covariance(d_c, r_c, m_c, noisy=True)
X0_cn = spectral_init_covariance(A_cn, b_cn, r=r_c)

objective_cn = lambda X: covariance_objective(A_cn, b_cn, X)
subgrad_cn = lambda X: covariance_subgrad(A_cn, b_cn, X)
distfn_cn = lambda X: dist_procrustes(X, X_true_cn)

const_cov_noisy = {}
for a in ALPHAS_CONST:
    const_cov_noisy[a] = run_subgradient(
        X0_cn, objective_cn, subgrad_cn, distfn_cn, K_c,
        step_fn=lambda X, g, z, k, aa=a: step_constant(X, g, z, k, alpha=aa),
    )

plot_dist(
    {f"α={a}": const_cov_noisy[a][1] for a in ALPHAS_CONST},
    "NOISY — Covariance estimation — Constant step: Dist(k) ",
    logy=True,
)


# =========================================================
# (B) NOISY — GEOMETRIC step
#   2 графика: Phase noisy (6 кривых) + Cov noisy (6 кривых)
# =========================================================
# --- Phase noisy (те же данные A_pn,b_pn,x_true_pn и тот же старт x0_pn) ---
geom_phase_noisy = {}
for q in QS_GEOM:
    geom_phase_noisy[q] = run_subgradient(
        x0_pn, objective_pn, subgrad_pn, distfn_pn, K_p,
        step_fn=lambda x, g, z, k, qq=q: step_geometric(x, g, z, k, lam=1.0, q=qq),
    )

plot_dist(
    {f"q={q}": geom_phase_noisy[q][1] for q in QS_GEOM},
    "NOISY — Phase retrieval — Geometric step: Dist(k)  ",
    logy=True,
)

# --- Covariance noisy (те же данные A_cn,b_cn,X_true_cn и старт X0_cn) ---
geom_cov_noisy = {}
for q in QS_GEOM:
    geom_cov_noisy[q] = run_subgradient(
        X0_cn, objective_cn, subgrad_cn, distfn_cn, K_c,
        step_fn=lambda X, g, z, k, qq=q: step_geometric(X, g, z, k, lam=1.0, q=qq),
    )

plot_dist(
    {f"q={q}": geom_cov_noisy[q][1] for q in QS_GEOM},
    "NOISY — Covariance estimation — Geometric step: Dist(k) ",
    logy=True,
)


# =========================================================
# (C) EXACT — FINAL comparison (Dist only)
#   2 графика:
#     - Phase exact: Polyak + best-3 const + best-3 geom
#     - Cov exact:   Polyak + best-3 const + best-3 geom
# =========================================================
# --- Phase exact ---
set_seed(1)
A_pe, b_pe, x_true_pe = generate_phase(d_p, m_p, noisy=False)
x0_pe = spectral_init_phase(A_pe, b_pe)

objective_pe = lambda x: phase_objective(A_pe, b_pe, x)
subgrad_pe = lambda x: phase_subgrad(A_pe, b_pe, x)
distfn_pe = lambda x: dist_phase(x, x_true_pe)

# Polyak (в точном случае g* = 0)
_, dist_poly_pe = run_subgradient(
    x0_pe, objective_pe, subgrad_pe, distfn_pe, K_p,
    step_fn=lambda x, g, z, k: step_polyak(x, g, z, k, g_star=0.0),
)

const_phase_exact = {}
for a in ALPHAS_CONST:
    const_phase_exact[a] = run_subgradient(
        x0_pe, objective_pe, subgrad_pe, distfn_pe, K_p,
        step_fn=lambda x, g, z, k, aa=a: step_constant(x, g, z, k, alpha=aa),
    )

geom_phase_exact = {}
for q in QS_GEOM:
    geom_phase_exact[q] = run_subgradient(
        x0_pe, objective_pe, subgrad_pe, distfn_pe, K_p,
        step_fn=lambda x, g, z, k, qq=q: step_geometric(x, g, z, k, lam=1.0, q=qq),
    )

top3_const_pe = pick_top_k(const_phase_exact, k=3)
top3_geom_pe = pick_top_k(geom_phase_exact, k=3)

curves_phase_exact = {"polyak": dist_poly_pe}
for a in top3_const_pe:
    curves_phase_exact[f"const α={a}"] = const_phase_exact[a][1]
for q in top3_geom_pe:
    curves_phase_exact[f"geom q={q}"] = geom_phase_exact[q][1]

plot_dist(
    curves_phase_exact,
    "EXACT — Phase retrieval: Polyak + constant + geometric (Dist)",
    logy=True,
)

# --- Covariance exact ---
set_seed(3)
A_ce, b_ce, X_true_ce = generate_covariance(d_c, r_c, m_c, noisy=False)
X0_ce = spectral_init_covariance(A_ce, b_ce, r=r_c)

objective_ce = lambda X: covariance_objective(A_ce, b_ce, X)
subgrad_ce = lambda X: covariance_subgrad(A_ce, b_ce, X)
distfn_ce = lambda X: dist_procrustes(X, X_true_ce)

_, dist_poly_ce = run_subgradient(
    X0_ce, objective_ce, subgrad_ce, distfn_ce, K_c,
    step_fn=lambda X, g, z, k: step_polyak(X, g, z, k, g_star=0.0),
)

const_cov_exact = {}
for a in ALPHAS_CONST:
    const_cov_exact[a] = run_subgradient(
        X0_ce, objective_ce, subgrad_ce, distfn_ce, K_c,
        step_fn=lambda X, g, z, k, aa=a: step_constant(X, g, z, k, alpha=aa),
    )

geom_cov_exact = {}
for q in QS_GEOM:
    geom_cov_exact[q] = run_subgradient(
        X0_ce, objective_ce, subgrad_ce, distfn_ce, K_c,
        step_fn=lambda X, g, z, k, qq=q: step_geometric(X, g, z, k, lam=1.0, q=qq),
    )

top3_const_ce = pick_top_k(const_cov_exact, k=3)
top3_geom_ce = pick_top_k(geom_cov_exact, k=3)

curves_cov_exact = {"polyak": dist_poly_ce}
for a in top3_const_ce:
    curves_cov_exact[f"const α={a}"] = const_cov_exact[a][1]
for q in top3_geom_ce:
    curves_cov_exact[f"geom q={q}"] = geom_cov_exact[q][1]

plot_dist(
    curves_cov_exact,
    "EXACT — Covariance estimation: Polyak + constant + geometric (Dist)",
    logy=True,
)