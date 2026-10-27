import math
import random
import time


def random_points(n, rng, size=100.0):
    return [(rng.uniform(0, size), rng.uniform(0, size)) for _ in range(n)]


def dist_matrix(points):
    return [[math.dist(a, b) for b in points] for a in points]


def tour_length(D, tour):
    n = len(tour)
    return sum(D[tour[i]][tour[(i + 1) % n]] for i in range(n))


def nearest_neighbor(D, start):
    n = len(D)
    tour, visited = [start], {start}
    while len(tour) < n:
        last = tour[-1]
        nxt = min((j for j in range(n) if j not in visited), key=lambda j: D[last][j])
        tour.append(nxt)
        visited.add(nxt)
    return tour


def two_opt(D, tour):
    """2-opt (primera mejora). Devuelve (tour_final, numero_de_mejoras)."""
    tour = tour[:]
    n = len(tour)
    mejoras = 0
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                a, b = tour[i - 1], tour[i]
                c, d = tour[k], tour[(k + 1) % n]
                delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]
                if delta < -1e-9:
                    tour[i:k + 1] = tour[i:k + 1][::-1]
                    mejoras += 1
                    improved = True
    return tour, mejoras


# ---------- Recocido simulado ----------
def schedule_linear(t, N, T0):
    return max(T0 * (1 - t / N), 1e-9)


def schedule_exponential(t, N, T0, Tf_ratio=1e-3):
    return T0 * (Tf_ratio ** (t / N))


def schedule_log(t, N, T0):
    return T0 / (1 + math.log(1 + t))


SCHEDULES = {
    "Lineal": schedule_linear,
    "Exponencial": schedule_exponential,
    "Logarítmico": schedule_log,
}


def simulated_annealing(D, tour0, schedule, n_iter, T0, seed, record_every=0):
    """SA con 2-opt como movimiento. Devuelve (mejor_costo, mejor_tour, historial, segundos)."""
    rng = random.Random(seed)
    tour = tour0[:]
    n = len(tour)
    cur = tour_length(D, tour)
    best, best_tour = cur, tour[:]
    hist = []
    t0 = time.perf_counter()
    for t in range(n_iter):
        T = schedule(t, n_iter, T0)
        i = rng.randrange(1, n - 1)
        k = rng.randrange(i + 1, n)
        a, b = tour[i - 1], tour[i]
        c, d = tour[k], tour[(k + 1) % n]
        delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]
        if delta < 0 or rng.random() < math.exp(-delta / T):
            tour[i:k + 1] = tour[i:k + 1][::-1]
            cur += delta
            if cur < best - 1e-9:
                best, best_tour = cur, tour[:]
        if record_every and t % record_every == 0:
            hist.append((t, T, cur, best))
    return best, best_tour, hist, time.perf_counter() - t0


# ---------- MST (Prim) + recorrido preorden ----------
def mst_tour(D):
    n = len(D)
    best = {v: (D[0][v], 0) for v in range(1, n)}
    children = {v: [] for v in range(n)}
    while best:
        v = min(best, key=lambda u: best[u][0])
        children[best.pop(v)[1]].append(v)
        for u in best:
            if D[v][u] < best[u][0]:
                best[u] = (D[v][u], v)

    def preorder(node):
        out = [node]
        for c in children[node]:
            out += preorder(c)
        return out

    return preorder(0)


# ---------- Óptimo exacto (Held-Karp) ----------
def held_karp(D):
    """Longitud del tour óptimo. Rápido para n <= 12."""
    n = len(D)
    m = n - 1
    full = 1 << m
    INF = float("inf")
    dp = [[INF] * m for _ in range(full)]
    for j in range(m):
        dp[1 << j][j] = D[0][j + 1]
    for mask in range(1, full):
        for j in range(m):
            cur = dp[mask][j]
            if not (mask >> j) & 1 or cur == INF:
                continue
            for k in range(m):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                v = cur + D[j + 1][k + 1]
                if v < dp[nm][k]:
                    dp[nm][k] = v
    return min(dp[full - 1][j] + D[j + 1][0] for j in range(m))