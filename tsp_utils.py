import itertools
import math
import random
import time

def two_opt_move(route: list[str], i: int, k: int) -> list[str]:
    """Reverse route[i..k]: removes edges (i-1, i) and (k, k+1)."""
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]


def mst_prim(points: list[tuple[float, float]]) -> dict[int, list[int]]:
    """Prim's algorithm from city 0; returns the tree as {node: children}."""
    best = {v: (math.dist(points[0], points[v]), 0) for v in range(1, len(points))}
    children = {v: [] for v in range(len(points))}
    while best:
        v = min(best, key=lambda u: best[u][0])
        children[best.pop(v)[1]].append(v)          # add the cheapest edge to the tree
        for u in best:
            d = math.dist(points[v], points[u])
            if d < best[u][0]:
                best[u] = (d, v)
    return children


def preorder(children: dict[int, list[int]], root: int = 0) -> list[int]:
    """Walk around the tree; visiting each city once = shortcutting repeated cities."""
    return [root] + [c for child in children[root] for c in preorder(children, child)]


def tour_length(points, tour) -> float:
    return sum(math.dist(points[tour[i]], points[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))


def optimo_fuerza_bruta(points):
    """Fuerza bruta del profe: (n-1)! tours (8! = 40,320 para 9 ciudades)."""
    return min(((0,) + p for p in itertools.permutations(range(1, len(points)))),
               key=lambda t: tour_length(points, t))

def random_points(n, rng):
    """Mismo rango que el profe: coordenadas en [0, 10]."""
    return [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(n)]


def nearest_neighbor(points, start):
    """Vecino más cercano: siempre ir a la ciudad no visitada más cercana."""
    n = len(points)
    tour, visited = [start], {start}
    while len(tour) < n:
        last = tour[-1]
        nxt = min((j for j in range(n) if j not in visited),
                  key=lambda j: math.dist(points[last], points[j]))
        tour.append(nxt)
        visited.add(nxt)
    return tour


def two_opt(points, tour):
    """2-opt usando two_opt_move del profe. Devuelve (tour_final, numero_de_mejoras)."""
    tour = list(tour)
    n = len(tour)
    best = tour_length(points, tour)
    mejoras = 0
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                nuevo = two_opt_move(tour, i, k)
                L = tour_length(points, nuevo)
                if L < best - 1e-9:
                    tour, best = nuevo, L
                    mejoras += 1
                    improved = True
    return tour, mejoras


# ---------- Esquemas de enfriamiento ----------
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


def temperatura_inicial(points):
    """Distancia media entre ciudades."""
    n = len(points)
    return sum(math.dist(a, b) for a in points for b in points) / (n * n)


def simulated_annealing(points, tour0, schedule, n_iter, T0, seed, record_every=0):
    """Pseudocódigo SA del profe adaptado a minimización:
    sucesor aleatorio (2-opt) y si empeora (delta > 0) se acepta con prob. e^(-delta/T).
    Devuelve (mejor_costo, mejor_tour, historial, segundos)."""
    rng = random.Random(seed)
    n = len(tour0)
    current = list(tour0)
    cur_cost = tour_length(points, current)
    best, best_cost = current[:], cur_cost
    hist = []
    t_ini = time.perf_counter()
    for t in range(n_iter):
        T = schedule(t, n_iter, T0)
        i = rng.randrange(1, n - 1)
        k = rng.randrange(i + 1, n)
        nxt = two_opt_move(current, i, k)
        nxt_cost = tour_length(points, nxt)
        delta = nxt_cost - cur_cost
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current, cur_cost = nxt, nxt_cost
            if cur_cost < best_cost:
                best, best_cost = current[:], cur_cost
        if record_every and t % record_every == 0:
            hist.append((t, T, cur_cost, best_cost))
    return best_cost, best, hist, time.perf_counter() - t_ini
