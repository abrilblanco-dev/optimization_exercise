import os
import random
import statistics

import matplotlib.pyplot as plt

from tsp_utils import (SCHEDULES, dist_matrix, held_karp, mst_tour, nearest_neighbor,
                       random_points, simulated_annealing, tour_length, two_opt)


def main():
    rng = random.Random(2026)
    n, n_inst = 9, 100
    ratios = {"MST puro": [], "MST + 2-opt": [], "2-opt (desde NN)": [], "Recocido simulado": []}
    costos = {k: [] for k in ratios}
    optimos = []

    for inst in range(n_inst):
        D = dist_matrix(random_points(n, rng))
        opt = held_karp(D)
        optimos.append(opt)

        t_mst = mst_tour(D)
        c_mst = tour_length(D, t_mst)
        t_mst2, _ = two_opt(D, t_mst)
        c_mst2 = tour_length(D, t_mst2)

        t_nn, _ = two_opt(D, nearest_neighbor(D, rng.randrange(n)))
        c_ls = tour_length(D, t_nn)

        t0 = list(range(n))
        rng.shuffle(t0)
        T0 = sum(map(sum, D)) / (n * n)
        c_sa, _, _, _ = simulated_annealing(D, t0, SCHEDULES["Exponencial"], 10_000, T0, seed=inst)

        for k, c in [("MST puro", c_mst), ("MST + 2-opt", c_mst2),
                     ("2-opt (desde NN)", c_ls), ("Recocido simulado", c_sa)]:
            costos[k].append(c)
            ratios[k].append(c / opt)

    r = ratios["MST puro"]
    print(f"Razón MST puro: media {statistics.mean(r):.3f} | mediana {statistics.median(r):.3f} | máx {max(r):.3f} (límite teórico 2)")
    print(f"\n{'Método':20s} {'Razón media':>12s} {'Razón máx':>10s} {'% óptimos':>10s}")
    for k, v in ratios.items():
        print(f"{k:20s} {statistics.mean(v):12.3f} {max(v):10.3f} {100 * sum(x < 1 + 1e-6 for x in v) / n_inst:9.0f}%")

    os.makedirs("figures", exist_ok=True)

    # Histograma de razones del MST puro
    plt.figure(figsize=(7, 5))
    plt.hist(r, bins=15, edgecolor="black", alpha=0.8)
    plt.axvline(2, color="red", linestyle="--", label="Límite teórico = 2")
    plt.axvline(1, color="green", linestyle=":", label="Óptimo = 1")
    plt.xlim(0.95, 2.1)
    plt.xlabel("Razón de aproximación  C_MST / C_óptimo")
    plt.ylabel("Número de instancias")
    plt.title("Aproximación MST: 100 instancias de 9 ciudades")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/ex5_histograma.png", dpi=150)

    # Boxplot comparativo
    plt.figure(figsize=(9, 5))
    plt.boxplot(list(ratios.values()))
    plt.xticks(range(1, len(ratios) + 1), list(ratios.keys()))
    plt.axhline(1, color="green", linestyle=":", label="Óptimo")
    plt.ylabel("Costo obtenido / costo óptimo")
    plt.title("Comparación de métodos (mismas 100 instancias)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/ex5_comparacion.png", dpi=150)
    print("\nGráficas guardadas en figures/ex5_histograma.png y figures/ex5_comparacion.png")


if __name__ == "__main__":
    main()