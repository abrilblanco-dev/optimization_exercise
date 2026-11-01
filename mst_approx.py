import os
import random
import statistics

import matplotlib.pyplot as plt

from tsp_utils import (SCHEDULES, mst_prim, nearest_neighbor, optimo_fuerza_bruta, preorder,
                       simulated_annealing, temperatura_inicial, tour_length, two_opt)


def main():
    random.seed(2026)
    n, n_inst = 9, 100
    ratios = {"MST puro": [], "MST + 2-opt": [], "2-opt (desde NN)": [], "Recocido simulado": []}

    for inst in range(n_inst):
        points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(n)]
        opt = tour_length(points, optimo_fuerza_bruta(points))      # fuerza bruta del profe

        mst = preorder(mst_prim(points))                             # aproximación MST (profe)
        c_mst = tour_length(points, mst)
        c_mst2 = tour_length(points, two_opt(points, mst)[0])        # MST + 2-opt

        nn = nearest_neighbor(points, random.randrange(n))
        c_ls = tour_length(points, two_opt(points, nn)[0])           # búsqueda local desde NN

        t0 = list(range(n))
        random.shuffle(t0)
        c_sa = simulated_annealing(points, t0, SCHEDULES["Exponencial"], 10_000,
                                   temperatura_inicial(points), seed=inst)[0]

        for k, c in [("MST puro", c_mst), ("MST + 2-opt", c_mst2),
                     ("2-opt (desde NN)", c_ls), ("Recocido simulado", c_sa)]:
            ratios[k].append(c / opt)

    r = ratios["MST puro"]
    print(f"Razón MST puro: media {statistics.mean(r):.3f} | mediana {statistics.median(r):.3f} "
          f"| máx {max(r):.3f} (límite teórico 2)")
    print(f"\n{'Método':20s} {'Razón media':>12s} {'Mediana':>8s} {'Razón máx':>10s} {'% óptimos':>10s}")
    for k, v in ratios.items():
        print(f"{k:20s} {statistics.mean(v):12.3f} {statistics.median(v):8.3f} {max(v):10.3f} "
              f"{100 * sum(x < 1 + 1e-6 for x in v) / n_inst:9.0f}%")

    os.makedirs("figures", exist_ok=True)

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
