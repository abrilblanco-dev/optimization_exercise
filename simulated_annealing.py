import os
import random
import statistics

import matplotlib.pyplot as plt

from tsp_utils import SCHEDULES, dist_matrix, random_points, simulated_annealing


def main():
    rng = random.Random(11)
    n = 30
    D = dist_matrix(random_points(n, rng))
    T0 = sum(map(sum, D)) / (n * n)      # temperatura inicial ~ distancia media entre ciudades

    tour0 = list(range(n))
    rng.shuffle(tour0)                    # misma solución inicial aleatoria para todos

    os.makedirs("figures", exist_ok=True)

    # 1) Curvas de convergencia (N fijo, una semilla)
    N = 50_000
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for nombre, sch in SCHEDULES.items():
        _, _, hist, _ = simulated_annealing(D, tour0, sch, N, T0, seed=0, record_every=100)
        its = [h[0] for h in hist]
        axs[0].plot(its, [h[2] for h in hist], alpha=0.35)
        axs[0].plot(its, [h[3] for h in hist], label=f"{nombre} (mejor)", linewidth=2)
        axs[1].plot([h[1] for h in hist], [h[2] for h in hist], alpha=0.8, label=nombre)
    axs[0].set_xlabel("Iteración")
    axs[0].set_ylabel("Costo (delgado: actual, grueso: mejor)")
    axs[0].set_title("Convergencia: costo vs. iteraciones")
    axs[0].legend()
    axs[1].set_xscale("log")
    axs[1].invert_xaxis()
    axs[1].set_xlabel("Temperatura (escala log, decreciente)")
    axs[1].set_ylabel("Costo actual")
    axs[1].set_title("Costo vs. temperatura")
    axs[1].legend()
    for ax in axs:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/ex4_convergencia.png", dpi=150)

    # 2) Tiempo vs. calidad final (varios N y semillas)
    Ns = [5_000, 20_000, 50_000]
    semillas = range(5)
    print(f"{'Esquema':14s} {'N':>7s} {'Tiempo medio (s)':>17s} {'Costo final medio':>18s} {'Mejor':>8s}")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for nombre, sch in SCHEDULES.items():
        tiempos, costos = [], []
        for Nit in Ns:
            res = [simulated_annealing(D, tour0, sch, Nit, T0, seed=s) for s in semillas]
            t_med = statistics.mean(r[3] for r in res)
            c_med = statistics.mean(r[0] for r in res)
            c_min = min(r[0] for r in res)
            tiempos.append(t_med)
            costos.append(c_med)
            print(f"{nombre:14s} {Nit:7d} {t_med:17.3f} {c_med:18.2f} {c_min:8.2f}")
        ax.plot(tiempos, costos, "o-", label=nombre)
        for t, c, Nit in zip(tiempos, costos, Ns):
            ax.annotate(f"{Nit // 1000}k", (t, c), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Tiempo de ejecución (s)")
    ax.set_ylabel("Costo final medio (menor = mejor)")
    ax.set_title("Tiempo vs. calidad por esquema de enfriamiento")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/ex4_tiempo_vs_calidad.png", dpi=150)
    print("\nGráficas guardadas en figures/ex4_convergencia.png y figures/ex4_tiempo_vs_calidad.png")


if __name__ == "__main__":
    main()