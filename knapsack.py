import os
import random

import matplotlib.pyplot as plt


def greedy(values, weights, W, key):
    """Voraz genérico: ordena los índices con `key` y mete lo que quepa"""
    order = sorted(range(len(values)), key=key)
    chosen, used, total = [], 0, 0
    for i in order:
        if used + weights[i] <= W:
            chosen.append(i)
            used += weights[i]
            total += values[i]
    return sorted(chosen), used, total


def greedy_ratio(values, weights, W):
    return greedy(values, weights, W, key=lambda i: -values[i] / weights[i])


def greedy_min_weight(values, weights, W):
    return greedy(values, weights, W, key=lambda i: weights[i])


def optimo(values, weights, W):
    best = [0] * (W + 1)
    for v, w in zip(values, weights):
        for c in range(W, w - 1, -1):
            best[c] = max(best[c], v + best[c - w])
    return best[W]


def main():
    rng = random.Random(3)
    n = 10
    vals = [rng.randint(10, 100) for _ in range(n)]
    wts = [rng.randint(5, 40) for _ in range(n)]

    instancias = {
        "Profe\n(W=50)": (["A", "B", "C"], [60, 100, 120], [10, 20, 30], 50),
        "Contraejemplo\n(W=10)": (["A", "B", "C"], [1, 2, 100], [1, 2, 10], 10),
        "Aleatoria\n(10 objetos, W=80)": ([f"O{i}" for i in range(n)], vals, wts, 80),
    }

    resultados = {}
    print(f"{'Instancia':32s} {'Método':14s} {'Objetos':32s} {'Peso':>5s} {'Valor':>6s}")
    for nombre, (names, v, w, W) in instancias.items():
        limpio = nombre.replace("\n", " ")
        res = {}
        for metodo, f in [("Valor/peso", greedy_ratio), ("Menor peso", greedy_min_weight)]:
            ch, used, total = f(v, w, W)
            res[metodo] = (used, total)
            print(f"{limpio:32s} {metodo:14s} {str([names[i] for i in ch]):32s} {used:5d} {total:6d}")
        res["Óptimo (DP)"] = (None, optimo(v, w, W))
        print(f"{limpio:32s} {'Óptimo (DP)':14s} {'':32s} {'':>5s} {res['Óptimo (DP)'][1]:6d}")
        resultados[nombre] = (res, W)

    # Gráfica
    os.makedirs("figures", exist_ok=True)
    metodos = ["Valor/peso", "Menor peso", "Óptimo (DP)"]
    nombres = list(instancias)
    ancho = 0.25
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for j, m in enumerate(metodos):
        xs = [x + (j - 1) * ancho for x in range(len(nombres))]
        axs[0].bar(xs, [resultados[nm][0][m][1] for nm in nombres], ancho, label=m)
        if m != "Óptimo (DP)":
            axs[1].bar(xs, [100 * resultados[nm][0][m][0] / resultados[nm][1] for nm in nombres],
                       ancho, label=m)
    for ax, titulo, ylab in [(axs[0], "Valor total", "Valor"),
                             (axs[1], "Capacidad usada", "% de la capacidad")]:
        ax.set_xticks(range(len(nombres)))
        ax.set_xticklabels(nombres)
        ax.set_title(titulo)
        ax.set_ylabel(ylab)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/ex2_knapsack.png", dpi=150)
    print("\nGráfica guardada en figures/ex2_knapsack.png")

    print("\nContraejemplo: capacidad 10; A(w=1,v=1), B(w=2,v=2), C(w=10,v=100).")
    print("Menor peso toma A y B (valor 3) y C ya no cabe; valor/peso toma C (valor 100).")


if __name__ == "__main__":
    main()