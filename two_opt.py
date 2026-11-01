import os
import random
import statistics

import matplotlib.pyplot as plt

from tsp_utils import nearest_neighbor, random_points, tour_length, two_opt


def main():
    random.seed(42)
    rng = random
    n_ciudades, n_instancias = 30, 30
    mej_rand, mej_nn, len_rand, len_nn = [], [], [], []

    for _ in range(n_instancias):
        points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(n_ciudades)]

        inicio = list(range(n_ciudades))
        random.shuffle(inicio)                              #solución inicial aleatoria
        tour, m = two_opt(points, inicio)
        mej_rand.append(m)
        len_rand.append(tour_length(points, tour))

        inicio = nearest_neighbor(points, random.randrange(n_ciudades))   #vecino más cercano
        tour, m = two_opt(points, inicio)
        mej_nn.append(m)
        len_nn.append(tour_length(points, tour))

    print(f"Instancias: {n_instancias} | ciudades por instancia: {n_ciudades}")
    print(f"Mejoras  - aleatorio: media {statistics.mean(mej_rand):.1f} (mediana {statistics.median(mej_rand):.0f})")
    print(f"Mejoras  - vecino más cercano: media {statistics.mean(mej_nn):.1f} (mediana {statistics.median(mej_nn):.0f})")
    print(f"Longitud - aleatorio: media {statistics.mean(len_rand):.2f} (mediana {statistics.median(len_rand):.2f})")
    print(f"Longitud - vecino más cercano: media {statistics.mean(len_nn):.2f} (mediana {statistics.median(len_nn):.2f})")
    print(f"NN necesitó menos mejoras en {sum(b < a for a, b in zip(mej_rand, mej_nn))}/{n_instancias} instancias")
    print(f"NN terminó con mejor tour en {sum(b < a for a, b in zip(len_rand, len_nn))}/{n_instancias} instancias")

    os.makedirs("figures", exist_ok=True)
    etiquetas = ["Inicio aleatorio", "Inicio vecino\nmás cercano"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    axs[0].boxplot([mej_rand, mej_nn])
    axs[0].set_xticklabels(etiquetas)
    axs[0].set_title("Número de mejoras hasta el óptimo local")
    axs[0].set_ylabel("Mejoras 2-opt")
    axs[1].boxplot([len_rand, len_nn])
    axs[1].set_xticklabels(etiquetas)
    axs[1].set_title("Longitud del tour final")
    axs[1].set_ylabel("Longitud")
    for ax in axs:
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/ex3_two_opt.png", dpi=150)
    print("Gráfica guardada en figures/ex3_two_opt.png")


if __name__ == "__main__":
    main()
