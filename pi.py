import multiprocessing as mp
import os
import random
import statistics
import time

import matplotlib.pyplot as plt


def estimate_pi(seed: int, samples: int = 1_000_000) -> float:
    """Monte Carlo: fracción de puntos aleatorios dentro del cuarto de círculo unitario."""
    rng = random.Random(seed)
    inside = sum(rng.random() ** 2 + rng.random() ** 2 <= 1.0 for _ in range(samples))
    return 4 * inside / samples


def medir(p: int, seeds: list[int], repeticiones: int) -> float:
    """Devuelve la mediana del tiempo (s) de resolver todas las semillas con p procesos."""
    medidas = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        with mp.Pool(processes=p) as pool:
            pool.map(estimate_pi, seeds)
        medidas.append(time.perf_counter() - t0)
    return statistics.median(medidas)


def main():
    seeds = list(range(8))        # carga total fija: 8 corridas x 1M muestras
    procesos = [1, 2, 4, 8]
    repeticiones = 5

    print(f"Núcleos lógicos detectados: {mp.cpu_count()}")

    tiempos = {}
    for p in procesos:
        tiempos[p] = medir(p, seeds, repeticiones)
        print(f"p = {p}: {tiempos[p]:.3f} s")

    speedup = [tiempos[1] / tiempos[p] for p in procesos]
    eficiencia = [s / p for s, p in zip(speedup, procesos)]

    print("\n p | tiempo (s) | speedup | eficiencia")
    for p, s, e in zip(procesos, speedup, eficiencia):
        print(f"{p:2d} | {tiempos[p]:10.3f} | {s:7.2f} | {e:10.2f}")

    # Gráfica
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(procesos, speedup, "o-", label="Speedup medido")
    plt.plot(procesos, procesos, "--", label="Speedup ideal (lineal)")
    plt.xlabel("Número de procesos")
    plt.ylabel("Speedup  T(1) / T(p)")
    plt.title("Aceleración paralela de estimate_pi")
    plt.xticks(procesos)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/ex1_speedup.png", dpi=150)
    print("\nGráfica guardada en figures/ex1_speedup.png")


if __name__ == "__main__":
    main()