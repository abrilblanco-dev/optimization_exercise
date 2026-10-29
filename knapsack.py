
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def knapsack(values: list[int], weights: list[int], W: int):
    n = len(values)
    K = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        v, w = values[i - 1], weights[i - 1]

        for c in range(W + 1):
            K[i][c] = K[i - 1][c]

            if w <= c:
                K[i][c] = max(K[i][c], v + K[i - 1][c - w])

    chosen, c = [], W

    for i in range(n, 0, -1):
        if K[i][c] != K[i - 1][c]:
            chosen.append(i - 1)
            c -= weights[i - 1]

    return K, K[n][W], chosen[::-1]


def greedy(values: list[int], weights: list[int], W: int, criterio: str):
    n = len(values)

    if criterio == "valor/peso":
        orden = sorted(
            range(n),
            key=lambda i: values[i] / weights[i],
            reverse=True
        )
    else:
        orden = sorted(range(n), key=lambda i: weights[i])

    chosen = []
    used = 0
    total = 0

    for i in orden:
        if used + weights[i] <= W:
            chosen.append(i)
            used += weights[i]
            total += values[i]

    return chosen, used, total


def main():
    os.makedirs("figures", exist_ok=True)

    items = ["A", "B", "C"]
    values = [1, 2, 100]
    weights = [1, 2, 10]
    W = 10

    _, best, chosen = knapsack(values, weights, W)
    used = sum(weights[i] for i in chosen)

    resultados = [
        ("Valor/peso", *greedy(values, weights, W, "valor/peso")),
        ("Menor peso", *greedy(values, weights, W, "menor peso")),
        ("Optimo (DP)", chosen, used, best)
    ]

    print(f"Capacidad de la mochila: {W}\n")
    print(f"{'Metodo':15} {'Objetos':15} {'Peso':>5} {'Valor':>6}")

    for nombre, seleccion, peso, valor in resultados:
        objetos = [items[i] for i in seleccion]
        print(f"{nombre:15} {str(objetos):15} {peso:5d} {valor:6d}")

    metodos = [r[0] for r in resultados]
    valores_totales = [r[3] for r in resultados]
    pesos_usados = [r[2] for r in resultados]

    # Comparacion de valores
    plt.figure(figsize=(8, 5))
    plt.bar(metodos, valores_totales, label="Valor total")
    plt.ylabel("Valor")
    plt.title("Comparacion del valor obtenido")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/ex2_valores.png", dpi=150)
    plt.close()

    # Comparacion de capacidad utilizada
    plt.figure(figsize=(8, 5))
    plt.bar(metodos, pesos_usados, label="Peso utilizado")
    plt.axhline(W, color="green", linestyle=":", label="Capacidad maxima")
    plt.ylabel("Peso")
    plt.title("Comparacion de capacidad utilizada")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/ex2_capacidad.png", dpi=150)
    plt.close()

    print("\nGraficas guardadas en:")
    print("figures/ex2_valores.png")
    print("figures/ex2_capacidad.png")


if __name__ == "__main__":
    main()
