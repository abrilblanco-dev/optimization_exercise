import os

import matplotlib.pyplot as plt

def knapsack(values: list[int], weights: list[int], W: int):
    n = len(values)
    K = [[0] * (W + 1) for _ in range(n + 1)]   # K[i][c]: best value with items 1..i and capacity c
    for i in range(1, n + 1):
        v, w = values[i - 1], weights[i - 1]
        for c in range(W + 1):
            K[i][c] = K[i - 1][c]                                   # skip item i
            if w <= c:
                K[i][c] = max(K[i][c], v + K[i - 1][c - w])         # take item i
    chosen, c = [], W                           # walk back through the table
    for i in range(n, 0, -1):
        if K[i][c] != K[i - 1][c]:
            chosen.append(i - 1)
            c -= weights[i - 1]
    return K, K[n][W], chosen[::-1]


def seleccionar(orden, valores, pesos, capacidad):
    """Recorre los objetos en `orden` y mete cada uno si cabe."""
    objetos, peso_usado, valor_total = [], 0, 0
    for i in orden:
        if peso_usado + pesos[i] <= capacidad:
            objetos.append(i)
            peso_usado += pesos[i]
            valor_total += valores[i]
    return objetos, peso_usado, valor_total


def valor_peso(valores, pesos, capacidad):
    """Voraz original: mayor valor/peso primero."""
    orden = sorted(range(len(valores)), key=lambda i: valores[i] / pesos[i], reverse=True)
    return seleccionar(orden, valores, pesos, capacidad)


def menor_peso(valores, pesos, capacidad):
    """Modificación pedida: menor peso primero."""
    orden = sorted(range(len(pesos)), key=lambda i: pesos[i])
    return seleccionar(orden, valores, pesos, capacidad)


def main():
    instancias = [
        ("Ejemplo del profe", ["A", "B", "C"], [60, 100, 120], [10, 20, 30], 50),
        ("Contraejemplo", ["A", "B", "C"], [1, 2, 100], [1, 2, 10], 10),
    ]
    metodos = [("Valor/peso", valor_peso), ("Menor peso", menor_peso)]

    resultados = {}
    print(f"{'Instancia':18} {'Metodo':12} {'Objetos':14} {'Peso usado':>10} {'Valor':>6}")
    for titulo, nombres, valores, pesos, W in instancias:
        for nombre, algoritmo in metodos:
            sel, peso, valor = algoritmo(valores, pesos, W)
            print(f"{titulo:18} {nombre:12} {str([nombres[i] for i in sel]):14} "
                  f"{str(peso) + '/' + str(W):>10} {valor:6}")
            resultados[(titulo, nombre)] = (peso, W, valor)
        _, optimo, _ = knapsack(valores, pesos, W)
        print(f"{titulo:18} {'Óptimo (DP)':12} {'':14} {'':>10} {optimo:6}")

    # Gráfica de barras: valor total y capacidad usada, por instancia y método
    os.makedirs("figures", exist_ok=True)
    titulos = [t for t, *_ in instancias]
    ancho = 0.35
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for j, (nombre, _) in enumerate(metodos):
        xs = [x + (j - 0.5) * ancho for x in range(len(titulos))]
        axs[0].bar(xs, [resultados[(t, nombre)][2] for t in titulos], ancho, label=nombre)
        axs[1].bar(xs, [100 * resultados[(t, nombre)][0] / resultados[(t, nombre)][1] for t in titulos],
                   ancho, label=nombre)
    axs[0].set_title("Valor total")
    axs[0].set_ylabel("Valor")
    axs[1].set_title("Capacidad usada")
    axs[1].set_ylabel("% de la capacidad")
    for ax in axs:
        ax.set_xticks(range(len(titulos)))
        ax.set_xticklabels(titulos)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/ex2_knapsack.png", dpi=150)
    print("\nGrafica guardada en figures/ex2_knapsack.png")


if __name__ == "__main__":
    main()
