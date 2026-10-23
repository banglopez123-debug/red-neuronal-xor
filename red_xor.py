from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Los cuatro casos posibles de XOR.
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

# Resultado correcto de cada caso.
y = np.array([[0], [1], [1], [0]], dtype=float)


def sigmoide(z):
    """Convierte los valores en salidas entre 0 y 1."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def entrenar(tasa, epocas=5000, semilla=42):
    """Entrena una red de 2 entradas, 4 neuronas ocultas y 1 salida."""

    # La misma semilla permite comenzar ambos experimentos
    # con exactamente los mismos pesos.
    rng = np.random.default_rng(semilla)

    W1 = rng.normal(0, 0.5, (2, 4))
    b1 = np.zeros((1, 4))

    W2 = rng.normal(0, 0.5, (4, 1))
    b2 = np.zeros((1, 1))

    perdidas = []
    n = len(X)

    # Incluye la pérdida inicial, antes de actualizar los pesos.
    for epoca in range(epocas + 1):

        # 1. FORWARD PASS: calcular las predicciones.
        oculta = np.tanh(X @ W1 + b1)
        logits = oculta @ W2 + b2
        salida = sigmoide(logits)

        # 2. ERROR: entropía cruzada binaria.
        # logaddexp evita problemas numéricos con los logaritmos.
        perdida = np.mean(np.logaddexp(0, logits) - y * logits)
        perdidas.append(float(perdida))

        if epoca % 1000 == 0:
            print(
                f"Tasa {tasa} | Época {epoca:4d} | "
                f"Pérdida: {perdida:.6f}"
            )

        if epoca == epocas:
            break

        # 3. BACKPROPAGATION: calcular los gradientes.
        # Derivada de entropía cruzada con salida sigmoide.
        delta_salida = (salida - y) / n

        grad_W2 = oculta.T @ delta_salida
        grad_b2 = np.sum(delta_salida, axis=0, keepdims=True)

        # Derivada de tanh: 1 - tanh(z)^2.
        delta_oculta = (delta_salida @ W2.T) * (1 - oculta ** 2)

        grad_W1 = X.T @ delta_oculta
        grad_b1 = np.sum(delta_oculta, axis=0, keepdims=True)

        # 4. ACTUALIZACIÓN: descenso de gradiente.
        W1 -= tasa * grad_W1
        b1 -= tasa * grad_b1
        W2 -= tasa * grad_W2
        b2 -= tasa * grad_b2

    clases = (salida >= 0.5).astype(int)
    exactitud = float(np.mean(clases == y) * 100)

    return {
        "tasa": tasa,
        "perdidas": perdidas,
        "salidas": salida,
        "clases": clases,
        "exactitud": exactitud
    }


def main():
    # Los archivos se guardan junto al programa.
    carpeta = Path(__file__).resolve().parent / "resultados"
    carpeta.mkdir(exist_ok=True)

    experimentos = []

    for tasa in [0.1, 1.0]:
        print(f"\nENTRENAMIENTO CON TASA DE APRENDIZAJE {tasa}")
        experimentos.append(entrenar(tasa))

    # Mostrar y guardar los resultados de ambos experimentos.
    resumen = [
        "RED NEURONAL PARA XOR",
        "Arquitectura: 2 entradas, 4 neuronas ocultas y 1 salida.",
        "Activaciones: tanh y sigmoide.",
        "Épocas: 5000. Semilla: 42.",
        "Evaluación sobre los mismos cuatro casos de entrenamiento."
    ]

    for resultado in experimentos:
        resumen.append(f"\nTasa de aprendizaje: {resultado['tasa']}")
        resumen.append(
            f"Pérdida final: {resultado['perdidas'][-1]:.6f}"
        )
        resumen.append(
            f"Exactitud: {resultado['exactitud']:.2f}%"
        )
        resumen.append("Entrada | Esperado | Salida de la red | Clase")

        for entrada, esperado, salida, clase in zip(
            X, y, resultado["salidas"], resultado["clases"]
        ):
            resumen.append(
                f"{entrada.astype(int).tolist()} | "
                f"{int(esperado[0])} | "
                f"{salida[0]:.6f} | {int(clase[0])}"
            )

    texto = "\n".join(resumen)
    print("\n" + texto)

    (carpeta / "resumen_resultados.txt").write_text(
        texto, encoding="utf-8"
    )

    # Guardar las pérdidas para conservar los datos de la gráfica.
    datos = np.column_stack([
        np.arange(len(experimentos[0]["perdidas"])),
        experimentos[0]["perdidas"],
        experimentos[1]["perdidas"]
    ])

    np.savetxt(
        carpeta / "perdidas.csv",
        datos,
        delimiter=",",
        header="epoca,perdida_tasa_0.1,perdida_tasa_1.0",
        comments=""
    )

    # Comparación de las curvas de pérdida.
    plt.figure(figsize=(9, 5))

    for resultado in experimentos:
        plt.plot(
            resultado["perdidas"],
            label=f"Tasa de aprendizaje = {resultado['tasa']}"
        )

    plt.title("Aprendizaje de XOR: comparación de experimentos")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida (entropía cruzada binaria)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(carpeta / "comparacion_perdida.png", dpi=200)

    print(f"\nArchivos guardados en: {carpeta}")
    plt.show()


if __name__ == "__main__":
    main()