from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def grafico_dispersao_parametro(
    df,
    coluna_ref,
    coluna_est,
    titulo,
    caminho_saida
):
    """
    Gera gráfico de dispersão entre parâmetro oficial e estimado.
    """

    dados = df[[coluna_ref, coluna_est]].dropna()

    if len(dados) == 0:
        print(f"Nenhum dado disponível para {titulo}")
        return

    x = dados[coluna_ref]
    y = dados[coluna_est]

    plt.figure(figsize=(8, 6))

    plt.scatter(x, y)

    minimo = min(x.min(), y.min())
    maximo = max(x.max(), y.max())

    plt.plot(
        [minimo, maximo],
        [minimo, maximo],
        linestyle="--"
    )

    plt.xlabel(coluna_ref)
    plt.ylabel(coluna_est)
    plt.title(titulo)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(caminho_saida, dpi=300)

    plt.close()