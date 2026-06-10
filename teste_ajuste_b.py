from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    pasta_resultados = Path("resultados/enem_2009")

    caminho_parametros = (
        pasta_resultados /
        "parametros_comparados_3pl_mml_em.csv"
    )

    df = pd.read_csv(caminho_parametros)

    df = df.dropna(
        subset=[
            "B_REF",
            "B_EST"
        ]
    ).copy()

    # ============================================================
    # Ajuste por deslocamento de média
    # ============================================================

    media_b_ref = df["B_REF"].mean()
    media_b_est = df["B_EST"].mean()

    offset_b = media_b_ref - media_b_est

    df["B_EST_AJUSTADO"] = df["B_EST"] + offset_b

    # ============================================================
    # Métricas antes e depois
    # ============================================================

    corr_original = pearsonr(
        df["B_REF"],
        df["B_EST"]
    )[0]

    corr_ajustado = pearsonr(
        df["B_REF"],
        df["B_EST_AJUSTADO"]
    )[0]

    mae_original = mean_absolute_error(
        df["B_REF"],
        df["B_EST"]
    )

    rmse_original = rmse(
        df["B_REF"],
        df["B_EST"]
    )

    mae_ajustado = mean_absolute_error(
        df["B_REF"],
        df["B_EST_AJUSTADO"]
    )

    rmse_ajustado = rmse(
        df["B_REF"],
        df["B_EST_AJUSTADO"]
    )

    # ============================================================
    # Impressão dos resultados
    # ============================================================

    print("\n" + "=" * 80)
    print("TESTE DE AJUSTE DO PARÂMETRO B")
    print("=" * 80)

    print("\nMédias:")
    print(f"Média B_REF: {media_b_ref:.6f}")
    print(f"Média B_EST: {media_b_est:.6f}")
    print(f"Offset aplicado: {offset_b:.6f}")

    print("\nAntes do ajuste:")
    print(f"Correlação: {corr_original:.6f}")
    print(f"MAE       : {mae_original:.6f}")
    print(f"RMSE      : {rmse_original:.6f}")

    print("\nDepois do ajuste:")
    print(f"Correlação: {corr_ajustado:.6f}")
    print(f"MAE       : {mae_ajustado:.6f}")
    print(f"RMSE      : {rmse_ajustado:.6f}")

    # ============================================================
    # Salvar tabela ajustada
    # ============================================================

    caminho_saida_csv = (
        pasta_resultados /
        "teste_ajuste_b.csv"
    )

    df.to_csv(
        caminho_saida_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nTabela salva em: {caminho_saida_csv}")

    # ============================================================
    # Gráfico B_REF x B_EST original
    # ============================================================

    pasta_graficos = pasta_resultados / "graficos_diagnosticos"
    pasta_graficos.mkdir(parents=True, exist_ok=True)

    xmin = min(
        df["B_REF"].min(),
        df["B_EST"].min(),
        df["B_EST_AJUSTADO"].min()
    )

    xmax = max(
        df["B_REF"].max(),
        df["B_EST"].max(),
        df["B_EST_AJUSTADO"].max()
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df["B_REF"],
        df["B_EST"]
    )

    plt.plot(
        [xmin, xmax],
        [xmin, xmax],
        "--"
    )

    plt.xlabel("B_REF")
    plt.ylabel("B_EST")
    plt.title(
        f"B_REF x B_EST\n"
        f"Corr = {corr_original:.4f} | "
        f"MAE = {mae_original:.4f} | "
        f"RMSE = {rmse_original:.4f}"
    )

    plt.grid(True)
    plt.tight_layout()

    caminho_grafico_original = (
        pasta_graficos /
        "scatter_B_REF_B_EST_original.png"
    )

    plt.savefig(
        caminho_grafico_original,
        dpi=300
    )

    plt.close()

    # ============================================================
    # Gráfico B_REF x B_EST_AJUSTADO
    # ============================================================

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df["B_REF"],
        df["B_EST_AJUSTADO"]
    )

    plt.plot(
        [xmin, xmax],
        [xmin, xmax],
        "--"
    )

    plt.xlabel("B_REF")
    plt.ylabel("B_EST_AJUSTADO")
    plt.title(
        f"B_REF x B_EST_AJUSTADO\n"
        f"Corr = {corr_ajustado:.4f} | "
        f"MAE = {mae_ajustado:.4f} | "
        f"RMSE = {rmse_ajustado:.4f}"
    )

    plt.grid(True)
    plt.tight_layout()

    caminho_grafico_ajustado = (
        pasta_graficos /
        "scatter_B_REF_B_EST_ajustado.png"
    )

    plt.savefig(
        caminho_grafico_ajustado,
        dpi=300
    )

    plt.close()

    print(f"Gráfico original salvo em: {caminho_grafico_original}")
    print(f"Gráfico ajustado salvo em: {caminho_grafico_ajustado}")

    print("\n" + "=" * 80)
    print("FIM DO TESTE")
    print("=" * 80)


    resultado = linregress(
        df["B_EST"],
        df["B_REF"]
    )

    print(resultado.slope)
    print(resultado.intercept)
    print(resultado.rvalue)



if __name__ == "__main__":
    main()