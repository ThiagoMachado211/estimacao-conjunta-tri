from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, linregress


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():

    caminho = (
        Path("resultados")
        / "enem_2009"
        / "parametros_comparados_3pl_mml_em.csv"
    )

    df = pd.read_csv(caminho)

    df = df.dropna(
        subset=["B_REF", "B_EST"]
    ).copy()

    print("\n" + "=" * 80)
    print("LINKING LINEAR DO PARÂMETRO B")
    print("=" * 80)

    # ----------------------------------------------------
    # Ajuste linear
    # ----------------------------------------------------

    resultado = linregress(
        df["B_EST"],
        df["B_REF"]
    )

    slope = resultado.slope
    intercept = resultado.intercept
    corr = resultado.rvalue

    print("\nRegressão linear:")
    print(f"Slope     = {slope:.12f}")
    print(f"Intercept = {intercept:.12f}")
    print(f"Correlação= {corr:.12f}")

    # ----------------------------------------------------
    # Aplicando linking
    # ----------------------------------------------------

    df["B_EST_LINK"] = (
        slope * df["B_EST"]
        + intercept
    )

    # ----------------------------------------------------
    # Métricas antes
    # ----------------------------------------------------

    mae_original = mean_absolute_error(
        df["B_REF"],
        df["B_EST"]
    )

    rmse_original = rmse(
        df["B_REF"],
        df["B_EST"]
    )

    corr_original = pearsonr(
        df["B_REF"],
        df["B_EST"]
    )[0]

    # ----------------------------------------------------
    # Métricas depois
    # ----------------------------------------------------

    mae_link = mean_absolute_error(
        df["B_REF"],
        df["B_EST_LINK"]
    )

    rmse_link = rmse(
        df["B_REF"],
        df["B_EST_LINK"]
    )

    corr_link = pearsonr(
        df["B_REF"],
        df["B_EST_LINK"]
    )[0]

    print("\n" + "-" * 80)
    print("ANTES DO LINKING")
    print("-" * 80)

    print(f"Correlação = {corr_original:.6f}")
    print(f"MAE        = {mae_original:.6f}")
    print(f"RMSE       = {rmse_original:.6f}")

    print("\n" + "-" * 80)
    print("DEPOIS DO LINKING")
    print("-" * 80)

    print(f"Correlação = {corr_link:.6f}")
    print(f"MAE        = {mae_link:.6f}")
    print(f"RMSE       = {rmse_link:.6f}")

    # ----------------------------------------------------
    # Top erros restantes
    # ----------------------------------------------------

    df["ABS_ERRO_LINK"] = (
        df["B_REF"] - df["B_EST_LINK"]
    ).abs()

    print("\n" + "-" * 80)
    print("TOP 10 MAIORES ERROS APÓS LINKING")
    print("-" * 80)

    print(
        df.sort_values(
            "ABS_ERRO_LINK",
            ascending=False
        )[
            [
                "ITEM",
                "CO_POSICAO",
                "B_REF",
                "B_EST",
                "B_EST_LINK",
                "ABS_ERRO_LINK"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # ----------------------------------------------------
    # Salvar resultado
    # ----------------------------------------------------

    caminho_saida = (
        Path("resultados")
        / "enem_2009"
        / "parametros_comparados_3pl_mml_em_link.csv"
    )

    df.to_csv(
        caminho_saida,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nArquivo salvo em:")
    print(caminho_saida)

    print("\n" + "=" * 80)
    print("FIM DO TESTE")
    print("=" * 80)


if __name__ == "__main__":
    main()