import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def comparar_parametros(df_param_ref, df_param_est):
    """
    Compara parâmetros oficiais e estimados.

    Espera:
        df_param_ref:
            ITEM, A_REF, B_REF, C_REF

        df_param_est:
            ITEM, A_EST, B_EST, C_EST
    """

    df = df_param_ref.merge(
        df_param_est,
        on="ITEM",
        how="inner"
    )

    linhas = []

    pares = [
        ("A", "A_REF", "A_EST"),
        ("B", "B_REF", "B_EST"),
        ("C", "C_REF", "C_EST"),
    ]

    for nome, col_ref, col_est in pares:
        df_validos = df[[col_ref, col_est]].dropna()

        if len(df_validos) >= 2:
            corr = pearsonr(
                df_validos[col_ref],
                df_validos[col_est]
            )[0]

            mae = mean_absolute_error(
                df_validos[col_ref],
                df_validos[col_est]
            )

            erro_rmse = rmse(
                df_validos[col_ref],
                df_validos[col_est]
            )
        else:
            corr = np.nan
            mae = np.nan
            erro_rmse = np.nan

        linhas.append({
            "PARAMETRO": nome,
            "CORRELACAO": corr,
            "MAE": mae,
            "RMSE": erro_rmse
        })

    df_metricas = pd.DataFrame(linhas)

    return df, df_metricas