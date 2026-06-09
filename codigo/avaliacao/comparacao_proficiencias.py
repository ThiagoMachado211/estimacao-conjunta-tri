import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def comparar_proficiencias(df_proficiencias):
    """
    Compara nota real com theta EAP e número de acertos.
    """

    df = df_proficiencias.copy()

    resultados = []

    comparacoes = [
        ("THETA_EAP", "NOTA_REAL", "THETA_EAP"),
        ("N_ACERTOS", "NOTA_REAL", "N_ACERTOS"),
    ]

    for nome, col_real, col_pred in comparacoes:
        dados = df[[col_real, col_pred]].dropna()

        if len(dados) < 2:
            corr = np.nan
            mae = np.nan
            erro_rmse = np.nan
            r2 = np.nan
        else:
            corr = pearsonr(dados[col_real], dados[col_pred])[0]
            mae = mean_absolute_error(dados[col_real], dados[col_pred])
            erro_rmse = rmse(dados[col_real], dados[col_pred])
            r2 = r2_score(dados[col_real], dados[col_pred])

        resultados.append({
            "VARIAVEL": nome,
            "CORRELACAO_COM_NOTA_REAL": corr,
            "MAE": mae,
            "RMSE": erro_rmse,
            "R2": r2
        })

    return pd.DataFrame(resultados)