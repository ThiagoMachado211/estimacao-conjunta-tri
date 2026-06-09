import numpy as np


def gerar_diagnostico_itens(df_param_comparados):
    """
    Cria tabela detalhada item a item.
    """

    df = df_param_comparados.copy()

    df["ERRO_A"] = (
        df["A_EST"] - df["A_REF"]
    )

    df["ERRO_B"] = (
        df["B_EST"] - df["B_REF"]
    )

    df["ERRO_C"] = (
        df["C_EST"] - df["C_REF"]
    )

    df["ABS_ERRO_A"] = np.abs(df["ERRO_A"])
    df["ABS_ERRO_B"] = np.abs(df["ERRO_B"])
    df["ABS_ERRO_C"] = np.abs(df["ERRO_C"])

    return df