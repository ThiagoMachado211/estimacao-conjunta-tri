import pandas as pd


def ler_respostas_simulado(caminho):
    return pd.read_csv(
        caminho,
        sep=",",
        encoding="utf-8",
        low_memory=False
    )


def ler_notas_reais_simulado(caminho):
    return pd.read_csv(
        caminho,
        sep=";",
        encoding="utf-8",
        low_memory=False
    )


def ler_parametros_referencia_simulado(caminho):
    return pd.read_excel(
        caminho,
        sheet_name="Planilha1"
    )