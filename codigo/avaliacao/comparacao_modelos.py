import pandas as pd


def adicionar_resultado_modelo(
    resultados,
    nome_modelo,
    metricas_parametros,
    corr_nota_theta,
    corr_theta_ref
):
    linha = {
        "MODELO": nome_modelo,
        "CORR_A": None,
        "CORR_B": None,
        "CORR_C": None,
        "CORR_NOTA_THETA": corr_nota_theta,
        "CORR_THETA_REF": corr_theta_ref,
    }

    for _, row in metricas_parametros.iterrows():

        parametro = row["PARAMETRO"]
        corr = row["CORRELACAO"]

        if parametro == "A":
            linha["CORR_A"] = corr

        elif parametro == "B":
            linha["CORR_B"] = corr

        elif parametro == "C":
            linha["CORR_C"] = corr

    resultados.append(linha)

    return resultados


def criar_tabela_comparativa(resultados):
    df = pd.DataFrame(resultados)

    colunas = [
        "MODELO",
        "CORR_A",
        "CORR_B",
        "CORR_C",
        "CORR_NOTA_THETA",
        "CORR_THETA_REF"
    ]

    return df[colunas]