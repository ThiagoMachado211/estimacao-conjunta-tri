import numpy as np
import pandas as pd


def corrigir_resposta(resposta, gabarito):
    """
    Corrige uma única resposta.

    Retorna:
        1   = acerto
        0   = erro
        NaN = ausente/inválida
    """

    if pd.isna(resposta) or pd.isna(gabarito):
        return np.nan

    resposta = str(resposta).strip().upper()
    gabarito = str(gabarito).strip().upper()

    if resposta in ["", ".", "*", "P", "R"]:
        return np.nan

    if resposta not in ["A", "B", "C", "D", "E"]:
        return np.nan

    return 1 if resposta == gabarito else 0


def montar_matriz_respostas_enem(df_enem_area, id_disciplina):
    """
    Monta matriz 0/1/NaN a partir das respostas e gabaritos do ENEM.

    Entrada esperada:
        INSC
        TX_RESPOSTAS
        TX_GABARITO
        NOTA_REAL
        CO_PROVA
        TP_PRESENCA
        AREA

    Saída:
        INSC
        ID_DISCIPLINA
        Q1 ... Q45
        NOTA_REAL
        CO_PROVA
    """

    registros = []

    for _, linha in df_enem_area.iterrows():
        respostas = linha["TX_RESPOSTAS"]
        gabarito = linha["TX_GABARITO"]

        if pd.isna(respostas) or pd.isna(gabarito):
            continue

        respostas = str(respostas).strip().upper()
        gabarito = str(gabarito).strip().upper()

        n_itens = min(len(respostas), len(gabarito))

        registro = {
            "INSC": linha["INSC"],
            "ID_DISCIPLINA": id_disciplina,
            "NOTA_REAL": linha["NOTA_REAL"],
            "CO_PROVA": linha["CO_PROVA"],
        }

        for i in range(45):
            if i < n_itens:
                registro[f"Q{i + 1}"] = corrigir_resposta(
                    respostas[i],
                    gabarito[i]
                )
            else:
                registro[f"Q{i + 1}"] = np.nan

        registros.append(registro)

    df_matriz = pd.DataFrame(registros)

    colunas_q = [f"Q{i}" for i in range(1, 46)]

    colunas_finais = (
        ["INSC", "ID_DISCIPLINA"]
        + colunas_q
        + ["NOTA_REAL", "CO_PROVA"]
    )

    df_matriz = df_matriz[colunas_finais]

    return df_matriz


def resumo_matriz_respostas(df_matriz):
    """
    Gera resumo simples da matriz de respostas.
    """

    colunas_q = [c for c in df_matriz.columns if c.startswith("Q")]

    total_alunos = len(df_matriz)
    total_itens = len(colunas_q)

    respostas_validas = df_matriz[colunas_q].notna().sum().sum()
    respostas_ausentes = df_matriz[colunas_q].isna().sum().sum()

    acertos = (df_matriz[colunas_q] == 1).sum().sum()
    erros = (df_matriz[colunas_q] == 0).sum().sum()

    return {
        "total_alunos": total_alunos,
        "total_itens": total_itens,
        "respostas_validas": respostas_validas,
        "respostas_ausentes": respostas_ausentes,
        "acertos": acertos,
        "erros": erros,
    }