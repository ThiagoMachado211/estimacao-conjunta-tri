import pandas as pd


def filtrar_alunos_respostas_validas(
    df_matriz,
    minimo_respostas_validas=5
):
    """
    Remove alunos com poucas respostas válidas.

    Respostas válidas:
        0 = erro
        1 = acerto

    NaN é tratado como ausência/inválida.
    """

    colunas_q = [c for c in df_matriz.columns if c.startswith("Q")]

    df = df_matriz.copy()

    df["N_RESPOSTAS_VALIDAS"] = df[colunas_q].notna().sum(axis=1)
    df["N_ACERTOS"] = df[colunas_q].sum(axis=1, skipna=True)

    df_filtrado = df[
        df["N_RESPOSTAS_VALIDAS"] >= minimo_respostas_validas
    ].copy()

    return df_filtrado


def aplicar_amostragem(
    df,
    usar_amostra=True,
    tamanho_amostra=20000,
    semente=42
):
    """
    Aplica amostragem aleatória simples.
    """

    if not usar_amostra:
        return df.copy()

    if len(df) <= tamanho_amostra:
        return df.copy()

    return df.sample(
        n=tamanho_amostra,
        random_state=semente
    ).copy()