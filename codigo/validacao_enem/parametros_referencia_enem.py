import pandas as pd


def preparar_parametros_referencia_enem(df_itens_area):
    """
    Prepara os parâmetros oficiais dos itens do ENEM na ordem Q1...Q45.

    Entrada esperada:
        CO_PROVA
        CO_POSICAO
        SG_AREA
        NU_PARAM_A
        NU_PARAM_B
        NU_PARAM_C
        TX_GABARITO

    Saída:
        ITEM
        CO_PROVA
        CO_POSICAO
        A_REF
        B_REF
        C_REF
        GABARITO_REF
    """

    df = df_itens_area.copy()

    df = df.sort_values("CO_POSICAO").reset_index(drop=True)

    df["ITEM"] = [f"Q{i}" for i in range(1, len(df) + 1)]

    df_param = df[
        [
            "ITEM",
            "CO_PROVA",
            "CO_POSICAO",
            "NU_PARAM_A",
            "NU_PARAM_B",
            "NU_PARAM_C",
            "TX_GABARITO",
        ]
    ].copy()

    df_param = df_param.rename(
        columns={
            "NU_PARAM_A": "A_REF",
            "NU_PARAM_B": "B_REF",
            "NU_PARAM_C": "C_REF",
            "TX_GABARITO": "GABARITO_REF",
        }
    )

    return df_param


def verificar_parametros_referencia(df_param):
    """
    Faz verificações básicas nos parâmetros oficiais.
    """

    problemas = []

    if len(df_param) != 45:
        problemas.append(
            f"Quantidade de itens diferente de 45: {len(df_param)}"
        )

    if df_param["ITEM"].duplicated().any():
        problemas.append("Existem itens duplicados em ITEM.")

    if df_param["CO_POSICAO"].duplicated().any():
        problemas.append("Existem posições duplicadas em CO_POSICAO.")

    for col in ["A_REF", "B_REF", "C_REF"]:
        if df_param[col].isna().any():
            problemas.append(f"Existem valores ausentes em {col}.")

    return problemas



def comparar_gabaritos_enem(df_enem_area, df_param_ref):
    """
    Compara o gabarito presente nos microdados com o gabarito
    do arquivo oficial de itens.
    """

    gabaritos_microdados = (
        df_enem_area["TX_GABARITO"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .unique()
    )

    if len(gabaritos_microdados) == 0:
        return {
            "status": "erro",
            "mensagem": "Nenhum gabarito encontrado nos microdados.",
            "total_diferencas": None,
            "diferencas": None,
        }

    if len(gabaritos_microdados) > 1:
        return {
            "status": "erro",
            "mensagem": f"Foram encontrados {len(gabaritos_microdados)} gabaritos diferentes nos microdados.",
            "total_diferencas": None,
            "diferencas": gabaritos_microdados,
        }

    gabarito_micro = gabaritos_microdados[0]

    df_ord = df_param_ref.sort_values("CO_POSICAO").copy()

    gabarito_itens = "".join(
        df_ord["GABARITO_REF"]
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    n = min(len(gabarito_micro), len(gabarito_itens))

    diferencas = []

    for i in range(n):
        if gabarito_micro[i] != gabarito_itens[i]:
            diferencas.append(
                {
                    "ITEM": f"Q{i + 1}",
                    "POSICAO": df_ord.iloc[i]["CO_POSICAO"],
                    "GABARITO_MICRODADOS": gabarito_micro[i],
                    "GABARITO_ITENS": gabarito_itens[i],
                }
            )

    return {
        "status": "ok" if len(diferencas) == 0 else "divergente",
        "mensagem": "Comparação concluída.",
        "tamanho_gabarito_microdados": len(gabarito_micro),
        "tamanho_gabarito_itens": len(gabarito_itens),
        "total_diferencas": len(diferencas),
        "diferencas": diferencas,
    }