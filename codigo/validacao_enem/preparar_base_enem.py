def obter_colunas_area(area):
    """
    Retorna os nomes das colunas do ENEM associadas à área escolhida.

    area:
        CN, CH, LC ou MT
    """

    area = area.upper()

    if area not in ["CN", "CH", "LC", "MT"]:
        raise ValueError("Área inválida. Use: CN, CH, LC ou MT.")

    return {
        "respostas": f"TX_RESPOSTAS_{area}",
        "gabarito": f"TX_GABARITO_{area}",
        "nota": f"NU_NOTA_{area}",
        "codigo_prova": f"CO_PROVA_{area}",
        "presenca": f"TP_PRESENCA_{area}",
    }


def filtrar_microdados_area(
    df,
    area,
    codigos_prova=None,
    apenas_presentes=True
):
    """
    Filtra os microdados do ENEM por área, presença e código de prova.
    """

    colunas = obter_colunas_area(area)

    df_filtrado = df.copy()

    if apenas_presentes:
        df_filtrado = df_filtrado[
            df_filtrado[colunas["presenca"]] == 1
        ].copy()

    if codigos_prova is not None:
        df_filtrado = df_filtrado[
            df_filtrado[colunas["codigo_prova"]].isin(codigos_prova)
        ].copy()

    df_filtrado = df_filtrado[
        [
            "NU_INSCRICAO",
            colunas["respostas"],
            colunas["gabarito"],
            colunas["nota"],
            colunas["codigo_prova"],
            colunas["presenca"],
        ]
    ].copy()

    df_filtrado = df_filtrado.rename(
        columns={
            "NU_INSCRICAO": "INSC",
            colunas["respostas"]: "TX_RESPOSTAS",
            colunas["gabarito"]: "TX_GABARITO",
            colunas["nota"]: "NOTA_REAL",
            colunas["codigo_prova"]: "CO_PROVA",
            colunas["presenca"]: "TP_PRESENCA",
        }
    )

    df_filtrado["AREA"] = area.upper()

    return df_filtrado


def filtrar_itens_area(
    df_itens,
    area,
    codigos_prova=None
):
    """
    Filtra o arquivo de itens por área e código de prova.
    """

    area = area.upper()

    df_filtrado = df_itens[
        df_itens["SG_AREA"] == area
    ].copy()

    if codigos_prova is not None:
        df_filtrado = df_filtrado[
            df_filtrado["CO_PROVA"].isin(codigos_prova)
        ].copy()

    df_filtrado = df_filtrado.sort_values(
        ["CO_PROVA", "CO_POSICAO"]
    ).reset_index(drop=True)

    return df_filtrado