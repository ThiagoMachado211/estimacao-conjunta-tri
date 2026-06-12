import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from codigo.configuracoes import (
    ARQUIVO_RESPOSTAS_DIA1,
    ARQUIVO_RESPOSTAS_DIA2,
    ARQUIVO_NOTAS_REAIS_SIMULADO,
    ARQUIVO_PARAMETROS_REFERENCIA_SIMULADO,
    PASTA_RESULTADOS_SIMULADO,
    ID_DISCIPLINA_SIMULADO_EXECUTAR,
    SERIE_SIMULADO_EXECUTAR,
    TAMANHO_AMOSTRA_SIMULADO,
    SEMENTE,
    THETA_MIN,
    THETA_MAX,
    N_PONTOS_THETA,
    MEDIA_PRIOR,
    DESVIO_PRIOR,
)

from codigo.leitura.leitura_simulado import (
    ler_respostas_simulado,
    ler_notas_reais_simulado,
    ler_parametros_referencia_simulado,
)

from codigo.preprocessamento.matriz_respostas import (
    montar_matriz_respostas_simulado,
    resumo_matriz_respostas,
)

from codigo.preprocessamento.amostragem import (
    filtrar_alunos_respostas_validas,
    aplicar_amostragem,
)

from codigo.calibracao.calibracao_3pl_mml_em import (
    calibrar_itens_3pl_mml_em,
)


def classificar_grupo_item(df_param):
    df = df_param.copy()

    df["GRUPO_ITEM"] = "INEDITA"

    df.loc[
        df["ORIGEM"].astype(str).str.contains("ENEM", case=False, na=False),
        "GRUPO_ITEM",
    ] = "ENEM"

    df.loc[
        df["ORIGEM"].astype(str).str.contains("ERRO", case=False, na=False),
        "GRUPO_ITEM",
    ] = "ERRO"

    return df


def preparar_parametros_area_simulado(df_param_ref, id_disciplina):
    mapa_area = {
        1: "CH",
        2: "CN",
        3: "LC",
        4: "MT",
    }

    area = mapa_area[id_disciplina]

    if area == "LC":
        df_area = df_param_ref[
            df_param_ref["GRANDE_AREA"].isin(
                ["LE - INGLÊS", "LE - ESPANHOL", "LCT"]
            )
        ].copy()
    else:
        df_area = df_param_ref[
            df_param_ref["GRANDE_AREA"] == area
        ].copy()

    df_area = (
        df_area
        .sort_values("POSICAO_PROVA")
        .reset_index(drop=True)
    )

    df_area["ITEM"] = [
        f"Q{i}"
        for i in range(1, len(df_area) + 1)
    ]

    df_area = df_area.rename(
        columns={
            "NU_PARAM_A": "A_REF",
            "NU_PARAM_B": "B_REF",
            "NU_PARAM_C": "C_REF",
            "GAB": "GABARITO_REF",
        }
    )

    return df_area


def juntar_notas_reais(df_matriz_amostra, df_notas, id_disciplina):
    df_notas_area = df_notas[
        df_notas["ID_DISCIPLINA"] == id_disciplina
    ].copy()

    df_notas_area = df_notas_area.rename(
        columns={"NOTA_TRI": "NOTA_REAL"}
    )

    df_notas_area["NOTA_REAL"] = (
        df_notas_area["NOTA_REAL"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )

    df_notas_area["NOTA_REAL"] = pd.to_numeric(
        df_notas_area["NOTA_REAL"],
        errors="coerce"
    )

    df = df_matriz_amostra.merge(
        df_notas_area[["INSC", "NOTA_REAL"]],
        on="INSC",
        how="left"
    )

    return df


def calcular_metricas_parametros(df, nome_grupo):
    print("--------------------------------------------------------")
    print(f"\nMÉTRICAS - {nome_grupo}")

    for parametro in ["A", "B", "C"]:
        col_ref = f"{parametro}_REF"
        col_est = f"{parametro}_EST"

        dados = df[[col_ref, col_est]].dropna()

        if len(dados) < 2:
            print(f"\nParâmetro {parametro}")
            print("Métricas não calculáveis.")
            continue

        corr = dados[col_ref].corr(dados[col_est])

        mae = mean_absolute_error(
            dados[col_ref],
            dados[col_est]
        )

        rmse = np.sqrt(
            mean_squared_error(
                dados[col_ref],
                dados[col_est]
            )
        )

        print(f"\nParâmetro {parametro}")
        print(f"Corr = {corr:.6f}")
        print(f"MAE  = {mae:.6f}")
        print(f"RMSE = {rmse:.6f}")


def main():

    print("=" * 80)
    print("EXPERIMENTO A - SIMULADO")
    print("=" * 80)

    print("\nConfiguração:")
    print(f"Série/modalidade: {SERIE_SIMULADO_EXECUTAR}")

    nome_grupo = (
        SERIE_SIMULADO_EXECUTAR
        .replace(" ", "_")
        .replace("º", "")
        .replace("-", "_")
    )

    print(f"ID disciplina: {ID_DISCIPLINA_SIMULADO_EXECUTAR}")
    print(f"Amostra: {TAMANHO_AMOSTRA_SIMULADO}")

    PASTA_RESULTADOS_SIMULADO.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nLendo respostas do simulado...")

    if ID_DISCIPLINA_SIMULADO_EXECUTAR in [2, 4]:
        arquivo_respostas = ARQUIVO_RESPOSTAS_DIA2
    else:
        arquivo_respostas = ARQUIVO_RESPOSTAS_DIA1

    df_respostas = ler_respostas_simulado(
        arquivo_respostas
    )

    print(f"Respostas lidas: {df_respostas.shape}")

    print("\nLendo notas reais...")

    df_notas = ler_notas_reais_simulado(
        ARQUIVO_NOTAS_REAIS_SIMULADO
    )

    print(f"Notas lidas: {df_notas.shape}")
    print(df_notas.head())

    print("\nLendo parâmetros de referência...")

    df_param_ref = ler_parametros_referencia_simulado(
        ARQUIVO_PARAMETROS_REFERENCIA_SIMULADO
    )

    print(f"Parâmetros lidos: {df_param_ref.shape}")

    print("--------------------------------------------------------")
    print("\nFiltrando série/modalidade...")

    df_respostas_filtradas = df_respostas[
        df_respostas["DESC_SERIE"] == SERIE_SIMULADO_EXECUTAR
    ].copy()

    print(f"Respostas após filtro de série: {df_respostas_filtradas.shape}")

    print("\nFiltrando disciplina e montando matriz 0/1/NaN...")

    df_matriz = montar_matriz_respostas_simulado(
        df_respostas_filtradas,
        id_disciplina=ID_DISCIPLINA_SIMULADO_EXECUTAR,
    )

    print("Matriz criada com sucesso.")
    print(f"Linhas: {df_matriz.shape[0]}")
    print(f"Colunas: {df_matriz.shape[1]}")
    print(df_matriz.head())

    print("\nResumo da matriz:")

    resumo = resumo_matriz_respostas(df_matriz)

    for chave, valor in resumo.items():
        print(f"{chave}: {valor}")

    print("--------------------------------------------------------")
    print("\nFiltrando alunos com poucas respostas válidas...")

    df_matriz_filtrada = filtrar_alunos_respostas_validas(
        df_matriz,
        minimo_respostas_validas=35
    )

    print(f"Antes do filtro: {len(df_matriz)}")
    print(f"Depois do filtro: {len(df_matriz_filtrada)}")

    print("\nAplicando amostragem...")

    df_matriz_amostra = aplicar_amostragem(
        df_matriz_filtrada,
        usar_amostra=True,
        tamanho_amostra=TAMANHO_AMOSTRA_SIMULADO,
        semente=SEMENTE
    )

    print(f"Alunos após amostragem: {len(df_matriz_amostra)}")
    print(df_matriz_amostra.head())

    print("--------------------------------------------------------")
    print("\nJuntando notas reais...")

    df_matriz_amostra = juntar_notas_reais(
        df_matriz_amostra,
        df_notas,
        ID_DISCIPLINA_SIMULADO_EXECUTAR
    )

    print("Resumo NOTA_REAL:")
    print(df_matriz_amostra["NOTA_REAL"].describe())
    print("Notas ausentes:", df_matriz_amostra["NOTA_REAL"].isna().sum())

    print("--------------------------------------------------------")
    print("\nDIAGNÓSTICO DA MATRIZ")

    colunas_q = [
        c for c in df_matriz_amostra.columns
        if c.startswith("Q")
    ]

    print(f"Quantidade de itens: {len(colunas_q)}")

    print("\nPrimeiras linhas:")
    print(
        df_matriz_amostra[
            ["INSC"] + colunas_q[:10]
        ].head()
    )

    print("\nValores únicos encontrados:")

    for q in colunas_q[:5]:
        print(
            q,
            sorted(
                df_matriz_amostra[q]
                .dropna()
                .unique()
                .tolist()
            )
        )

    print("--------------------------------------------------------")
    print("\nFREQUÊNCIA DE RESPOSTAS")

    n_validas = (
        df_matriz_amostra[colunas_q]
        .notna()
        .sum(axis=1)
    )

    print(n_validas.describe())

    print("--------------------------------------------------------")
    print("\nClassificando itens por origem...")

    df_param_ref = classificar_grupo_item(df_param_ref)

    print("\nQuantidade de itens por GRUPO_ITEM:")
    print(
        df_param_ref["GRUPO_ITEM"]
        .value_counts()
    )

    print("--------------------------------------------------------")
    print("\nPreparando parâmetros da área executada...")

    df_param_area = preparar_parametros_area_simulado(
        df_param_ref,
        ID_DISCIPLINA_SIMULADO_EXECUTAR
    )

    print(df_param_area.head())
    print(df_param_area.tail())

    print("\nComposição da área:")
    print(
        df_param_area["GRUPO_ITEM"]
        .value_counts()
    )

    caminho_param_area = (
        PASTA_RESULTADOS_SIMULADO /
        "parametros_referencia_area_experimento_a.csv"
    )

    df_param_area.to_csv(
        caminho_param_area,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nParâmetros da área salvos em: {caminho_param_area}")

    print("--------------------------------------------------------")
    print("\nFrequência de acerto por item...")

    freq_acerto = (
        df_matriz_amostra[colunas_q]
        .mean()
        .reset_index()
    )

    freq_acerto.columns = [
        "ITEM",
        "P_ACERTO"
    ]

    print(freq_acerto["P_ACERTO"].describe())

    print("\n10 itens com maior percentual de acerto:")
    print(
        freq_acerto
        .sort_values("P_ACERTO")
        .tail(10)
    )

    print("\n10 itens com menor percentual de acerto:")
    print(
        freq_acerto
        .sort_values("P_ACERTO")
        .head(10)
    )

    caminho_freq = (
        PASTA_RESULTADOS_SIMULADO /
        "frequencia_acerto_experimento_a.csv"
    )

    freq_acerto.to_csv(
        caminho_freq,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nFrequências salvas em: {caminho_freq}")

    print("--------------------------------------------------------")
    print("\nANÁLISE DOS PARÂMETROS OFICIAIS x FREQUÊNCIA OBSERVADA")

    df_freq_param = df_param_area.merge(
        freq_acerto,
        on="ITEM",
        how="inner"
    )

    print("\nPrimeiras linhas:")
    print(
        df_freq_param[
            [
                "ITEM",
                "GRUPO_ITEM",
                "ORIGEM",
                "A_REF",
                "B_REF",
                "C_REF",
                "P_ACERTO"
            ]
        ].head()
    )

    corr_a = df_freq_param["A_REF"].corr(df_freq_param["P_ACERTO"])
    corr_b = df_freq_param["B_REF"].corr(df_freq_param["P_ACERTO"])
    corr_c = df_freq_param["C_REF"].corr(df_freq_param["P_ACERTO"])

    print("\nCorrelações gerais:")
    print(f"Corr(A_REF, P_ACERTO) = {corr_a:.6f}")
    print(f"Corr(B_REF, P_ACERTO) = {corr_b:.6f}")
    print(f"Corr(C_REF, P_ACERTO) = {corr_c:.6f}")

    print("--------------------------------------------------------")
    print("\nANÁLISE DOS ITENS ENEM POR ANO")

    df_enem_freq = df_freq_param[
        df_freq_param["GRUPO_ITEM"] == "ENEM"
    ].copy()

    df_enem_freq["ANO_ENEM"] = (
        df_enem_freq["ORIGEM"]
        .astype(str)
        .str.extract(r"(20\d{2})")
    )

    resultados_ano = []

    for ano in sorted(df_enem_freq["ANO_ENEM"].dropna().unique()):

        temp = df_enem_freq[
            df_enem_freq["ANO_ENEM"] == ano
        ].copy()

        n_itens = len(temp)

        if n_itens < 2:
            continue

        resultados_ano.append({
            "ANO": ano,
            "N_ITENS": n_itens,
            "MEDIA_P_ACERTO": temp["P_ACERTO"].mean(),
            "MEDIA_A_REF": temp["A_REF"].mean(),
            "MEDIA_B_REF": temp["B_REF"].mean(),
            "MEDIA_C_REF": temp["C_REF"].mean(),
            "CORR_A": temp["A_REF"].corr(temp["P_ACERTO"]),
            "CORR_B": temp["B_REF"].corr(temp["P_ACERTO"]),
            "CORR_C": temp["C_REF"].corr(temp["P_ACERTO"]),
        })

    df_resultados_ano = pd.DataFrame(resultados_ano)

    print(df_resultados_ano)

    caminho_ano = (
        PASTA_RESULTADOS_SIMULADO /
        "analise_enem_por_ano.csv"
    )

    df_resultados_ano.to_csv(
        caminho_ano,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nResultado salvo em:\n{caminho_ano}")

    print("--------------------------------------------------------")
    print("\nQUANTIDADE DE ITENS ENEM POR ANO")

    print(
        df_enem_freq["ANO_ENEM"]
        .value_counts()
        .sort_index()
    )


    print("--------------------------------------------------------")
    print("\nDETALHAMENTO DOS ITENS ENEM DE 2019 E 2020")

    df_2019_2020 = df_enem_freq[
        df_enem_freq["ANO_ENEM"].isin(["2019", "2020"])
    ].copy()

    colunas_detalhe = [
        "ITEM",
        "POSICAO_PROVA",
        "ORIGEM",
        "ANO_ENEM",
        "A_REF",
        "B_REF",
        "C_REF",
        "P_ACERTO"
    ]

    print(
        df_2019_2020[
            colunas_detalhe
        ]
        .sort_values(["ANO_ENEM", "B_REF"])
        .to_string(index=False)
    )

    caminho_2019_2020 = (
        PASTA_RESULTADOS_SIMULADO /
        "detalhamento_itens_enem_2019_2020.csv"
    )

    df_2019_2020.to_csv(
        caminho_2019_2020,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nDetalhamento 2019/2020 salvo em:\n{caminho_2019_2020}")


    print("--------------------------------------------------------")
    print("\nDETALHAMENTO DOS ITENS ENEM DE 2021 E 2022")

    df_2021_2022 = df_enem_freq[
        df_enem_freq["ANO_ENEM"].isin(["2021", "2022"])
    ].copy()

    print(
        df_2021_2022[
            colunas_detalhe
        ]
        .sort_values(["ANO_ENEM", "B_REF"])
        .to_string(index=False)
    )

    caminho_2021_2022 = (
        PASTA_RESULTADOS_SIMULADO /
        "detalhamento_itens_enem_2021_2022.csv"
    )

    df_2021_2022.to_csv(
        caminho_2021_2022,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nDetalhamento 2021/2022 salvo em:\n{caminho_2021_2022}")



    print("--------------------------------------------------------")
    print("\nDistribuição de acertos por aluno...")

    df_matriz_amostra["N_ACERTOS"] = (
        df_matriz_amostra[colunas_q]
        .sum(axis=1)
    )

    print(df_matriz_amostra["N_ACERTOS"].describe())

    dados_corr = df_matriz_amostra[
        ["N_ACERTOS", "NOTA_REAL"]
    ].copy()

    dados_corr["N_ACERTOS"] = pd.to_numeric(
        dados_corr["N_ACERTOS"],
        errors="coerce"
    )

    dados_corr["NOTA_REAL"] = pd.to_numeric(
        dados_corr["NOTA_REAL"],
        errors="coerce"
    )

    dados_corr = dados_corr.dropna()

    if len(dados_corr) >= 2:
        corr_acertos_nota = dados_corr.corr().iloc[0, 1]
    else:
        corr_acertos_nota = None

    print("\nCorrelação N_ACERTOS x NOTA_REAL:")

    if corr_acertos_nota is None:
        print("Não calculável.")
    else:
        print(f"{corr_acertos_nota:.6f}")

    caminho_matriz_amostra = (
        PASTA_RESULTADOS_SIMULADO /
        "matriz_amostra_experimento_a.csv"
    )

    df_matriz_amostra.to_csv(
        caminho_matriz_amostra,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nMatriz amostral salva em: {caminho_matriz_amostra}")

    print("--------------------------------------------------------")
    print("\nCALIBRAÇÃO 3PL MML-EM - 45 ITENS")

    parametros_estimados = calibrar_itens_3pl_mml_em(
        df_matriz_amostra,
        theta_min=THETA_MIN,
        theta_max=THETA_MAX,
        n_pontos=N_PONTOS_THETA,
        media_prior=MEDIA_PRIOR,
        desvio_prior=DESVIO_PRIOR,
        max_iter=20,
        tol=1e-4,
    )

    caminho_parametros_estimados = (
        PASTA_RESULTADOS_SIMULADO /
        "parametros_estimados_experimento_a.csv"
    )

    parametros_estimados.to_csv(
        caminho_parametros_estimados,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nParâmetros salvos em:\n{caminho_parametros_estimados}")

    df_comparacao = df_param_area.merge(
        parametros_estimados,
        on="ITEM",
        how="inner"
    )

    caminho_comparacao = (
        PASTA_RESULTADOS_SIMULADO /
        "comparacao_parametros_experimento_a.csv"
    )

    df_comparacao.to_csv(
        caminho_comparacao,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nQuantidade de itens comparados:")
    print(len(df_comparacao))

    print("\nComposição:")
    print(
        df_comparacao["GRUPO_ITEM"]
        .value_counts()
    )

    calcular_metricas_parametros(
        df_comparacao[df_comparacao["GRUPO_ITEM"] == "ENEM"].copy(),
        "ITENS ENEM - 45 ITENS"
    )

    calcular_metricas_parametros(
        df_comparacao[df_comparacao["GRUPO_ITEM"] == "INEDITA"].copy(),
        "ITENS INÉDITOS - 45 ITENS"
    )

    print("--------------------------------------------------------")
    print("\nEXPERIMENTO A.1 - SOMENTE ITENS ENEM")

    df_param_enem = df_param_area[
        df_param_area["GRUPO_ITEM"] == "ENEM"
    ].copy()

    itens_enem = df_param_enem["ITEM"].tolist()

    print(f"Quantidade de itens ENEM: {len(itens_enem)}")
    print("Itens ENEM:")
    print(itens_enem)

    colunas_base = ["INSC", "ID_DISCIPLINA", "NOTA_REAL"]

    colunas_base = [
        c for c in colunas_base
        if c in df_matriz_amostra.columns
    ]

    df_matriz_enem = df_matriz_amostra[
        colunas_base + itens_enem
    ].copy()

    print("\nMatriz somente ENEM:")
    print(df_matriz_enem.shape)

    print("\nFrequência de acerto dos itens ENEM:")
    freq_enem = (
        df_matriz_enem[itens_enem]
        .mean()
        .reset_index()
    )

    freq_enem.columns = ["ITEM", "P_ACERTO"]

    print(freq_enem["P_ACERTO"].describe())

    caminho_freq_enem = (
        PASTA_RESULTADOS_SIMULADO /
        "frequencia_acerto_somente_enem_experimento_a.csv"
    )

    freq_enem.to_csv(
        caminho_freq_enem,
        index=False,
        encoding="utf-8-sig"
    )

    parametros_estimados_enem = calibrar_itens_3pl_mml_em(
        df_matriz_enem,
        theta_min=THETA_MIN,
        theta_max=THETA_MAX,
        n_pontos=N_PONTOS_THETA,
        media_prior=MEDIA_PRIOR,
        desvio_prior=DESVIO_PRIOR,
        max_iter=20,
        tol=1e-4,
    )

    caminho_parametros_enem = (
        PASTA_RESULTADOS_SIMULADO /
        f"parametros_estimados_somente_enem_{nome_grupo}.csv"
    )

    parametros_estimados_enem.to_csv(
        caminho_parametros_enem,
        index=False,
        encoding="utf-8-sig"
    )

    df_comparacao_enem = df_param_enem.merge(
        parametros_estimados_enem,
        on="ITEM",
        how="inner"
    )

    caminho_comparacao_enem = (
        PASTA_RESULTADOS_SIMULADO /
        f"comparacao_parametros_somente_enem_{nome_grupo}.csv"
    )

    df_comparacao_enem.to_csv(
        caminho_comparacao_enem,
        index=False,
        encoding="utf-8-sig"
    )

    calcular_metricas_parametros(
        df_comparacao_enem,
        "SOMENTE ITENS ENEM"
    )

    print("\n==============================")
    print("RESUMO DO GRUPO")
    print("==============================")

    print("Grupo:", SERIE_SIMULADO_EXECUTAR)
    print("Quantidade de alunos:", len(df_matriz_enem))
    print("Quantidade de itens:", len(df_comparacao_enem))




if __name__ == "__main__":
    main()