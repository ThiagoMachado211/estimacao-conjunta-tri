import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from codigo.configuracoes import (
    PASTA_RESULTADOS_SIMULADO,
)


ITENS_PROBLEMATICOS = [
    "Q5",
    "Q8",
    "Q27",
    "Q30",
    "Q7",
]


def calcular_rmse(y_true, y_pred):
    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


def imprimir_correlacoes_parametros_frequencia(df, titulo):
    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)

    print(f"N itens: {len(df)}")

    for parametro in ["A", "B", "C"]:
        col = f"{parametro}_REF"

        corr = df[col].corr(
            df["P_ACERTO"]
        )

        print(
            f"Corr({col}, P_ACERTO) = {corr:.6f}"
        )


def imprimir_metricas_ref_est(df, titulo):
    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)

    print(f"N itens: {len(df)}")

    for parametro in ["A", "B", "C"]:
        col_ref = f"{parametro}_REF"
        col_est = f"{parametro}_EST"

        dados = df[
            [
                col_ref,
                col_est
            ]
        ].dropna()

        if len(dados) < 2:
            print(f"\nParâmetro {parametro}")
            print("Métricas não calculáveis.")
            continue

        corr = dados[col_ref].corr(
            dados[col_est]
        )

        mae = mean_absolute_error(
            dados[col_ref],
            dados[col_est]
        )

        rmse = calcular_rmse(
            dados[col_ref],
            dados[col_est]
        )

        print(f"\nParâmetro {parametro}")
        print(f"Corr = {corr:.6f}")
        print(f"MAE  = {mae:.6f}")
        print(f"RMSE = {rmse:.6f}")


def preparar_base():
    caminho_param_area = (
        PASTA_RESULTADOS_SIMULADO /
        "parametros_referencia_area_experimento_a.csv"
    )

    caminho_freq = (
        PASTA_RESULTADOS_SIMULADO /
        "frequencia_acerto_experimento_a.csv"
    )

    caminho_comp = (
        PASTA_RESULTADOS_SIMULADO /
        "comparacao_parametros_experimento_a.csv"
    )

    df_param = pd.read_csv(caminho_param_area)
    df_freq = pd.read_csv(caminho_freq)
    df_comp = pd.read_csv(caminho_comp)

    df_param_freq = df_param.merge(
        df_freq,
        on="ITEM",
        how="inner"
    )

    df_comp = df_comp.merge(
        df_freq[["ITEM", "P_ACERTO"]],
        on="ITEM",
        how="left"
    )

    df_comp_enem = df_comp[
        df_comp["GRUPO_ITEM"] == "ENEM"
    ].copy()

    df_comp_enem["ANO_ENEM"] = (
        df_comp_enem["ORIGEM"]
        .astype(str)
        .str.extract(r"(20\d{2})")
    )

    df_comp_enem["ERRO_B"] = (
        df_comp_enem["B_EST"]
        - df_comp_enem["B_REF"]
    )

    df_comp_enem["ABS_ERRO_B"] = (
        df_comp_enem["ERRO_B"]
        .abs()
    )

    df_param_freq_enem = df_param_freq[
        df_param_freq["GRUPO_ITEM"] == "ENEM"
    ].copy()

    df_param_freq_enem["ANO_ENEM"] = (
        df_param_freq_enem["ORIGEM"]
        .astype(str)
        .str.extract(r"(20\d{2})")
    )

    return df_param_freq, df_param_freq_enem, df_comp, df_comp_enem


def analisar_parametros_frequencia(df_param_freq):
    print("\n" + "=" * 80)
    print("CORRELAÇÕES GERAIS - PARÂMETROS x FREQUÊNCIA OBSERVADA")
    print("=" * 80)

    for grupo in ["ENEM", "INEDITA"]:
        temp = df_param_freq[
            df_param_freq["GRUPO_ITEM"] == grupo
        ].copy()

        imprimir_correlacoes_parametros_frequencia(
            temp,
            f"Grupo: {grupo}"
        )


def analisar_enem_por_ano_parametros_frequencia(df_param_freq_enem):
    print("\n" + "=" * 80)
    print("ITENS ENEM POR ANO - PARÂMETROS x FREQUÊNCIA")
    print("=" * 80)

    resultados = []

    for ano in sorted(
        df_param_freq_enem["ANO_ENEM"]
        .dropna()
        .unique()
    ):
        temp = df_param_freq_enem[
            df_param_freq_enem["ANO_ENEM"] == ano
        ].copy()

        if len(temp) < 2:
            continue

        resultados.append({
            "ANO": ano,
            "N_ITENS": len(temp),
            "MEDIA_P_ACERTO": temp["P_ACERTO"].mean(),
            "MEDIA_A_REF": temp["A_REF"].mean(),
            "MEDIA_B_REF": temp["B_REF"].mean(),
            "MEDIA_C_REF": temp["C_REF"].mean(),
            "CORR_A_REF_P_ACERTO": temp["A_REF"].corr(temp["P_ACERTO"]),
            "CORR_B_REF_P_ACERTO": temp["B_REF"].corr(temp["P_ACERTO"]),
            "CORR_C_REF_P_ACERTO": temp["C_REF"].corr(temp["P_ACERTO"]),
        })

    df_resultados = pd.DataFrame(resultados)

    print(
        df_resultados.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    caminho_saida = (
        PASTA_RESULTADOS_SIMULADO /
        "teste_correlacoes_parametros_frequencia.csv"
    )

    df_resultados.to_csv(
        caminho_saida,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nResultado salvo em: {caminho_saida}")


def detalhar_anos(df_param_freq_enem):
    colunas = [
        "ITEM",
        "POSICAO_PROVA",
        "ORIGEM",
        "ANO_ENEM",
        "A_REF",
        "B_REF",
        "C_REF",
        "P_ACERTO",
    ]

    print("\n" + "=" * 80)
    print("DETALHAMENTO 2019 E 2020")
    print("=" * 80)

    df_2019_2020 = df_param_freq_enem[
        df_param_freq_enem["ANO_ENEM"].isin(
            [
                "2019",
                "2020",
            ]
        )
    ].copy()

    print(
        df_2019_2020[colunas]
        .sort_values(
            [
                "ANO_ENEM",
                "B_REF",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\n" + "=" * 80)
    print("DETALHAMENTO 2021 E 2022")
    print("=" * 80)

    df_2021_2022 = df_param_freq_enem[
        df_param_freq_enem["ANO_ENEM"].isin(
            [
                "2021",
                "2022",
            ]
        )
    ].copy()

    print(
        df_2021_2022[colunas]
        .sort_values(
            [
                "ANO_ENEM",
                "B_REF",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


def analisar_ref_est_por_ano(df_comp_enem):
    print("\n" + "=" * 80)
    print("CORRELAÇÕES REF x EST POR ANO")
    print("=" * 80)

    resultados = []

    for ano in sorted(
        df_comp_enem["ANO_ENEM"]
        .dropna()
        .unique()
    ):
        temp = df_comp_enem[
            df_comp_enem["ANO_ENEM"] == ano
        ].copy()

        resultados.append({
            "ANO": ano,
            "N_ITENS": len(temp),
            "CORR_A": temp["A_REF"].corr(temp["A_EST"]),
            "CORR_B": temp["B_REF"].corr(temp["B_EST"]),
            "CORR_C": temp["C_REF"].corr(temp["C_EST"]),
            "MAE_B": mean_absolute_error(temp["B_REF"], temp["B_EST"]),
            "RMSE_B": calcular_rmse(temp["B_REF"], temp["B_EST"]),
            "MEDIA_B_REF": temp["B_REF"].mean(),
            "MEDIA_B_EST": temp["B_EST"].mean(),
            "STD_B_REF": temp["B_REF"].std(),
            "STD_B_EST": temp["B_EST"].std(),
        })

    df_resultados = pd.DataFrame(resultados)

    print(
        df_resultados.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    caminho_saida = (
        PASTA_RESULTADOS_SIMULADO /
        "teste_correlacoes_ref_est_por_ano.csv"
    )

    df_resultados.to_csv(
        caminho_saida,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nResultado salvo em: {caminho_saida}")


def analisar_maiores_erros_b(df_comp_enem):
    print("\n" + "=" * 80)
    print("MAIORES ERROS ABSOLUTOS EM B - ITENS ENEM")
    print("=" * 80)

    colunas = [
        "ITEM",
        "ANO_ENEM",
        "ORIGEM",
        "B_REF",
        "B_EST",
        "ERRO_B",
        "ABS_ERRO_B",
        "P_ACERTO",
    ]

    top = (
        df_comp_enem
        .sort_values(
            "ABS_ERRO_B",
            ascending=False
        )
        .head(15)
    )

    print(
        top[colunas]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    caminho_saida = (
        PASTA_RESULTADOS_SIMULADO /
        "maiores_erros_b_itens_enem.csv"
    )

    top.to_csv(
        caminho_saida,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nTop erros salvo em: {caminho_saida}")


def analisar_b_ref_b_est_p_acerto(df_comp_enem):
    print("\n" + "=" * 80)
    print("CORRELAÇÕES B_REF, B_EST E P_ACERTO")
    print("=" * 80)

    corr_b_ref_b_est = df_comp_enem["B_REF"].corr(
        df_comp_enem["B_EST"]
    )

    corr_b_ref_acerto = df_comp_enem["B_REF"].corr(
        df_comp_enem["P_ACERTO"]
    )

    corr_b_est_acerto = df_comp_enem["B_EST"].corr(
        df_comp_enem["P_ACERTO"]
    )

    print(f"Corr(B_REF, B_EST)     = {corr_b_ref_b_est:.6f}")
    print(f"Corr(B_REF, P_ACERTO)  = {corr_b_ref_acerto:.6f}")
    print(f"Corr(B_EST, P_ACERTO)  = {corr_b_est_acerto:.6f}")


def analisar_grupos_problematicos(df_comp_enem):
    print("\n" + "=" * 80)
    print("ANÁLISE DOS ITENS PROBLEMÁTICOS")
    print("=" * 80)

    df_prob = df_comp_enem[
        df_comp_enem["ITEM"].isin(ITENS_PROBLEMATICOS)
    ].copy()

    df_normal = df_comp_enem[
        ~df_comp_enem["ITEM"].isin(ITENS_PROBLEMATICOS)
    ].copy()

    print("\nGrupo problemático")
    print(f"Itens: {len(df_prob)}")
    print(f"Lista: {ITENS_PROBLEMATICOS}")

    corr_ref_prob = df_prob["B_REF"].corr(
        df_prob["P_ACERTO"]
    )

    corr_est_prob = df_prob["B_EST"].corr(
        df_prob["P_ACERTO"]
    )

    corr_ref_est_prob = df_prob["B_REF"].corr(
        df_prob["B_EST"]
    )

    print(f"Corr(B_REF, P_ACERTO) = {corr_ref_prob:.6f}")
    print(f"Corr(B_EST, P_ACERTO) = {corr_est_prob:.6f}")
    print(f"Corr(B_REF, B_EST)    = {corr_ref_est_prob:.6f}")

    print("\nGrupo normal")
    print(f"Itens: {len(df_normal)}")

    corr_ref_normal = df_normal["B_REF"].corr(
        df_normal["P_ACERTO"]
    )

    corr_est_normal = df_normal["B_EST"].corr(
        df_normal["P_ACERTO"]
    )

    corr_ref_est_normal = df_normal["B_REF"].corr(
        df_normal["B_EST"]
    )

    print(f"Corr(B_REF, P_ACERTO) = {corr_ref_normal:.6f}")
    print(f"Corr(B_EST, P_ACERTO) = {corr_est_normal:.6f}")
    print(f"Corr(B_REF, B_EST)    = {corr_ref_est_normal:.6f}")

    print("\n" + "=" * 80)
    print("RESUMO DESCRITIVO DOS GRUPOS")
    print("=" * 80)

    print("\nGrupo problemático")
    print(
        df_prob[
            [
                "B_REF",
                "B_EST",
                "P_ACERTO",
            ]
        ].describe()
    )

    print("\nGrupo normal")
    print(
        df_normal[
            [
                "B_REF",
                "B_EST",
                "P_ACERTO",
            ]
        ].describe()
    )

    print("\n" + "=" * 80)
    print("ITENS DO GRUPO PROBLEMÁTICO")
    print("=" * 80)

    colunas_mostrar = [
        "ITEM",
        "ANO_ENEM",
        "ORIGEM",
        "B_REF",
        "B_EST",
        "ERRO_B",
        "ABS_ERRO_B",
        "P_ACERTO",
    ]

    print(
        df_prob[
            colunas_mostrar
        ]
        .sort_values("ITEM")
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    caminho_prob = (
        PASTA_RESULTADOS_SIMULADO /
        "itens_enem_grupo_problematico.csv"
    )

    caminho_normal = (
        PASTA_RESULTADOS_SIMULADO /
        "itens_enem_grupo_normal.csv"
    )

    df_prob.to_csv(
        caminho_prob,
        index=False,
        encoding="utf-8-sig"
    )

    df_normal.to_csv(
        caminho_normal,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nGrupo problemático salvo em: {caminho_prob}")
    print(f"Grupo normal salvo em: {caminho_normal}")


def main():
    (
        df_param_freq,
        df_param_freq_enem,
        df_comp,
        df_comp_enem,
    ) = preparar_base()

    analisar_parametros_frequencia(
        df_param_freq
    )

    analisar_enem_por_ano_parametros_frequencia(
        df_param_freq_enem
    )

    detalhar_anos(
        df_param_freq_enem
    )

    analisar_ref_est_por_ano(
        df_comp_enem
    )

    analisar_maiores_erros_b(
        df_comp_enem
    )

    analisar_b_ref_b_est_p_acerto(
        df_comp_enem
    )

    analisar_grupos_problematicos(
        df_comp_enem
    )


    corr_a_ref_b_ref = df_comp_enem["A_REF"].corr(
        df_comp_enem["B_REF"]
    )

    corr_a_est_b_est = df_comp_enem["A_EST"].corr(
        df_comp_enem["B_EST"]
    )

    print(corr_a_ref_b_ref)
    print(corr_a_est_b_est)



    print("\n" + "=" * 80)
    print("DETALHAMENTO COMPLETO DOS ITENS PROBLEMÁTICOS")
    print("=" * 80)

    itens_problematicos = [
        "Q5",
        "Q8",
        "Q27",
        "Q30",
        "Q7",
    ]

    df_detalhe_prob = df_comp_enem[
        df_comp_enem["ITEM"].isin(itens_problematicos)
    ].copy()

    colunas = [
        "ITEM",
        "ANO_ENEM",
        "ORIGEM",
        "HABILIDADE",
        "POSICAO_PROVA",
        "A_REF",
        "A_EST",
        "B_REF",
        "B_EST",
        "C_REF",
        "C_EST",
        "P_ACERTO",
    ]

    colunas = [
        c for c in colunas
        if c in df_detalhe_prob.columns
    ]

    print(
        df_detalhe_prob[colunas]
        .sort_values("ITEM")
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )




if __name__ == "__main__":
    main()