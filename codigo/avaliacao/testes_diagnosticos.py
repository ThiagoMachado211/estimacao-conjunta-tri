import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr


def correlacao_segura(df, col_x, col_y):
    dados = df[[col_x, col_y]].dropna()

    if len(dados) < 2:
        return np.nan

    if dados[col_x].std() == 0 or dados[col_y].std() == 0:
        return np.nan

    return pearsonr(dados[col_x], dados[col_y])[0]


def estatisticas_parametros(df_param_ref, df_param_est):
    print("\n==============================")
    print("ESTATÍSTICAS DOS PARÂMETROS")
    print("==============================")

    print("\nParâmetros oficiais:")
    print(df_param_ref[["A_REF", "B_REF", "C_REF"]].describe())

    print("\nParâmetros estimados:")
    print(df_param_est[["A_EST", "B_EST", "C_EST"]].describe())


def estatisticas_thetas(df_proficiencias, df_proficiencias_ref):
    print("\n==============================")
    print("ESTATÍSTICAS DOS THETAS")
    print("==============================")

    print("\nTheta com parâmetros estimados:")
    print(df_proficiencias["THETA_EAP"].describe())

    print("\nTheta com parâmetros oficiais:")
    print(df_proficiencias_ref["THETA_EAP_REF"].describe())


def correlacoes_diagnosticas(
    df_proficiencias,
    df_proficiencias_ref
):
    print("\n==============================")
    print("CORRELAÇÕES DIAGNÓSTICAS")
    print("==============================")

    df = df_proficiencias[
        ["INSC", "N_ACERTOS", "THETA_EAP", "NOTA_REAL"]
    ].merge(
        df_proficiencias_ref[
            ["INSC", "THETA_EAP_REF"]
        ],
        on="INSC",
        how="inner"
    )

    print(
        "Corr(N_ACERTOS, THETA_EAP):",
        correlacao_segura(df, "N_ACERTOS", "THETA_EAP")
    )

    print(
        "Corr(N_ACERTOS, THETA_EAP_REF):",
        correlacao_segura(df, "N_ACERTOS", "THETA_EAP_REF")
    )

    print(
        "Corr(THETA_EAP, THETA_EAP_REF):",
        correlacao_segura(df, "THETA_EAP", "THETA_EAP_REF")
    )

    print(
        "Corr(NOTA_REAL, THETA_EAP):",
        correlacao_segura(df, "NOTA_REAL", "THETA_EAP")
    )

    print(
        "Corr(NOTA_REAL, THETA_EAP_REF):",
        correlacao_segura(df, "NOTA_REAL", "THETA_EAP_REF")
    )

    return df


def grafico_histograma_comparado(
    serie_ref,
    serie_est,
    titulo,
    label_ref,
    label_est,
    caminho_saida
):
    dados_ref = serie_ref.dropna()
    dados_est = serie_est.dropna()

    plt.figure(figsize=(8, 6))

    plt.hist(dados_ref, bins=20, alpha=0.5, label=label_ref)
    plt.hist(dados_est, bins=20, alpha=0.5, label=label_est)

    plt.title(titulo)
    plt.xlabel("Valor")
    plt.ylabel("Frequência")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300)
    plt.close()


def grafico_dispersao(
    df,
    col_x,
    col_y,
    titulo,
    caminho_saida
):
    dados = df[[col_x, col_y]].dropna()

    plt.figure(figsize=(8, 6))

    plt.scatter(dados[col_x], dados[col_y], alpha=0.6)

    corr = correlacao_segura(dados, col_x, col_y)

    plt.title(f"{titulo}\nCorrelação = {corr:.4f}")
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300)
    plt.close()


def gerar_graficos_diagnosticos(
    df_param_ref,
    df_param_est,
    df_proficiencias,
    df_proficiencias_ref,
    pasta_graficos
):
    pasta_graficos.mkdir(parents=True, exist_ok=True)

    df_param = df_param_ref.merge(
        df_param_est,
        on="ITEM",
        how="inner"
    )

    # Histogramas dos parâmetros
    grafico_histograma_comparado(
        df_param["A_REF"],
        df_param["A_EST"],
        "Histograma A_REF x A_EST",
        "A_REF",
        "A_EST",
        pasta_graficos / "hist_A_REF_A_EST.png"
    )

    grafico_histograma_comparado(
        df_param["B_REF"],
        df_param["B_EST"],
        "Histograma B_REF x B_EST",
        "B_REF",
        "B_EST",
        pasta_graficos / "hist_B_REF_B_EST.png"
    )

    grafico_histograma_comparado(
        df_param["C_REF"],
        df_param["C_EST"],
        "Histograma C_REF x C_EST",
        "C_REF",
        "C_EST",
        pasta_graficos / "hist_C_REF_C_EST.png"
    )

    # Dispersão dos parâmetros
    grafico_dispersao(
        df_param,
        "A_REF",
        "A_EST",
        "A_REF x A_EST",
        pasta_graficos / "scatter_A_REF_A_EST.png"
    )

    grafico_dispersao(
        df_param,
        "B_REF",
        "B_EST",
        "B_REF x B_EST",
        pasta_graficos / "scatter_B_REF_B_EST.png"
    )

    grafico_dispersao(
        df_param,
        "C_REF",
        "C_EST",
        "C_REF x C_EST",
        pasta_graficos / "scatter_C_REF_C_EST.png"
    )

    # Comparação dos thetas
    df_theta = df_proficiencias[
        ["INSC", "NOTA_REAL", "THETA_EAP", "N_ACERTOS"]
    ].merge(
        df_proficiencias_ref[
            ["INSC", "THETA_EAP_REF"]
        ],
        on="INSC",
        how="inner"
    )

    grafico_dispersao(
        df_theta,
        "THETA_EAP_REF",
        "THETA_EAP",
        "THETA_EAP_REF x THETA_EAP",
        pasta_graficos / "scatter_THETA_REF_THETA_EST.png"
    )

    grafico_dispersao(
        df_theta,
        "THETA_EAP_REF",
        "NOTA_REAL",
        "THETA_EAP_REF x NOTA_REAL",
        pasta_graficos / "scatter_THETA_REF_NOTA_REAL.png"
    )

    grafico_dispersao(
        df_theta,
        "THETA_EAP",
        "NOTA_REAL",
        "THETA_EAP x NOTA_REAL",
        pasta_graficos / "scatter_THETA_EST_NOTA_REAL.png"
    )

    grafico_dispersao(
        df_theta,
        "N_ACERTOS",
        "THETA_EAP",
        "N_ACERTOS x THETA_EAP",
        pasta_graficos / "scatter_ACERTOS_THETA_EST.png"
    )

    grafico_dispersao(
        df_theta,
        "N_ACERTOS",
        "THETA_EAP_REF",
        "N_ACERTOS x THETA_EAP_REF",
        pasta_graficos / "scatter_ACERTOS_THETA_REF.png"
    )

    print("\nGráficos diagnósticos gerados com sucesso.")


def salvar_tabelas_diagnosticas(
    df_param_ref,
    df_param_est,
    df_proficiencias,
    df_proficiencias_ref,
    pasta_saida
):
    pasta_saida.mkdir(parents=True, exist_ok=True)

    df_param = df_param_ref.merge(
        df_param_est,
        on="ITEM",
        how="inner"
    )

    df_param.to_csv(
        pasta_saida / "diagnostico_parametros_ref_est.csv",
        index=False,
        encoding="utf-8-sig"
    )

    df_theta = df_proficiencias[
        ["INSC", "NOTA_REAL", "N_ACERTOS", "THETA_EAP"]
    ].merge(
        df_proficiencias_ref[
            ["INSC", "THETA_EAP_REF"]
        ],
        on="INSC",
        how="inner"
    )

    df_theta.to_csv(
        pasta_saida / "diagnostico_thetas_ref_est.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTabelas diagnósticas salvas com sucesso.")