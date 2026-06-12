import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from codigo.configuracoes import (
    PASTA_RESULTADOS_SIMULADO,
    THETA_MIN,
    THETA_MAX,
    N_PONTOS_THETA,
    MEDIA_PRIOR,
    DESVIO_PRIOR,
)

from codigo.calibracao.calibracao_3pl_mml_em import (
    calibrar_itens_3pl_mml_em,
)


def calcular_metricas_b(df, titulo):
    dados = df[["B_REF", "B_EST"]].dropna()

    corr = dados["B_REF"].corr(dados["B_EST"])
    mae = mean_absolute_error(dados["B_REF"], dados["B_EST"])
    rmse = np.sqrt(mean_squared_error(dados["B_REF"], dados["B_EST"]))

    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)
    print(f"N itens              = {len(dados)}")
    print(f"Corr(B_REF, B_EST)  = {corr:.6f}")
    print(f"MAE_B               = {mae:.6f}")
    print(f"RMSE_B              = {rmse:.6f}")


def main():
    pasta = PASTA_RESULTADOS_SIMULADO

    df_matriz = pd.read_csv(
        pasta / "matriz_amostra_experimento_a.csv"
    )

    df_comp = pd.read_csv(
        pasta / "comparacao_parametros_somente_enem_experimento_a.csv"
    )

    df_comp_enem = df_comp[
        df_comp["GRUPO_ITEM"] == "ENEM"
    ].copy()

    df_comp_enem["ABS_ERRO_B"] = (
        df_comp_enem["B_REF"] - df_comp_enem["B_EST"]
    ).abs()

    itens_remover = (
        df_comp_enem
        .sort_values("ABS_ERRO_B", ascending=False)
        .head(5)["ITEM"]
        .tolist()
    )

    print("\nItens removidos:")
    print(itens_remover)

    itens_enem_restantes = [
        item for item in df_comp_enem["ITEM"].tolist()
        if item not in itens_remover
    ]

    colunas_base = [
        c for c in ["INSC", "ID_DISCIPLINA", "NOTA_REAL"]
        if c in df_matriz.columns
    ]

    df_matriz_filtrada = df_matriz[
        colunas_base + itens_enem_restantes
    ].copy()

    print("\nMatriz após remoção:")
    print(df_matriz_filtrada.shape)

    print("\nCalibrando itens ENEM sem os 5 maiores outliers em B...")

    df_param_est = calibrar_itens_3pl_mml_em(
        df_matriz_filtrada,
        theta_min=THETA_MIN,
        theta_max=THETA_MAX,
        n_pontos=N_PONTOS_THETA,
        media_prior=MEDIA_PRIOR,
        desvio_prior=DESVIO_PRIOR,
        max_iter=20,
        tol=1e-4,
    )

    caminho_param_est = (
        pasta / "parametros_estimados_enem_sem_5_outliers_b.csv"
    )

    df_param_est.to_csv(
        caminho_param_est,
        index=False,
        encoding="utf-8-sig"
    )

    df_ref_restante = df_comp_enem[
        df_comp_enem["ITEM"].isin(itens_enem_restantes)
    ].copy()

    colunas_ref = [
        "ITEM",
        "ORIGEM",
        "GRUPO_ITEM",
        "A_REF",
        "B_REF",
        "C_REF",
    ]

    if "ANO_ENEM" in df_ref_restante.columns:
        colunas_ref.insert(2, "ANO_ENEM")

    df_ref_restante = df_ref_restante[colunas_ref]

    df_comparacao = df_ref_restante.merge(
        df_param_est,
        on="ITEM",
        how="inner"
    )

    caminho_comparacao = (
        pasta / "comparacao_enem_sem_5_outliers_b.csv"
    )

    df_comparacao.to_csv(
        caminho_comparacao,
        index=False,
        encoding="utf-8-sig"
    )

    calcular_metricas_b(
        df_comp_enem,
        "ANTES DA REMOÇÃO - TODOS OS ITENS ENEM"
    )

    calcular_metricas_b(
        df_comparacao,
        "DEPOIS DA REMOÇÃO - SEM 5 OUTLIERS EM B"
    )

    print("\nArquivos salvos:")
    print(caminho_param_est)
    print(caminho_comparacao)


if __name__ == "__main__":
    main()