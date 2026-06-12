import pandas as pd
import numpy as np
from scipy.stats import pearsonr


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_ENTRADA = "resultados/simulado/comparacao_parametros_somente_enem_experimento_a.csv"
ARQUIVO_SAIDA_RESUMO = "resultados/simulado/correlacoes_por_ano_enem.csv"
ARQUIVO_SAIDA_ITENS = "resultados/simulado/residuos_itens_enem_por_ano.csv"



COLUNAS_PARAMETROS = {
    "A_REF": "A_REF",
    "B_REF": "B_REF",
    "C_REF": "C_REF",
    "A_EST": "A_EST",
    "B_EST": "B_EST",
    "C_EST": "C_EST",
}

MAPA_ANOS = {
    2019: list(range(136, 144)),  # Q136 a Q143
    2020: list(range(144, 150)),  # Q144 a Q149
    2021: list(range(150, 157)),  # Q150 a Q156
    2022: list(range(157, 166)),  # Q157 a Q165
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def obter_ano_enem(questao):
    for ano, questoes in MAPA_ANOS.items():
        if int(questao) in questoes:
            return ano
    return np.nan


def calcular_correlacao(x, y):
    df = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(df) < 2:
        return np.nan

    if df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return np.nan

    return pearsonr(df["x"], df["y"])[0]


def calcular_metricas_grupo(df_grupo):
    a_ref = COLUNAS_PARAMETROS["A_REF"]
    b_ref = COLUNAS_PARAMETROS["B_REF"]
    c_ref = COLUNAS_PARAMETROS["C_REF"]
    a_est = COLUNAS_PARAMETROS["A_EST"]
    b_est = COLUNAS_PARAMETROS["B_EST"]
    c_est = COLUNAS_PARAMETROS["C_EST"]

    erro_b = df_grupo[b_est] - df_grupo[b_ref]

    return {
        "N_ITENS": len(df_grupo),
        "CORR_A": calcular_correlacao(df_grupo[a_ref], df_grupo[a_est]),
        "CORR_B": calcular_correlacao(df_grupo[b_ref], df_grupo[b_est]),
        "CORR_C": calcular_correlacao(df_grupo[c_ref], df_grupo[c_est]),
        "MAE_B": np.mean(np.abs(erro_b)),
        "RMSE_B": np.sqrt(np.mean(erro_b ** 2)),
        "ERRO_MEDIO_B": np.mean(erro_b),
        "DP_ERRO_B": np.std(erro_b, ddof=1) if len(erro_b) > 1 else np.nan,
    }


# ============================================================
# SCRIPT PRINCIPAL
# ============================================================

def main():
    print("Lendo arquivo de parâmetros...")

    df = pd.read_csv(ARQUIVO_ENTRADA)

    df["ANO_ENEM"] = (
        df["ORIGEM"]
        .astype(str)
        .str.extract(r"(\d{4})")[0]
    )

    linhas_sem_ano = df[df["ANO_ENEM"].isna()]

    if len(linhas_sem_ano) > 0:
        print("\nAtenção: há linhas em que não foi possível extrair o ano da coluna ORIGEM.")
        print(linhas_sem_ano[["ORIGEM", "ITEM", "POSICAO_PROVA"]].head(20))
        df = df.dropna(subset=["ANO_ENEM"])

    df["ANO_ENEM"] = df["ANO_ENEM"].astype(int)

    print(f"Arquivo lido: {ARQUIVO_ENTRADA}")
    print(f"Total de linhas: {len(df)}")

    colunas_obrigatorias = [
        "ITEM",
        "POSICAO_PROVA",
        "ORIGEM",
        "A_REF",
        "B_REF",
        "C_REF",
        "A_EST",
        "B_EST",
        "C_EST",
    ]

    faltantes = [col for col in colunas_obrigatorias if col not in df.columns]

    if faltantes:
        raise ValueError(
            "As seguintes colunas não foram encontradas no arquivo de entrada: "
            + ", ".join(faltantes)
        )

    # --------------------------------------------------------
    # Filtra apenas itens ENEM mapeados por ano
    # --------------------------------------------------------

    df_enem = df[df["ANO_ENEM"].notna()].copy()
    df_enem["ANO_ENEM"] = df_enem["ANO_ENEM"].astype(int)

    print(f"Itens ENEM encontrados: {len(df_enem)}")

    if df_enem.empty:
        raise ValueError("Nenhum item ENEM foi encontrado com o mapeamento atual.")

    # --------------------------------------------------------
    # Calcula resíduos dos parâmetros
    # --------------------------------------------------------

    df_enem["ERRO_A"] = (
        df_enem[COLUNAS_PARAMETROS["A_EST"]]
        - df_enem[COLUNAS_PARAMETROS["A_REF"]]
    )

    df_enem["ERRO_B"] = (
        df_enem[COLUNAS_PARAMETROS["B_EST"]]
        - df_enem[COLUNAS_PARAMETROS["B_REF"]]
    )

    df_enem["ERRO_C"] = (
        df_enem[COLUNAS_PARAMETROS["C_EST"]]
        - df_enem[COLUNAS_PARAMETROS["C_REF"]]
    )

    df_enem["ABS_ERRO_B"] = df_enem["ERRO_B"].abs()

    # --------------------------------------------------------
    # Tabela resumo por ano
    # --------------------------------------------------------

    resultados = []

    for ano, df_ano in df_enem.groupby("ANO_ENEM"):
        metricas = calcular_metricas_grupo(df_ano)
        metricas["ANO_ENEM"] = ano
        resultados.append(metricas)

    df_resumo = pd.DataFrame(resultados)

    df_resumo = df_resumo[
        [
            "ANO_ENEM",
            "N_ITENS",
            "CORR_A",
            "CORR_B",
            "CORR_C",
            "MAE_B",
            "RMSE_B",
            "ERRO_MEDIO_B",
            "DP_ERRO_B",
        ]
    ].sort_values("ANO_ENEM")

    # --------------------------------------------------------
    # Tabela item a item
    # --------------------------------------------------------

    colunas_itens = [
        "ITEM",
        "POSICAO_PROVA",
        "ORIGEM",
        "ANO_ENEM",
        COLUNAS_PARAMETROS["A_REF"],
        COLUNAS_PARAMETROS["A_EST"],
        "ERRO_A",
        COLUNAS_PARAMETROS["B_REF"],
        COLUNAS_PARAMETROS["B_EST"],
        "ERRO_B",
        "ABS_ERRO_B",
        COLUNAS_PARAMETROS["C_REF"],
        COLUNAS_PARAMETROS["C_EST"],
        "ERRO_C",
    ]

    df_itens = df_enem[colunas_itens].copy()

    df_itens = df_itens.sort_values(
        by="ABS_ERRO_B",
        ascending=False
    )

    # --------------------------------------------------------
    # Salva resultados
    # --------------------------------------------------------

    df_resumo.to_csv(
        ARQUIVO_SAIDA_RESUMO,
        index=False,
        encoding="utf-8-sig"
    )

    df_itens.to_csv(
        ARQUIVO_SAIDA_ITENS,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Exibe resultados no terminal
    # --------------------------------------------------------

    print("\n==============================================")
    print("CORRELAÇÕES POR ANO")
    print("==============================================")

    print(
        df_resumo.round(4)
        .to_string(index=False)
    )

    print("\n==============================================")
    print("ITENS COM MAIORES ERROS EM B")
    print("==============================================")

    print(
        df_itens[
            [
                "ITEM",
                "POSICAO_PROVA",
                "ORIGEM",
                "ANO_ENEM",
                COLUNAS_PARAMETROS["B_REF"],
                COLUNAS_PARAMETROS["B_EST"],
                "ERRO_B",
                "ABS_ERRO_B",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\nArquivos gerados:")
    print(f"- {ARQUIVO_SAIDA_RESUMO}")
    print(f"- {ARQUIVO_SAIDA_ITENS}")


if __name__ == "__main__":
    main()