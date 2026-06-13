import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


ARQUIVOS_GRUPOS = {
    "3EM_REGULAR":
        "resultados/simulado/comparacao_parametros_somente_enem_Ensino_Médio___3_ano.csv",

    "2EM_REGULAR":
        "resultados/simulado/comparacao_parametros_somente_enem_Ensino_Médio___2_ano.csv",

    "3EM_EJA":
        "resultados/simulado/comparacao_parametros_somente_enem_EJA___3_Ano.csv",

    "2EM_EJA":
        "resultados/simulado/comparacao_parametros_somente_enem_EJA___2_Ano.csv",
}

#ARQUIVOS_GRUPOS = {
#    "TODOS": "resultados/simulado/comparacao_parametros_somente_enem_experimento_a.csv",
#    "3EM_REGULAR": "resultados/simulado/comparacao_parametros_enem_3em_regular.csv",
#    "2EM_REGULAR": "resultados/simulado/comparacao_parametros_enem_2em_regular.csv",
#    "3EM_EJA": "resultados/simulado/comparacao_parametros_enem_3em_eja.csv",
#    "2EM_EJA": "resultados/simulado/comparacao_parametros_enem_2em_eja.csv",
#}

ARQUIVO_SAIDA_RESUMO = "resultados/simulado/teste_populacao_resumo.csv"
ARQUIVO_SAIDA_ITENS = "resultados/simulado/teste_populacao_itens.csv"


def corr_segura(x, y):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 2 or d["x"].nunique() < 2 or d["y"].nunique() < 2:
        return np.nan
    return pearsonr(d["x"], d["y"])[0]


def regressao_b(df):
    d = df.dropna(subset=["B_REF", "B_EST"]).copy()
    if len(d) < 2:
        return np.nan, np.nan, np.nan

    X = d[["B_EST"]]
    y = d["B_REF"]

    modelo = LinearRegression()
    modelo.fit(X, y)

    pred = modelo.predict(X)

    return r2_score(y, pred), modelo.intercept_, modelo.coef_[0]


def preparar_base(df):
    df = df.dropna(how="all").copy()

    df["ANO_ENEM"] = (
        df["ORIGEM"]
        .astype(str)
        .str.extract(r"(\d{4})")[0]
    )

    df = df.dropna(subset=["ANO_ENEM"])
    df["ANO_ENEM"] = df["ANO_ENEM"].astype(int)

    for col in ["A_REF", "B_REF", "C_REF", "A_EST", "B_EST", "C_EST"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def metricas(df, grupo):
    erro_a = df["A_EST"] - df["A_REF"]
    erro_b = df["B_EST"] - df["B_REF"]
    erro_c = df["C_EST"] - df["C_REF"]

    r2_b, alpha_b, beta_b = regressao_b(df)

    return {
        "GRUPO_ANALISE": grupo,
        "N_ITENS": len(df),
        "CORR_A": corr_segura(df["A_REF"], df["A_EST"]),
        "CORR_B": corr_segura(df["B_REF"], df["B_EST"]),
        "CORR_C": corr_segura(df["C_REF"], df["C_EST"]),
        "MAE_A": np.mean(np.abs(erro_a)),
        "MAE_B": np.mean(np.abs(erro_b)),
        "MAE_C": np.mean(np.abs(erro_c)),
        "RMSE_A": np.sqrt(np.mean(erro_a ** 2)),
        "RMSE_B": np.sqrt(np.mean(erro_b ** 2)),
        "RMSE_C": np.sqrt(np.mean(erro_c ** 2)),
        "ERRO_MEDIO_A": np.mean(erro_a),
        "ERRO_MEDIO_B": np.mean(erro_b),
        "ERRO_MEDIO_C": np.mean(erro_c),
        "R2_B": r2_b,
        "ALPHA_B": alpha_b,
        "BETA_B": beta_b,
    }


def main():
    resultados = []
    itens = []

    for grupo, caminho in ARQUIVOS_GRUPOS.items():
        if not os.path.exists(caminho):
            print(f"[AVISO] Arquivo não encontrado para {grupo}: {caminho}")
            continue

        print(f"\nLendo {grupo}: {caminho}")

        df = pd.read_csv(caminho)
        df = preparar_base(df)

        resultados.append(metricas(df, grupo))

        df["GRUPO_ANALISE"] = grupo
        df["ERRO_A"] = df["A_EST"] - df["A_REF"]
        df["ERRO_B"] = df["B_EST"] - df["B_REF"]
        df["ERRO_C"] = df["C_EST"] - df["C_REF"]
        df["ABS_ERRO_A"] = df["ERRO_A"].abs()
        df["ABS_ERRO_B"] = df["ERRO_B"].abs()
        df["ABS_ERRO_C"] = df["ERRO_C"].abs()

        itens.append(df)

    if not resultados:
        raise ValueError("Nenhum arquivo de comparação foi encontrado.")

    df_resumo = pd.DataFrame(resultados)
    df_itens = pd.concat(itens, ignore_index=True)

    df_resumo.to_csv(ARQUIVO_SAIDA_RESUMO, index=False, encoding="utf-8-sig")
    df_itens.to_csv(ARQUIVO_SAIDA_ITENS, index=False, encoding="utf-8-sig")

    print("\n==============================================")
    print("RESUMO POR POPULAÇÃO")
    print("==============================================")
    print(df_resumo.round(4).to_string(index=False))

    print("\n==============================================")
    print("MAIORES ERROS EM B")
    print("==============================================")
    print(
        df_itens[
            [
                "GRUPO_ANALISE",
                "ORIGEM",
                "ITEM",
                "POSICAO_PROVA",
                "B_REF",
                "B_EST",
                "ERRO_B",
                "ABS_ERRO_B",
            ]
        ]
        .sort_values("ABS_ERRO_B", ascending=False)
        .head(30)
        .round(4)
        .to_string(index=False)
    )

    print("\nArquivos gerados:")
    print(f"- {ARQUIVO_SAIDA_RESUMO}")
    print(f"- {ARQUIVO_SAIDA_ITENS}")


if __name__ == "__main__":
    main()