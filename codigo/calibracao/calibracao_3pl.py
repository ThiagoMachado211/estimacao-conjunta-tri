import numpy as np
import pandas as pd

from scipy.optimize import minimize

from codigo.calibracao.modelos_irt import log_likelihood_item_3pl


def inicializar_theta_por_acertos(df_matriz):
    """
    Inicializa theta usando escore padronizado de acertos.
    """

    colunas_q = [c for c in df_matriz.columns if c.startswith("Q")]

    acertos = df_matriz[colunas_q].sum(axis=1, skipna=True)
    validas = df_matriz[colunas_q].notna().sum(axis=1)

    proporcao = acertos / validas
    proporcao = proporcao.replace([np.inf, -np.inf], np.nan)
    proporcao = proporcao.fillna(proporcao.mean())

    theta = (proporcao - proporcao.mean()) / proporcao.std()

    theta = theta.fillna(0).values

    return theta


def calibrar_item_3pl(theta, respostas_item):
    """
    Calibra um único item no modelo 3PL.
    """

    chute_inicial = np.nanmean(respostas_item)

    if np.isnan(chute_inicial):
        chute_inicial = 0.5

    chute_inicial = np.clip(chute_inicial, 0.05, 0.95)

    params_iniciais = np.array([
        np.log(1.0),   # log_a
        0.0,           # b
        np.log(0.20 / 0.80)  # logit_c
    ])

    limites = [
        (np.log(0.01), np.log(5.0)),     # a
        (-4.0, 4.0),                    # b
        (np.log(0.01 / 0.99), np.log(0.35 / 0.65))  # c
    ]

    resultado = minimize(
        log_likelihood_item_3pl,
        params_iniciais,
        args=(theta, respostas_item),
        method="L-BFGS-B",
        bounds=limites,
        options={
            "maxiter": 500
        }
    )

    if not resultado.success:
        return {
            "A_EST": np.nan,
            "B_EST": np.nan,
            "C_EST": np.nan,
            "CONVERGIU": False,
            "MENSAGEM": resultado.message
        }

    log_a, b, logit_c = resultado.x

    a = np.exp(log_a)
    c = 1 / (1 + np.exp(-logit_c))

    return {
        "A_EST": a,
        "B_EST": b,
        "C_EST": c,
        "CONVERGIU": True,
        "MENSAGEM": resultado.message
    }


def calibrar_itens_3pl(df_matriz):
    """
    Calibra todos os itens Q1...Q45 no modelo 3PL.
    """

    colunas_q = [c for c in df_matriz.columns if c.startswith("Q")]

    theta_inicial = inicializar_theta_por_acertos(df_matriz)

    resultados = []

    for item in colunas_q:
        print(f"Calibrando {item}...")

        respostas_item = df_matriz[item].values.astype(float)

        resultado_item = calibrar_item_3pl(
            theta_inicial,
            respostas_item
        )

        resultados.append({
            "ITEM": item,
            **resultado_item
        })

    df_parametros = pd.DataFrame(resultados)

    return df_parametros, theta_inicial