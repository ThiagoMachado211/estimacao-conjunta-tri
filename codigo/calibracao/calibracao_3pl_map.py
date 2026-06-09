import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import norm, beta

from codigo.calibracao.modelos_irt import probabilidade_3pl


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def inicializar_theta_por_acertos(df_matriz):
    colunas_q = [c for c in df_matriz.columns if c.startswith("Q")]

    acertos = df_matriz[colunas_q].sum(axis=1, skipna=True)
    validas = df_matriz[colunas_q].notna().sum(axis=1)

    proporcao = acertos / validas
    proporcao = proporcao.replace([np.inf, -np.inf], np.nan)
    proporcao = proporcao.fillna(proporcao.mean())

    theta = (proporcao - proporcao.mean()) / proporcao.std()
    theta = theta.fillna(0).values

    return theta


def log_posterior_negativo_item_3pl(params, theta, respostas):
    """
    MAP para item 3PL.

    params = [log_a, b, logit_c]

    Priors:
        log(a) ~ Normal(0, 0.5)
        b      ~ Normal(0, 2)
        c      ~ Beta(5, 20)
    """

    log_a, b, logit_c = params

    a = np.exp(log_a)
    c = 1 / (1 + np.exp(-logit_c))

    respostas = np.asarray(respostas, dtype=float)
    theta = np.asarray(theta, dtype=float)

    mascara = ~np.isnan(respostas)

    if mascara.sum() == 0:
        return 1e9

    u = respostas[mascara]
    th = theta[mascara]

    p = probabilidade_3pl(th, a, b, c)
    p = np.clip(p, 1e-9, 1 - 1e-9)

    log_likelihood = np.sum(
        u * np.log(p) + (1 - u) * np.log(1 - p)
    )

    log_prior_a = norm.logpdf(log_a, loc=0, scale=0.5)
    log_prior_b = norm.logpdf(b, loc=0, scale=2)
    log_prior_c = beta.logpdf(c, a=5, b=20)

    # Correção da transformação logit(c)
    # p(logit_c) = p(c) * c * (1-c)
    log_jacobiano_c = np.log(c) + np.log(1 - c)

    log_posterior = (
        log_likelihood
        + log_prior_a
        + log_prior_b
        + log_prior_c
        + log_jacobiano_c
    )

    return -log_posterior


def calibrar_item_3pl_map(theta, respostas_item):
    """
    Calibra um item 3PL via MAP.
    """

    params_iniciais = np.array([
        np.log(1.0),   # log_a
        0.0,           # b
        logit(0.20)    # logit_c
    ])

    limites = [
        (np.log(0.01), np.log(5.0)),      # a
        (-4.0, 4.0),                     # b
        (logit(0.01), logit(0.35))       # c
    ]

    resultado = minimize(
        log_posterior_negativo_item_3pl,
        params_iniciais,
        args=(theta, respostas_item),
        method="L-BFGS-B",
        bounds=limites,
        options={"maxiter": 500}
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


def calibrar_itens_3pl_map(df_matriz):
    """
    Calibra todos os itens Q1...Q45 via 3PL MAP.
    """

    colunas_q = [c for c in df_matriz.columns if c.startswith("Q")]

    theta_inicial = inicializar_theta_por_acertos(df_matriz)

    resultados = []

    for item in colunas_q:
        print(f"Calibrando {item} via 3PL MAP...")

        respostas_item = df_matriz[item].values.astype(float)

        resultado_item = calibrar_item_3pl_map(
            theta_inicial,
            respostas_item
        )

        resultados.append({
            "ITEM": item,
            **resultado_item
        })

    df_parametros = pd.DataFrame(resultados)

    return df_parametros, theta_inicial