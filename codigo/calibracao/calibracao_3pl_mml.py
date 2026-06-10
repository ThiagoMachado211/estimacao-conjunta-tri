# codigo/calibracao/calibracao_3pl_mml.py

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import norm, beta

from codigo.calibracao.modelos_irt import probabilidade_3pl
from codigo.calibracao.quadratura import criar_quadratura_normal


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def log_verossimilhanca_marginal_item_3pl(
    params,
    respostas_item,
    grade_theta,
    pesos_theta
):
    """
    Log-verossimilhança marginal negativa de um item no modelo 3PL.

    Aqui o theta não é fixado por aluno.
    Ele é marginalizado pela distribuição N(0,1).

    params = [log_a, b, logit_c]
    """

    log_a, b, logit_c = params

    a = np.exp(log_a)
    c = 1 / (1 + np.exp(-logit_c))

    respostas = np.asarray(respostas_item, dtype=float)
    mascara = ~np.isnan(respostas)

    if mascara.sum() == 0:
        return 1e9

    u = respostas[mascara]

    p_theta = probabilidade_3pl(
        grade_theta,
        a,
        b,
        c
    )

    p_theta = np.clip(p_theta, 1e-9, 1 - 1e-9)

    # Para um item isolado:
    # P(U=1) = integral P(theta) f(theta) dtheta
    # P(U=0) = integral (1-P(theta)) f(theta) dtheta
    p_acerto_marginal = np.sum(p_theta * pesos_theta)
    p_erro_marginal = np.sum((1 - p_theta) * pesos_theta)

    p_acerto_marginal = np.clip(p_acerto_marginal, 1e-9, 1 - 1e-9)
    p_erro_marginal = np.clip(p_erro_marginal, 1e-9, 1 - 1e-9)

    n_acertos = np.sum(u == 1)
    n_erros = np.sum(u == 0)

    log_likelihood = (
        n_acertos * np.log(p_acerto_marginal)
        + n_erros * np.log(p_erro_marginal)
    )

    return -log_likelihood


def log_posterior_marginal_negativo_item_3pl(
    params,
    respostas_item,
    grade_theta,
    pesos_theta
):
    """
    Versão MAP marginal:
    log posterior = log verossimilhança marginal + log priors.
    """

    nll = log_verossimilhanca_marginal_item_3pl(
        params,
        respostas_item,
        grade_theta,
        pesos_theta
    )

    log_a, b, logit_c = params

    c = 1 / (1 + np.exp(-logit_c))

    log_prior_a = norm.logpdf(log_a, loc=0, scale=0.5)
    log_prior_b = norm.logpdf(b, loc=0, scale=2)
    log_prior_c = beta.logpdf(c, a=5, b=20)

    log_jacobiano_c = np.log(c) + np.log(1 - c)

    log_prior_total = (
        log_prior_a
        + log_prior_b
        + log_prior_c
        + log_jacobiano_c
    )

    return nll - log_prior_total


def calibrar_item_3pl_mml(
    respostas_item,
    grade_theta,
    pesos_theta
):
    """
    Calibra um item por MML/MAP marginal.
    """

    params_iniciais = np.array([
        np.log(1.0),
        0.0,
        logit(0.20)
    ])

    limites = [
        (np.log(0.01), np.log(5.0)),
        (-4.0, 4.0),
        (logit(0.01), logit(0.35))
    ]

    resultado = minimize(
        log_posterior_marginal_negativo_item_3pl,
        params_iniciais,
        args=(respostas_item, grade_theta, pesos_theta),
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


def calibrar_itens_3pl_mml(
    df_matriz,
    theta_min=-4,
    theta_max=4,
    n_pontos=41,
    media_prior=0,
    desvio_prior=1
):
    """
    Calibra todos os itens por 3PL MML/MAP marginal.
    """

    colunas_q = [
        c for c in df_matriz.columns
        if c.startswith("Q")
    ]

    grade_theta, pesos_theta = criar_quadratura_normal(
        theta_min=theta_min,
        theta_max=theta_max,
        n_pontos=n_pontos,
        media=media_prior,
        desvio=desvio_prior
    )

    resultados = []

    for item in colunas_q:
        print(f"Calibrando {item} via 3PL MML...")

        respostas_item = df_matriz[item].values.astype(float)

        resultado_item = calibrar_item_3pl_mml(
            respostas_item,
            grade_theta,
            pesos_theta
        )

        resultados.append({
            "ITEM": item,
            **resultado_item
        })

    return pd.DataFrame(resultados)