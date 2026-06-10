import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import norm, beta

from codigo.calibracao.modelos_irt import probabilidade_3pl
from codigo.calibracao.quadratura import criar_quadratura_normal


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def inv_logit(x):
    x = np.clip(x, -35, 35)
    return 1 / (1 + np.exp(-x))


def inicializar_parametros_3pl(df_matriz):
    """
    Inicializa parâmetros dos itens usando proporção de acertos.
    """

    colunas_q = [c for c in df_matriz.columns if c.startswith("Q")]

    resultados = []

    for item in colunas_q:
        respostas = df_matriz[item].values.astype(float)

        p = np.nanmean(respostas)
        p = np.clip(p, 0.05, 0.95)

        b_ini = -np.log(p / (1 - p))

        resultados.append(
            {
                "ITEM": item,
                "A_EST": 1.0,
                "B_EST": b_ini,
                "C_EST": 0.20,
            }
        )

    return pd.DataFrame(resultados)


def calcular_posterior_theta(
    matriz_respostas,
    a,
    b,
    c,
    grade_theta,
    pesos_theta
):
    """
    Etapa E do EM.

    Calcula posterior P(theta_k | respostas_aluno)
    para todos os alunos e todos os pontos de quadratura.
    """

    n_alunos = matriz_respostas.shape[0]
    n_theta = len(grade_theta)

    log_post = np.zeros((n_alunos, n_theta), dtype=float)

    for k, theta in enumerate(grade_theta):
        p = probabilidade_3pl(theta, a, b, c)
        p = np.clip(p, 1e-9, 1 - 1e-9)

        log_p = np.log(p)
        log_q = np.log(1 - p)

        temp = np.where(
            np.isnan(matriz_respostas),
            0,
            matriz_respostas * log_p + (1 - matriz_respostas) * log_q
        )

        log_post[:, k] = temp.sum(axis=1) + np.log(pesos_theta[k])

    log_post = log_post - np.max(log_post, axis=1, keepdims=True)

    posterior = np.exp(log_post)

    posterior = posterior / posterior.sum(axis=1, keepdims=True)

    return posterior


def log_posterior_item_em_negativo(
    params,
    respostas_item,
    posterior_theta,
    grade_theta
):
    """
    Etapa M para um item.

    Maximiza esperança da log-verossimilhança completa
    ponderada pela posterior dos thetas.
    """

    log_a, b, logit_c = params

    a = np.exp(log_a)
    c = inv_logit(logit_c)

    respostas = np.asarray(respostas_item, dtype=float)

    mascara = ~np.isnan(respostas)

    if mascara.sum() == 0:
        return 1e9

    u = respostas[mascara]
    w = posterior_theta[mascara, :]

    p = probabilidade_3pl(
        grade_theta[None, :],
        a,
        b,
        c
    )

    p = np.clip(p, 1e-9, 1 - 1e-9)

    log_likelihood = np.sum(
        w * (
            u[:, None] * np.log(p)
            + (1 - u[:, None]) * np.log(1 - p)
        )
    )

    log_prior_a = norm.logpdf(log_a, loc=0, scale=0.5)
    log_prior_b = norm.logpdf(b, loc=0, scale=2)
    log_prior_c = beta.logpdf(c, a=5, b=20)

    log_jacobiano_c = np.log(c) + np.log(1 - c)

    log_posterior = (
        log_likelihood
        + log_prior_a
        + log_prior_b
        + log_prior_c
        + log_jacobiano_c
    )

    return -log_posterior


def atualizar_item_em(
    respostas_item,
    posterior_theta,
    grade_theta,
    params_atuais
):
    """
    Atualiza um item na etapa M.
    """

    a0, b0, c0 = params_atuais

    params_iniciais = np.array(
        [
            np.log(np.clip(a0, 0.01, 5.0)),
            np.clip(b0, -4.0, 4.0),
            logit(np.clip(c0, 0.01, 0.35)),
        ]
    )

    limites = [
        (np.log(0.01), np.log(5.0)),
        (-4.0, 4.0),
        (logit(0.01), logit(0.35)),
    ]

    resultado = minimize(
        log_posterior_item_em_negativo,
        params_iniciais,
        args=(respostas_item, posterior_theta, grade_theta),
        method="L-BFGS-B",
        bounds=limites,
        options={"maxiter": 300},
    )

    if not resultado.success:
        return a0, b0, c0, False, resultado.message

    log_a, b, logit_c = resultado.x

    a = np.exp(log_a)
    c = inv_logit(logit_c)

    return a, b, c, True, resultado.message


def calibrar_itens_3pl_mml_em(
    df_matriz,
    theta_min=-4,
    theta_max=4,
    n_pontos=81,
    media_prior=0,
    desvio_prior=1,
    max_iter=5,
    tol=1e-4
):
    """
    Calibração 3PL por MML/EM.

    Retorna:
        df_parametros
    """

    colunas_q = [c for c in df_matriz.columns if c.startswith("Q")]

    matriz_respostas = df_matriz[colunas_q].values.astype(float)

    grade_theta, pesos_theta = criar_quadratura_normal(
        theta_min=theta_min,
        theta_max=theta_max,
        n_pontos=n_pontos,
        media=media_prior,
        desvio=desvio_prior,
    )

    df_param = inicializar_parametros_3pl(df_matriz)

    a = df_param["A_EST"].values.astype(float)
    b = df_param["B_EST"].values.astype(float)
    c = df_param["C_EST"].values.astype(float)

    for iteracao in range(1, max_iter + 1):
        print("=" * 70)
        print(f"Iteração EM {iteracao}/{max_iter}")
        print("=" * 70)

        a_ant = a.copy()
        b_ant = b.copy()
        c_ant = c.copy()

        posterior_theta = calcular_posterior_theta(
            matriz_respostas,
            a,
            b,
            c,
            grade_theta,
            pesos_theta,
        )

        convergencias = []

        for j, item in enumerate(colunas_q):
            print(f"Atualizando {item}...")

            respostas_item = matriz_respostas[:, j]

            a[j], b[j], c[j], convergiu, mensagem = atualizar_item_em(
                respostas_item,
                posterior_theta,
                grade_theta,
                params_atuais=(a[j], b[j], c[j]),
            )

            convergencias.append(convergiu)

        delta = max(
            np.max(np.abs(a - a_ant)),
            np.max(np.abs(b - b_ant)),
            np.max(np.abs(c - c_ant)),
        )

        print(f"Delta máximo: {delta:.8f}")
        print(f"Itens convergidos na etapa M: {sum(convergencias)}/{len(convergencias)}")

        if delta < tol:
            print("Convergência EM atingida.")
            break

    df_resultado = pd.DataFrame(
        {
            "ITEM": colunas_q,
            "A_EST": a,
            "B_EST": b,
            "C_EST": c,
            "CONVERGIU": True,
            "MENSAGEM": "EM finalizado",
        }
    )

    return df_resultado