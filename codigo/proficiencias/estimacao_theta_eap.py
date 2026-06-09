import numpy as np
import pandas as pd

from scipy.stats import norm

from codigo.calibracao.modelos_irt import probabilidade_3pl


def estimar_theta_eap_aluno(
    respostas,
    a,
    b,
    c,
    grade_theta,
    prior
):
    """
    Estima theta por EAP para um aluno.
    """

    respostas = np.asarray(respostas, dtype=float)

    mascara = ~np.isnan(respostas)

    if mascara.sum() == 0:
        return np.nan

    u = respostas[mascara]
    aa = a[mascara]
    bb = b[mascara]
    cc = c[mascara]

    log_likelihood = np.zeros_like(grade_theta, dtype=float)

    for j, theta in enumerate(grade_theta):
        p = probabilidade_3pl(theta, aa, bb, cc)
        p = np.clip(p, 1e-9, 1 - 1e-9)

        log_likelihood[j] = np.sum(
            u * np.log(p) + (1 - u) * np.log(1 - p)
        )

    log_posterior = log_likelihood + np.log(prior)

    log_posterior = log_posterior - np.max(log_posterior)

    posterior = np.exp(log_posterior)
    posterior = posterior / posterior.sum()

    theta_eap = np.sum(grade_theta * posterior)

    return theta_eap



def estimar_thetas_eap(
    df_matriz,
    df_param_est,
    theta_min=-4,
    theta_max=4,
    n_pontos=81,
    media_prior=0,
    desvio_prior=1
):
    """
    Estima theta EAP para todos os alunos usando parâmetros 3PL.

    A lista de itens é definida a partir de df_param_est["ITEM"],
    permitindo estimar theta com subconjuntos de itens, por exemplo,
    quando alguns itens oficiais não possuem parâmetros.
    """

    df_param = df_param_est.copy()

    colunas_q = df_param["ITEM"].tolist()

    colunas_q = [
        item for item in colunas_q
        if item in df_matriz.columns
    ]

    df_param = df_param.set_index("ITEM").loc[colunas_q].reset_index()

    a = df_param["A_EST"].values.astype(float)
    b = df_param["B_EST"].values.astype(float)
    c = df_param["C_EST"].values.astype(float)

    grade_theta = np.linspace(theta_min, theta_max, n_pontos)

    prior = norm.pdf(
        grade_theta,
        loc=media_prior,
        scale=desvio_prior
    )

    prior = prior / prior.sum()

    resultados = []

    for idx, linha in df_matriz.iterrows():
        if idx % 1000 == 0:
            print(f"Estimando theta EAP: aluno {idx + 1}/{len(df_matriz)}")

        respostas = linha[colunas_q].values.astype(float)

        theta_eap = estimar_theta_eap_aluno(
            respostas=respostas,
            a=a,
            b=b,
            c=c,
            grade_theta=grade_theta,
            prior=prior
        )

        resultados.append({
            "INSC": linha["INSC"],
            "ID_DISCIPLINA": linha["ID_DISCIPLINA"],
            "CO_PROVA": linha["CO_PROVA"],
            "NOTA_REAL": linha["NOTA_REAL"],
            "N_RESPOSTAS_VALIDAS": linha.get("N_RESPOSTAS_VALIDAS", np.nan),
            "N_ACERTOS": linha.get("N_ACERTOS", np.nan),
            "THETA_EAP": theta_eap
        })

    return pd.DataFrame(resultados)