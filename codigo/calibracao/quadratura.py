# codigo/calibracao/quadratura.py

import numpy as np
from scipy.stats import norm


def criar_quadratura_normal(
    theta_min=-4,
    theta_max=4,
    n_pontos=41,
    media=0,
    desvio=1
):
    """
    Cria uma grade de quadratura para aproximar integrais em theta.

    Retorna:
        grade_theta: pontos de theta
        pesos: pesos normalizados da prior N(media, desvio)
    """

    grade_theta = np.linspace(theta_min, theta_max, n_pontos)

    pesos = norm.pdf(
        grade_theta,
        loc=media,
        scale=desvio
    )

    pesos = pesos / pesos.sum()

    return grade_theta, pesos