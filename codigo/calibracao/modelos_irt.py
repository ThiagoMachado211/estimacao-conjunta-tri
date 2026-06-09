import numpy as np


def sigmoid(x):
    """
    Função logística numericamente estável.
    """

    x = np.clip(x, -35, 35)

    return 1.0 / (1.0 + np.exp(-x))


def probabilidade_3pl(theta, a, b, c):
    """
    Probabilidade de acerto no modelo 3PL.

    P(X=1|theta) = c + (1-c) / (1 + exp(-a(theta-b)))
    """

    return c + (1.0 - c) * sigmoid(a * (theta - b))


def log_likelihood_item_3pl(params, theta, respostas):
    """
    Log-verossimilhança negativa de um item no modelo 3PL.

    params = [log_a, b, logit_c]

    Transformações:
        a = exp(log_a), garantindo a > 0
        c = sigmoid(logit_c), garantindo 0 < c < 1
    """

    log_a, b, logit_c = params

    a = np.exp(log_a)
    c = sigmoid(logit_c)

    respostas = np.asarray(respostas, dtype=float)
    theta = np.asarray(theta, dtype=float)

    mascara = ~np.isnan(respostas)

    if mascara.sum() == 0:
        return 1e9

    u = respostas[mascara]
    th = theta[mascara]

    p = probabilidade_3pl(th, a, b, c)

    p = np.clip(p, 1e-9, 1 - 1e-9)

    ll = u * np.log(p) + (1 - u) * np.log(1 - p)

    return -np.sum(ll)


def log_likelihood_aluno_3pl(theta, respostas, a, b, c):
    """
    Log-verossimilhança negativa de um aluno no modelo 3PL.

    theta é escalar.
    """

    respostas = np.asarray(respostas, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    mascara = ~np.isnan(respostas)

    if mascara.sum() == 0:
        return 1e9

    u = respostas[mascara]
    aa = a[mascara]
    bb = b[mascara]
    cc = c[mascara]

    p = probabilidade_3pl(theta, aa, bb, cc)

    p = np.clip(p, 1e-9, 1 - 1e-9)

    ll = u * np.log(p) + (1 - u) * np.log(1 - p)

    return -np.sum(ll)