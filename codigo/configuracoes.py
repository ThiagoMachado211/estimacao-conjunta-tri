# codigo/configuracoes.py

from pathlib import Path

# =========================
# Caminhos principais
# =========================

PASTA_RAIZ = Path(__file__).resolve().parents[1]

PASTA_DADOS = PASTA_RAIZ / "dados"
PASTA_RESULTADOS = PASTA_RAIZ / "resultados"

PASTA_DADOS_ENEM = PASTA_DADOS / "enem"
PASTA_RESULTADOS_ENEM = PASTA_RESULTADOS / "enem_2009"

ARQUIVO_MICRODADOS_ENEM = PASTA_DADOS_ENEM / "MICRODADOS_ENEM_2009.csv"
ARQUIVO_ITENS_ENEM = PASTA_DADOS_ENEM / "ITENS_PROVA_2009.csv"

# =========================
# Controle de execução
# =========================

MODO_VALIDACAO_ENEM = True

ID_DISCIPLINA_EXECUTAR = 4
AREA_ENEM_EXECUTAR = "MT"

CODIGOS_PROVA_EXECUTAR = [61]

USAR_AMOSTRA = True
TAMANHO_AMOSTRA = 20000
SEMENTE = 42

# =========================
# Modelo TRI
# =========================

MODELO_CALIBRACAO = "2PL"
# Opções futuras:
# "2PL"
# "3PL_C_FIXO"
# "3PL"

C_FIXO = 0.20

MAX_ITER = 50
TOL = 1e-4

# =========================
# EAP
# =========================

THETA_MIN = -4
THETA_MAX = 4
N_PONTOS_THETA = 81

MEDIA_PRIOR = 0
DESVIO_PRIOR = 1

# =========================
# Escala ENEM
# =========================

USAR_TRANSFORMACAO_LINEAR = True