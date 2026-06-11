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

NROWS_TESTE = 100_000
USAR_NROWS_TESTE = True

# =========================
# Controle de execução
# =========================

MODO_VALIDACAO_ENEM = True

ID_DISCIPLINA_EXECUTAR = 4
AREA_ENEM_EXECUTAR = "MT"
# Apenas para LC
LINGUA_ESTRANGEIRA_EXECUTAR = 0 # (0 para inglês e 1 para espanhol)

CODIGOS_PROVA_EXECUTAR = [62]
USAR_AMOSTRA = True
TAMANHO_AMOSTRA = 5000
SEMENTE = 42

MINIMO_RESPOSTAS_VALIDAS = 5

# =========================
# Modelo TRI
# =========================

MODELO_CALIBRACAO = "3PL"

MAX_ITER = 5
TOL = 1e-4

A_MIN = 0.01
A_MAX = 5.00

B_MIN = -4.00
B_MAX = 4.00

C_MIN = 0.01
C_MAX = 0.35


# =========================
# EAP
# =========================

THETA_MIN = -4
THETA_MAX = 4
N_PONTOS_THETA = 41

MEDIA_PRIOR = 0
DESVIO_PRIOR = 1

# =========================
# Escala ENEM
# =========================

USAR_TRANSFORMACAO_LINEAR = True