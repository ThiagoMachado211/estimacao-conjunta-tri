import pandas as pd


def ler_microdados_enem(caminho_arquivo, colunas=None, nrows=None):
    """
    Lê os microdados do ENEM.

    Parâmetros
    ----------
    caminho_arquivo : str ou Path
        Caminho do arquivo MICRODADOS_ENEM_2009.csv.

    colunas : list ou None
        Lista de colunas que serão carregadas.
        Se None, carrega todas as colunas.

    nrows : int ou None
        Número máximo de linhas para leitura.

    Retorna
    -------
    pandas.DataFrame
    """

    df = pd.read_csv(
        caminho_arquivo,
        sep=";",
        encoding="latin1",
        usecols=colunas,
        nrows=nrows,
        low_memory=False
    )

    return df


def ler_itens_enem(caminho_arquivo, colunas=None):
    """
    Lê o arquivo de itens do ENEM.

    Parâmetros
    ----------
    caminho_arquivo : str ou Path
        Caminho do arquivo ITENS_PROVA_2009.csv.

    colunas : list ou None
        Lista de colunas que serão carregadas.
        Se None, carrega todas as colunas.

    Retorna
    -------
    pandas.DataFrame
    """

    df = pd.read_csv(
        caminho_arquivo,
        sep=";",
        encoding="latin1",
        usecols=colunas,
        low_memory=False
    )

    return df