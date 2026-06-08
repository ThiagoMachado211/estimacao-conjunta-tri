from codigo.configuracoes import (
    ARQUIVO_MICRODADOS_ENEM,
    ARQUIVO_ITENS_ENEM
)

from codigo.leitura.leitura_enem import (
    ler_microdados_enem,
    ler_itens_enem
)


def main():
    print("Lendo microdados do ENEM...")

    colunas_microdados = [
        "NU_INSCRICAO",

        "TX_RESPOSTAS_CN",
        "TX_RESPOSTAS_CH",
        "TX_RESPOSTAS_LC",
        "TX_RESPOSTAS_MT",

        "TX_GABARITO_CN",
        "TX_GABARITO_CH",
        "TX_GABARITO_LC",
        "TX_GABARITO_MT",

        "NU_NOTA_CN",
        "NU_NOTA_CH",
        "NU_NOTA_LC",
        "NU_NOTA_MT",

        "CO_PROVA_CN",
        "CO_PROVA_CH",
        "CO_PROVA_LC",
        "CO_PROVA_MT",

        "TP_PRESENCA_CN",
        "TP_PRESENCA_CH",
        "TP_PRESENCA_LC",
        "TP_PRESENCA_MT",
    ]

    df_microdados = ler_microdados_enem(
        ARQUIVO_MICRODADOS_ENEM,
        colunas=colunas_microdados
    )

    print("Microdados lidos com sucesso.")
    print(f"Linhas: {df_microdados.shape[0]}")
    print(f"Colunas: {df_microdados.shape[1]}")

    print("\nLendo arquivo de itens do ENEM...")

    colunas_itens = [
        "CO_PROVA",
        "CO_POSICAO",
        "SG_AREA",
        "NU_PARAM_A",
        "NU_PARAM_B",
        "NU_PARAM_C",
        "TX_GABARITO"
    ]

    df_itens = ler_itens_enem(
        ARQUIVO_ITENS_ENEM,
        colunas=colunas_itens
    )

    print("Itens lidos com sucesso.")
    print(f"Linhas: {df_itens.shape[0]}")
    print(f"Colunas: {df_itens.shape[1]}")

    print("\nPrévia dos microdados:")
    print(df_microdados.head())

    print("\nPrévia dos itens:")
    print(df_itens.head())


if __name__ == "__main__":
    main()