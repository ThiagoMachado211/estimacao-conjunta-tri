from codigo.leitura.leitura_enem import (
    ler_microdados_enem,
    ler_itens_enem
)

from codigo.configuracoes import (
    ARQUIVO_MICRODADOS_ENEM,
    ARQUIVO_ITENS_ENEM,
    USAR_NROWS_TESTE,
    NROWS_TESTE,
    AREA_ENEM_EXECUTAR,
    CODIGOS_PROVA_EXECUTAR
)

from codigo.validacao_enem.preparar_base_enem import (
    filtrar_microdados_area,
    filtrar_itens_area
)

from codigo.configuracoes import ID_DISCIPLINA_EXECUTAR

from codigo.preprocessamento.matriz_respostas import (
    montar_matriz_respostas_enem,
    resumo_matriz_respostas
)

from codigo.validacao_enem.parametros_referencia_enem import (
    preparar_parametros_referencia_enem,
    verificar_parametros_referencia,
    comparar_gabaritos_enem
)

from codigo.configuracoes import (
    MINIMO_RESPOSTAS_VALIDAS,
    USAR_AMOSTRA,
    TAMANHO_AMOSTRA,
    SEMENTE
)

from codigo.preprocessamento.amostragem import (
    filtrar_alunos_respostas_validas,
    aplicar_amostragem
)

from codigo.calibracao.calibracao_3pl import calibrar_itens_3pl

from codigo.configuracoes import PASTA_RESULTADOS_ENEM

from codigo.avaliacao.comparacao_parametros import comparar_parametros

from codigo.visualizacao.graficos_parametros import (
    grafico_dispersao_parametro
)

from codigo.avaliacao.diagnostico_itens import (
    gerar_diagnostico_itens
)





def main():
    print("Lendo microdados do ENEM...")
    colunas_microdados = ["NU_INSCRICAO", "TX_RESPOSTAS_CN", "TX_RESPOSTAS_CH", "TX_RESPOSTAS_LC", "TX_RESPOSTAS_MT", "TX_GABARITO_CN", "TX_GABARITO_CH", "TX_GABARITO_LC", 
                          "TX_GABARITO_MT", "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "CO_PROVA_CN", "CO_PROVA_CH", "CO_PROVA_LC", "CO_PROVA_MT", "TP_PRESENCA_CN", 
                          "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT"]
    df_microdados = ler_microdados_enem(ARQUIVO_MICRODADOS_ENEM, colunas=colunas_microdados, nrows=NROWS_TESTE if USAR_NROWS_TESTE else None)
    print("Microdados lidos com sucesso.")

    print("---------------------------------------------------------------")
    print("\nFiltrando microdados por área, presença e código de prova...")
    df_enem_area = filtrar_microdados_area(df_microdados, area=AREA_ENEM_EXECUTAR, codigos_prova=CODIGOS_PROVA_EXECUTAR, apenas_presentes=True)
    print("Microdados filtrados:")
    print(f"Linhas: {df_enem_area.shape[0]}")
    print(f"Colunas: {df_enem_area.shape[1]}")
    print(df_enem_area.head())

    print("---------------------------------------------------------------")
    print("\nLendo arquivo de itens do ENEM...")
    colunas_itens = ["CO_PROVA", "CO_POSICAO", "SG_AREA", "NU_PARAM_A", "NU_PARAM_B", "NU_PARAM_C", "TX_GABARITO"]

    df_itens = ler_itens_enem(
        ARQUIVO_ITENS_ENEM,
        colunas=colunas_itens
    )

    print("Itens lidos com sucesso.")
    print(f"Linhas: {df_itens.shape[0]}")
    print(f"Colunas: {df_itens.shape[1]}")

    print("---------------------------------------------------------------")
    print("\nPrévia dos microdados:")
    print(df_microdados.head())

    print("---------------------------------------------------------------")
    print("\nPrévia dos itens:")
    print(df_itens.head())

    print("---------------------------------------------------------------")
    print("\nFiltrando itens por área e código de prova...")

    df_itens_area = filtrar_itens_area(
        df_itens,
        area=AREA_ENEM_EXECUTAR,
        codigos_prova=CODIGOS_PROVA_EXECUTAR
    )

    print("Itens filtrados:")
    print(f"Linhas: {df_itens_area.shape[0]}")
    print(f"Colunas: {df_itens_area.shape[1]}")
    print("---------------------------------------------------------------")
    print(df_itens_area.head())

    print("---------------------------------------------------------------")
    print("\nMontando matriz 0/1/NaN...")

    df_matriz = montar_matriz_respostas_enem(
        df_enem_area,
        id_disciplina=ID_DISCIPLINA_EXECUTAR
    )

    print("Matriz criada com sucesso.")
    print(f"Linhas: {df_matriz.shape[0]}")
    print(f"Colunas: {df_matriz.shape[1]}")

    print("---------------------------------------------------------------")
    print("\nPrévia da matriz:")
    print(df_matriz.head())

    print("---------------------------------------------------------------")
    print("\nResumo da matriz:")
    resumo = resumo_matriz_respostas(df_matriz)

    print("---------------------------------------------------------------")
    for chave, valor in resumo.items():
        print(f"{chave}: {valor}")

    print("---------------------------------------------------------------")
    print("\nPreparando parâmetros oficiais dos itens...")
    df_param_ref = preparar_parametros_referencia_enem(df_itens_area)
    print("Parâmetros oficiais preparados.")
    print(f"Linhas: {df_param_ref.shape[0]}")
    print(f"Colunas: {df_param_ref.shape[1]}")
    print("\nPrévia dos parâmetros oficiais:")
    print(df_param_ref.head())
    print("\nVerificando parâmetros oficiais...")
    problemas_param = verificar_parametros_referencia(df_param_ref)
    if problemas_param:
        print("Problemas encontrados:")
        for problema in problemas_param:
            print(f"- {problema}")
    else:
        print("Nenhum problema encontrado nos parâmetros oficiais.")

    print("---------------------------------------------------------------")
    print("\nItens com parâmetros ausentes:")
    print(
        df_param_ref[
            df_param_ref[
                ["A_REF", "B_REF", "C_REF"]
            ].isna().any(axis=1)
        ]
    )


    print("---------------------------------------------------------------")
    print("\nComparando gabarito dos microdados com gabarito dos itens...")
    resultado_gabarito = comparar_gabaritos_enem(
        df_enem_area,
        df_param_ref
    )
    print(f"Status: {resultado_gabarito['status']}")
    print(f"Mensagem: {resultado_gabarito['mensagem']}")
    if "tamanho_gabarito_microdados" in resultado_gabarito:
        print(
            "Tamanho gabarito microdados:",
            resultado_gabarito["tamanho_gabarito_microdados"]
        )
    if "tamanho_gabarito_itens" in resultado_gabarito:
        print(
            "Tamanho gabarito itens:",
            resultado_gabarito["tamanho_gabarito_itens"]
        )
    print("Total de diferenças:", resultado_gabarito["total_diferencas"])
    if resultado_gabarito["total_diferencas"]:
        print("\nDiferenças encontradas:")
        for dif in resultado_gabarito["diferencas"][:10]:
            print(dif)



    print("---------------------------------------------------------------")
    print("\nFiltrando alunos com poucas respostas válidas...")

    linhas_antes = len(df_matriz)

    df_matriz_filtrada = filtrar_alunos_respostas_validas(
        df_matriz,
        minimo_respostas_validas=MINIMO_RESPOSTAS_VALIDAS
    )

    linhas_depois = len(df_matriz_filtrada)

    print(f"Alunos antes do filtro: {linhas_antes}")
    print(f"Alunos depois do filtro: {linhas_depois}")
    print(f"Alunos removidos: {linhas_antes - linhas_depois}")

    print("\nAplicando amostragem...")

    df_matriz_amostra = aplicar_amostragem(
        df_matriz_filtrada,
        usar_amostra=USAR_AMOSTRA,
        tamanho_amostra=TAMANHO_AMOSTRA,
        semente=SEMENTE
    )

    print(f"Alunos após amostragem: {len(df_matriz_amostra)}")
    print("\nPrévia da base final para calibração:")
    print(df_matriz_amostra.head())



    print("---------------------------------------------------------------")
    print("\nIniciando calibração 3PL dos itens...")

    df_param_est, theta_inicial = calibrar_itens_3pl(
        df_matriz_amostra
    )

    print("\nCalibração concluída.")

    print(df_param_est)

    print("\nResumo de convergência:")
    print(df_param_est["CONVERGIU"].value_counts())



    print("---------------------------------------------------------------")
    print("\nSalvando parâmetros estimados...")

    PASTA_RESULTADOS_ENEM.mkdir(parents=True, exist_ok=True)

    PASTA_GRAFICOS = PASTA_RESULTADOS_ENEM / "graficos"

    PASTA_GRAFICOS.mkdir(
        parents=True,
        exist_ok=True
    )

    caminho_param_est = PASTA_RESULTADOS_ENEM / "parametros_estimados_3pl.csv"

    df_param_est.to_csv(
        caminho_param_est,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Parâmetros estimados salvos em: {caminho_param_est}")

    print("\nComparando parâmetros estimados com parâmetros oficiais...")

    df_param_comparados, df_metricas_param = comparar_parametros(
        df_param_ref,
        df_param_est
    )

    caminho_param_comparados = (
        PASTA_RESULTADOS_ENEM / "parametros_comparados_3pl.csv"
    )

    caminho_metricas_param = (
        PASTA_RESULTADOS_ENEM / "metricas_parametros_3pl.csv"
    )

    df_param_comparados.to_csv(
        caminho_param_comparados,
        index=False,
        encoding="utf-8-sig"
    )

    df_metricas_param.to_csv(
        caminho_metricas_param,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nMétricas dos parâmetros:")
    print(df_metricas_param)
    print(f"\nComparação salva em: {caminho_param_comparados}")
    print(f"Métricas salvas em: {caminho_metricas_param}")



    print("---------------------------------------------------------------")
    print("\nGerando gráficos dos parâmetros...")
    grafico_dispersao_parametro(
        df_param_comparados,
        "A_REF",
        "A_EST",
        "A_REF x A_EST",
        PASTA_GRAFICOS / "A_REF_x_A_EST.png"
    )
    grafico_dispersao_parametro(
        df_param_comparados,
        "B_REF",
        "B_EST",
        "B_REF x B_EST",
        PASTA_GRAFICOS / "B_REF_x_B_EST.png"
    )
    grafico_dispersao_parametro(
        df_param_comparados,
        "C_REF",
        "C_EST",
        "C_REF x C_EST",
        PASTA_GRAFICOS / "C_REF_x_C_EST.png"
    )
    print("Gráficos gerados com sucesso.")



    print("---------------------------------------------------------------")
    print("\nGerando diagnóstico dos itens...")
    df_diagnostico = gerar_diagnostico_itens(
        df_param_comparados
    )
    caminho_diagnostico = (
        PASTA_RESULTADOS_ENEM /
        "diagnostico_itens.csv"
    )

    df_diagnostico.to_csv(
        caminho_diagnostico,
        index=False,
        encoding="utf-8-sig"
    )
    print("\nTop 10 maiores erros em A:")

    print(
        df_diagnostico
        .sort_values("ABS_ERRO_A", ascending=False)
        [
            [
                "ITEM",
                "CO_POSICAO",
                "A_REF",
                "A_EST",
                "ABS_ERRO_A"
            ]
        ]
        .head(10)
    )
    print("\nTop 10 maiores erros em B:")

    print(
        df_diagnostico
        .sort_values("ABS_ERRO_B", ascending=False)
        [
            [
                "ITEM",
                "CO_POSICAO",
                "B_REF",
                "B_EST",
                "ABS_ERRO_B"
            ]
        ]
        .head(10)
    )

    print("\nTop 10 maiores erros em C:")

    print(
        df_diagnostico
        .sort_values("ABS_ERRO_C", ascending=False)
        [
            [
                "ITEM",
                "CO_POSICAO",
                "C_REF",
                "C_EST",
                "ABS_ERRO_C"
            ]
        ]
        .head(10)
    )





if __name__ == "__main__":
    main()