from scipy.stats import pearsonr

from codigo.configuracoes import (
    ARQUIVO_MICRODADOS_ENEM,
    ARQUIVO_ITENS_ENEM,
    USAR_NROWS_TESTE,
    NROWS_TESTE,
    AREA_ENEM_EXECUTAR,
    LINGUA_ESTRANGEIRA_EXECUTAR,
    CODIGOS_PROVA_EXECUTAR,
    ID_DISCIPLINA_EXECUTAR,
    MINIMO_RESPOSTAS_VALIDAS,
    USAR_AMOSTRA,
    TAMANHO_AMOSTRA,
    SEMENTE,
    PASTA_RESULTADOS_ENEM,
    THETA_MIN,
    THETA_MAX,
    N_PONTOS_THETA,
    MEDIA_PRIOR,
    DESVIO_PRIOR,
)

from codigo.leitura.leitura_enem import (
    ler_microdados_enem,
    ler_itens_enem,
)

from codigo.validacao_enem.preparar_base_enem import (
    filtrar_microdados_area,
    filtrar_itens_area,
)

from codigo.preprocessamento.matriz_respostas import (
    montar_matriz_respostas_enem,
    resumo_matriz_respostas,
)

from codigo.validacao_enem.parametros_referencia_enem import (
    preparar_parametros_referencia_enem,
    verificar_parametros_referencia,
    comparar_gabaritos_enem,
)

from codigo.preprocessamento.amostragem import (
    filtrar_alunos_respostas_validas,
    aplicar_amostragem,
)

from codigo.calibracao.calibracao_3pl_mml_em import calibrar_itens_3pl_mml_em

from codigo.avaliacao.comparacao_parametros import comparar_parametros

from codigo.visualizacao.graficos_parametros import (
    grafico_dispersao_parametro,
)

from codigo.avaliacao.diagnostico_itens import (
    gerar_diagnostico_itens,
)

from codigo.proficiencias.estimacao_theta_eap import (
    estimar_thetas_eap,
)

from codigo.avaliacao.comparacao_proficiencias import (
    comparar_proficiencias,
)

from codigo.avaliacao.testes_diagnosticos import (
    estatisticas_parametros,
    estatisticas_thetas,
    correlacoes_diagnosticas,
    gerar_graficos_diagnosticos,
    salvar_tabelas_diagnosticas,
)

from codigo.avaliacao.comparacao_modelos import (
    adicionar_resultado_modelo,
    criar_tabela_comparativa,
)


def calcular_correlacao_segura(df, col_x, col_y):
    dados = df[[col_x, col_y]].dropna()

    if len(dados) < 2:
        return None

    if dados[col_x].std() == 0 or dados[col_y].std() == 0:
        return None

    return pearsonr(dados[col_x], dados[col_y])[0]


def main():
    if len(CODIGOS_PROVA_EXECUTAR) != 1:
        raise ValueError(
            "Por enquanto, o pipeline aceita apenas um código de prova por execução. "
            "Use CODIGOS_PROVA_EXECUTAR = [codigo]."
        )

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
        colunas=colunas_microdados,
        nrows=NROWS_TESTE if USAR_NROWS_TESTE else None,
    )

    print("Microdados lidos com sucesso.")

    print("---------------------------------------------------------------")
    print("Filtrando microdados por área, presença e código de prova...")

    df_enem_area = filtrar_microdados_area(
        df_microdados,
        area=AREA_ENEM_EXECUTAR,
        codigos_prova=CODIGOS_PROVA_EXECUTAR,
        apenas_presentes=True,
    )

    # ==========================================================
    # Tratamento específico para Linguagens (apenas a partir de 2010)
    # ==========================================================

#    if AREA_ENEM_EXECUTAR == "LC":
#        print("\nFiltrando língua estrangeira...")
#        df_enem_area = df_enem_area.merge(df_microdados[["NU_INSCRICAO", "TP_LINGUA"]], left_on="INSC", right_on="NU_INSCRICAO", how="left")
#        print(df_enem_area["TP_LINGUA"].value_counts(dropna=False).sort_index())
#        df_enem_area = df_enem_area[df_enem_area["TP_LINGUA"] == LINGUA_ESTRANGEIRA_EXECUTAR].copy()
#        print(
#            f"\nRegistros após filtro de língua "
#            f"({LINGUA_ESTRANGEIRA_EXECUTAR}): "
#            f"{len(df_enem_area)}"
#        )
#    else:
#        print(
#            "\nÁrea diferente de LC. "
#            "Nenhum filtro de língua aplicado."
#        ) 

    print(f"Linhas: {df_enem_area.shape[0]}")
    print(f"Colunas: {df_enem_area.shape[1]}")
    print(df_enem_area.head())



    print("---------------------------------------------------------------")
    print("Lendo arquivo de itens do ENEM...")

    colunas_itens = [
        "CO_PROVA",
        "CO_POSICAO",
        "SG_AREA",
        "NU_PARAM_A",
        "NU_PARAM_B",
        "NU_PARAM_C",
        "TX_GABARITO",
    ]

    df_itens = ler_itens_enem(
        ARQUIVO_ITENS_ENEM,
        colunas=colunas_itens,
    )

    print("Itens lidos com sucesso.")
    print(f"Linhas: {df_itens.shape[0]}")
    print(f"Colunas: {df_itens.shape[1]}")

    print("---------------------------------------------------------------")
    print("Filtrando itens por área e código de prova...")

    df_itens_area = filtrar_itens_area(
        df_itens,
        area=AREA_ENEM_EXECUTAR,
        codigos_prova=CODIGOS_PROVA_EXECUTAR,
    )

    print(f"Linhas: {df_itens_area.shape[0]}")
    print(f"Colunas: {df_itens_area.shape[1]}")
    print(df_itens_area.head())

    print("---------------------------------------------------------------")
    print("Montando matriz 0/1/NaN...")

    df_matriz = montar_matriz_respostas_enem(
        df_enem_area,
        id_disciplina=ID_DISCIPLINA_EXECUTAR,
    )

    print("Matriz criada com sucesso.")
    print(f"Linhas: {df_matriz.shape[0]}")
    print(f"Colunas: {df_matriz.shape[1]}")
    print(df_matriz.head())

    print("---------------------------------------------------------------")
    print("Resumo da matriz:")

    resumo = resumo_matriz_respostas(df_matriz)

    for chave, valor in resumo.items():
        print(f"{chave}: {valor}")

    print("---------------------------------------------------------------")
    print("Preparando parâmetros oficiais dos itens...")

    df_param_ref = preparar_parametros_referencia_enem(df_itens_area)

    print("Parâmetros oficiais preparados.")
    print(f"Linhas: {df_param_ref.shape[0]}")
    print(f"Colunas: {df_param_ref.shape[1]}")
    print(df_param_ref.head())

    print("---------------------------------------------------------------")
    print("Verificando parâmetros oficiais...")

    problemas_param = verificar_parametros_referencia(df_param_ref)

    if problemas_param:
        print("Problemas encontrados:")
        for problema in problemas_param:
            print(f"- {problema}")
    else:
        print("Nenhum problema encontrado nos parâmetros oficiais.")

    print("---------------------------------------------------------------")
    print("Itens com parâmetros ausentes:")

    df_itens_sem_parametros = df_param_ref[
        df_param_ref[["A_REF", "B_REF", "C_REF"]].isna().any(axis=1)
    ].copy()

    print(df_itens_sem_parametros)

    print("---------------------------------------------------------------")
    print("Comparando gabarito dos microdados com gabarito dos itens...")

    resultado_gabarito = comparar_gabaritos_enem(
        df_enem_area,
        df_param_ref,
    )

    print(f"Status: {resultado_gabarito['status']}")
    print(f"Mensagem: {resultado_gabarito['mensagem']}")

    if "tamanho_gabarito_microdados" in resultado_gabarito:
        print(
            "Tamanho gabarito microdados:",
            resultado_gabarito["tamanho_gabarito_microdados"],
        )

    if "tamanho_gabarito_itens" in resultado_gabarito:
        print(
            "Tamanho gabarito itens:",
            resultado_gabarito["tamanho_gabarito_itens"],
        )

    print("Total de diferenças:", resultado_gabarito["total_diferencas"])

    if resultado_gabarito["total_diferencas"]:
        print("Diferenças encontradas:")
        for dif in resultado_gabarito["diferencas"][:10]:
            print(dif)

    print("---------------------------------------------------------------")
    print("Filtrando alunos com poucas respostas válidas...")

    linhas_antes = len(df_matriz)

    df_matriz_filtrada = filtrar_alunos_respostas_validas(
        df_matriz,
        minimo_respostas_validas=MINIMO_RESPOSTAS_VALIDAS,
    )

    linhas_depois = len(df_matriz_filtrada)

    print(f"Alunos antes do filtro: {linhas_antes}")
    print(f"Alunos depois do filtro: {linhas_depois}")
    print(f"Alunos removidos: {linhas_antes - linhas_depois}")

    print("---------------------------------------------------------------")
    print("Aplicando amostragem...")

    df_matriz_amostra = aplicar_amostragem(
        df_matriz_filtrada,
        usar_amostra=USAR_AMOSTRA,
        tamanho_amostra=TAMANHO_AMOSTRA,
        semente=SEMENTE,
    )

    print(f"Alunos após amostragem: {len(df_matriz_amostra)}")
    print(df_matriz_amostra.head())

    print("---------------------------------------------------------------")
    print("Iniciando calibração 3PL MML-EM dos itens...")

    df_param_est = calibrar_itens_3pl_mml_em(
        df_matriz_amostra,
        theta_min=THETA_MIN,
        theta_max=THETA_MAX,
        n_pontos=N_PONTOS_THETA,
        media_prior=MEDIA_PRIOR,
        desvio_prior=DESVIO_PRIOR,
        max_iter=5,
        tol=1e-4,
    )

    print("Calibração concluída.")
    print(df_param_est)

    print("Resumo de convergência:")
    print(df_param_est["CONVERGIU"].value_counts())

    print("---------------------------------------------------------------")
    print("Salvando parâmetros estimados...")

    PASTA_RESULTADOS_ENEM.mkdir(parents=True, exist_ok=True)

    PASTA_GRAFICOS = PASTA_RESULTADOS_ENEM / "graficos"
    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

    caminho_param_est = (
        PASTA_RESULTADOS_ENEM / "parametros_estimados_3pl_mml_em.csv"
    )

    df_param_est.to_csv(
        caminho_param_est,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Parâmetros estimados salvos em: {caminho_param_est}")

    print("---------------------------------------------------------------")
    print("Comparando parâmetros estimados com parâmetros oficiais...")

    df_param_comparados, df_metricas_param = comparar_parametros(
        df_param_ref,
        df_param_est,
    )

    caminho_param_comparados = (
        PASTA_RESULTADOS_ENEM / "parametros_comparados_3pl_mml_em.csv"
    )

    caminho_metricas_param = (
        PASTA_RESULTADOS_ENEM / "metricas_parametros_3pl_mml_em.csv"
    )

    df_param_comparados.to_csv(
        caminho_param_comparados,
        index=False,
        encoding="utf-8-sig",
    )

    df_metricas_param.to_csv(
        caminho_metricas_param,
        index=False,
        encoding="utf-8-sig",
    )

    print("Métricas dos parâmetros:")
    print(df_metricas_param)

    print("---------------------------------------------------------------")
    print("Gerando gráficos dos parâmetros...")

    grafico_dispersao_parametro(
        df_param_comparados,
        "A_REF",
        "A_EST",
        "A_REF x A_EST",
        PASTA_GRAFICOS / "A_REF_x_A_EST.png",
    )

    grafico_dispersao_parametro(
        df_param_comparados,
        "B_REF",
        "B_EST",
        "B_REF x B_EST",
        PASTA_GRAFICOS / "B_REF_x_B_EST.png",
    )

    grafico_dispersao_parametro(
        df_param_comparados,
        "C_REF",
        "C_EST",
        "C_REF x C_EST",
        PASTA_GRAFICOS / "C_REF_x_C_EST.png",
    )

    print("Gráficos gerados com sucesso.")

    print("---------------------------------------------------------------")
    print("Gerando diagnóstico dos itens...")

    df_diagnostico = gerar_diagnostico_itens(
        df_param_comparados,
    )

    caminho_diagnostico = (
        PASTA_RESULTADOS_ENEM / "diagnostico_itens.csv"
    )

    df_diagnostico.to_csv(
        caminho_diagnostico,
        index=False,
        encoding="utf-8-sig",
    )

    print("Top 10 maiores erros em A:")
    print(
        df_diagnostico.sort_values("ABS_ERRO_A", ascending=False)[
            ["ITEM", "CO_POSICAO", "A_REF", "A_EST", "ABS_ERRO_A"]
        ].head(10)
    )

    print("Top 10 maiores erros em B:")
    print(
        df_diagnostico.sort_values("ABS_ERRO_B", ascending=False)[
            ["ITEM", "CO_POSICAO", "B_REF", "B_EST", "ABS_ERRO_B"]
        ].head(10)
    )

    print("Top 10 maiores erros em C:")
    print(
        df_diagnostico.sort_values("ABS_ERRO_C", ascending=False)[
            ["ITEM", "CO_POSICAO", "C_REF", "C_EST", "ABS_ERRO_C"]
        ].head(10)
    )

    print("---------------------------------------------------------------")
    print("Estimando proficiências por EAP com parâmetros estimados...")

    df_proficiencias = estimar_thetas_eap(
        df_matriz_amostra,
        df_param_est,
        theta_min=THETA_MIN,
        theta_max=THETA_MAX,
        n_pontos=N_PONTOS_THETA,
        media_prior=MEDIA_PRIOR,
        desvio_prior=DESVIO_PRIOR,
    )

    caminho_proficiencias = (
        PASTA_RESULTADOS_ENEM / "proficiencias_eap_3pl_mml_em.csv"
    )

    df_proficiencias.to_csv(
        caminho_proficiencias,
        index=False,
        encoding="utf-8-sig",
    )

    print(df_proficiencias.head())
    print(f"Proficiências salvas em: {caminho_proficiencias}")

    print("---------------------------------------------------------------")
    print("Comparando proficiências com nota real...")

    df_metricas_proficiencias = comparar_proficiencias(
        df_proficiencias,
    )

    print("Métricas das proficiências:")
    print(df_metricas_proficiencias)

    caminho_metricas_proficiencias = (
        PASTA_RESULTADOS_ENEM / "metricas_proficiencias_3pl_mml_em.csv"
    )

    df_metricas_proficiencias.to_csv(
        caminho_metricas_proficiencias,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Métricas salvas em: {caminho_metricas_proficiencias}")

    print("---------------------------------------------------------------")
    print("Estimando proficiências usando parâmetros oficiais do ENEM...")

    df_param_ref_para_eap = df_param_ref.rename(
        columns={
            "A_REF": "A_EST",
            "B_REF": "B_EST",
            "C_REF": "C_EST",
        }
    )[["ITEM", "A_EST", "B_EST", "C_EST"]].copy()

    qtd_itens_antes = len(df_param_ref_para_eap)

    df_param_ref_para_eap = df_param_ref_para_eap.dropna(
        subset=["A_EST", "B_EST", "C_EST"]
    ).copy()

    qtd_itens_depois = len(df_param_ref_para_eap)

    itens_validos_ref = df_param_ref_para_eap["ITEM"].tolist()

    print(f"Itens oficiais antes do filtro: {qtd_itens_antes}")
    print(f"Itens oficiais depois do filtro: {qtd_itens_depois}")
    print(f"Itens removidos: {qtd_itens_antes - qtd_itens_depois}")
    print("Itens válidos para EAP_REF:")
    print(itens_validos_ref)

    colunas_base = [
        "INSC",
        "ID_DISCIPLINA",
        "NOTA_REAL",
        "CO_PROVA",
        "N_RESPOSTAS_VALIDAS",
        "N_ACERTOS",
    ]

    df_matriz_ref = df_matriz_amostra[
        colunas_base + itens_validos_ref
    ].copy()

    df_proficiencias_ref = estimar_thetas_eap(
        df_matriz_ref,
        df_param_ref_para_eap,
        theta_min=THETA_MIN,
        theta_max=THETA_MAX,
        n_pontos=N_PONTOS_THETA,
        media_prior=MEDIA_PRIOR,
        desvio_prior=DESVIO_PRIOR,
    )

    df_proficiencias_ref = df_proficiencias_ref.rename(
        columns={
            "THETA_EAP": "THETA_EAP_REF",
        }
    )

    print("Proficiências com parâmetros oficiais:")
    print(df_proficiencias_ref.head())

    print("Resumo THETA_EAP_REF:")
    print(df_proficiencias_ref["THETA_EAP_REF"].describe())

    print("Quantidade de THETA_EAP_REF ausentes:")
    print(df_proficiencias_ref["THETA_EAP_REF"].isna().sum())

    caminho_proficiencias_ref = (
        PASTA_RESULTADOS_ENEM / "proficiencias_eap_parametros_ref.csv"
    )

    df_proficiencias_ref.to_csv(
        caminho_proficiencias_ref,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Proficiências com parâmetros oficiais salvas em: {caminho_proficiencias_ref}")

    print("---------------------------------------------------------------")
    print("Experimento com parâmetros oficiais:")

    corr_theta_ref = calcular_correlacao_segura(
        df_proficiencias_ref,
        "NOTA_REAL",
        "THETA_EAP_REF",
    )

    if corr_theta_ref is None:
        print("Correlação NOTA_REAL × THETA_EAP_REF: não calculável")
    else:
        print(f"Correlação NOTA_REAL × THETA_EAP_REF: {corr_theta_ref:.6f}")

    df_comparacao_theta_ref_est = df_proficiencias[
        ["INSC", "THETA_EAP"]
    ].merge(
        df_proficiencias_ref[["INSC", "THETA_EAP_REF", "NOTA_REAL"]],
        on="INSC",
        how="inner",
    )

    df_comparacao_theta_ref_est = df_comparacao_theta_ref_est.dropna(
        subset=["THETA_EAP", "THETA_EAP_REF"]
    )

    corr_theta_est_ref = calcular_correlacao_segura(
        df_comparacao_theta_ref_est,
        "THETA_EAP",
        "THETA_EAP_REF",
    )

    if corr_theta_est_ref is None:
        print("Correlação THETA_EAP_EST × THETA_EAP_REF: não calculável")
    else:
        print(
            "Correlação THETA_EAP_EST × THETA_EAP_REF:",
            f"{corr_theta_est_ref:.6f}",
        )

    print("---------------------------------------------------------------")
    print("Montando tabela comparativa dos modelos...")

    corr_nota_theta = calcular_correlacao_segura(
        df_proficiencias,
        "NOTA_REAL",
        "THETA_EAP",
    )

    resultados_modelos = []

    resultados_modelos = adicionar_resultado_modelo(
        resultados=resultados_modelos,
        nome_modelo="3PL_MML_EM",
        metricas_parametros=df_metricas_param,
        corr_nota_theta=corr_nota_theta,
        corr_theta_ref=corr_theta_est_ref,
    )

    df_comparativo_modelos = criar_tabela_comparativa(
        resultados_modelos,
    )

    print("\n")
    print("=" * 90)
    print("COMPARAÇÃO ENTRE MODELOS")
    print("=" * 90)

    print(
        df_comparativo_modelos.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    caminho_comparacao_modelos = (
        PASTA_RESULTADOS_ENEM / "comparacao_modelos.csv"
    )

    df_comparativo_modelos.to_csv(
        caminho_comparacao_modelos,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nTabela comparativa salva em: {caminho_comparacao_modelos}")

    print("---------------------------------------------------------------")
    print("Executando testes diagnósticos finais...")

    PASTA_DIAGNOSTICOS = PASTA_RESULTADOS_ENEM / "diagnosticos"
    PASTA_GRAFICOS_DIAGNOSTICOS = (
        PASTA_RESULTADOS_ENEM / "graficos_diagnosticos"
    )

    estatisticas_parametros(
        df_param_ref,
        df_param_est,
    )

    estatisticas_thetas(
        df_proficiencias,
        df_proficiencias_ref,
    )

    correlacoes_diagnosticas(
        df_proficiencias,
        df_proficiencias_ref,
    )

    gerar_graficos_diagnosticos(
        df_param_ref,
        df_param_est,
        df_proficiencias,
        df_proficiencias_ref,
        PASTA_GRAFICOS_DIAGNOSTICOS,
    )

    salvar_tabelas_diagnosticas(
        df_param_ref,
        df_param_est,
        df_proficiencias,
        df_proficiencias_ref,
        PASTA_DIAGNOSTICOS,
    )


    print("\n")
    print("=" * 80)
    print("CORRELAÇÕES PRINCIPAIS")
    print("=" * 80)

    # --------------------------------------------------
    # PARÂMETROS
    # --------------------------------------------------

    corr_a = df_metricas_param.loc[
        df_metricas_param["PARAMETRO"] == "A",
        "CORRELACAO"
    ].iloc[0]

    corr_b = df_metricas_param.loc[
        df_metricas_param["PARAMETRO"] == "B",
        "CORRELACAO"
    ].iloc[0]

    corr_c = df_metricas_param.loc[
        df_metricas_param["PARAMETRO"] == "C",
        "CORRELACAO"
    ].iloc[0]

    print("\nPARÂMETROS")
    print("-" * 40)
    print(f"Corr(A_REF, A_EST) = {corr_a:.6f}")
    print(f"Corr(B_REF, B_EST) = {corr_b:.6f}")
    print(f"Corr(C_REF, C_EST) = {corr_c:.6f}")

    # --------------------------------------------------
    # PROFICIÊNCIAS
    # --------------------------------------------------

    corr_theta_est_ref = calcular_correlacao_segura(
        df_comparacao_theta_ref_est,
        "THETA_EAP",
        "THETA_EAP_REF"
    )

    corr_nota_theta = calcular_correlacao_segura(
        df_proficiencias,
        "NOTA_REAL",
        "THETA_EAP"
    )

    corr_nota_theta_ref = calcular_correlacao_segura(
        df_proficiencias_ref,
        "NOTA_REAL",
        "THETA_EAP_REF"
    )

    corr_acertos_theta = calcular_correlacao_segura(
        df_proficiencias,
        "N_ACERTOS",
        "THETA_EAP"
    )

    corr_acertos_theta_ref = calcular_correlacao_segura(
        df_proficiencias_ref,
        "N_ACERTOS",
        "THETA_EAP_REF"
    )

    print("\nPROFICIÊNCIAS")
    print("-" * 40)
    print(f"Corr(THETA_EAP, THETA_EAP_REF)     = {corr_theta_est_ref:.6f}")
    print(f"Corr(NOTA_REAL, THETA_EAP)         = {corr_nota_theta:.6f}")
    print(f"Corr(NOTA_REAL, THETA_EAP_REF)     = {corr_nota_theta_ref:.6f}")
    print(f"Corr(N_ACERTOS, THETA_EAP)         = {corr_acertos_theta:.6f}")
    print(f"Corr(N_ACERTOS, THETA_EAP_REF)     = {corr_acertos_theta_ref:.6f}")

    print("\n" + "=" * 80)




if __name__ == "__main__":
    main()