import pandas as pd

from codigo.configuracoes import (
    PASTA_RESULTADOS_SIMULADO,
)


ITENS_ANALISAR = [
    "Q5",
    "Q7",
    "Q8",
    "Q27",
    "Q30",
]

FAIXAS = [
    (-1, 5, "0-5"),
    (5, 10, "6-10"),
    (10, 15, "11-15"),
    (15, 20, "16-20"),
    (20, 25, "21-25"),
    (25, 45, "26+"),
]


def classificar_faixa(n_acertos):
    for limite_inf, limite_sup, rotulo in FAIXAS:
        if limite_inf < n_acertos <= limite_sup:
            return rotulo

    return None


def main():
    caminho_matriz = (
        PASTA_RESULTADOS_SIMULADO /
        "matriz_amostra_experimento_a.csv"
    )

    df = pd.read_csv(caminho_matriz)

    colunas_q = [
        c for c in df.columns
        if c.startswith("Q")
    ]

    df["N_ACERTOS_TOTAL"] = (
        df[colunas_q]
        .sum(axis=1)
    )

    df["FAIXA_ACERTOS"] = (
        df["N_ACERTOS_TOTAL"]
        .apply(classificar_faixa)
    )

    resultados = []

    for faixa in [f[2] for f in FAIXAS]:
        temp = df[
            df["FAIXA_ACERTOS"] == faixa
        ].copy()

        linha = {
            "FAIXA_ACERTOS": faixa,
            "N_ALUNOS": len(temp),
            "MEDIA_ACERTOS_TOTAL": temp["N_ACERTOS_TOTAL"].mean(),
        }

        for item in ITENS_ANALISAR:
            linha[item] = temp[item].mean()

        resultados.append(linha)

    df_curvas = pd.DataFrame(resultados)

    print("\n" + "=" * 80)
    print("CURVAS EMPÍRICAS DOS ITENS PROBLEMÁTICOS")
    print("=" * 80)

    print(
        df_curvas.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    caminho_saida = (
        PASTA_RESULTADOS_SIMULADO /
        "curvas_empiricas_itens_problematicos.csv"
    )

    df_curvas.to_csv(
        caminho_saida,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nArquivo salvo em: {caminho_saida}")


if __name__ == "__main__":
    main()