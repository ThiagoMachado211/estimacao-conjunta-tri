import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

PASTA = Path("resultados/enem_2009")

param = pd.read_csv(PASTA / "parametros_comparados_3pl_mml_em.csv")

print("\n" + "=" * 80)
print("ESTATÍSTICAS DOS PARÂMETROS")
print("=" * 80)

print("\nParâmetros oficiais:")
print(param[["A_REF", "B_REF", "C_REF"]].describe().round(6))

print("\nParâmetros estimados:")
print(param[["A_EST", "B_EST", "C_EST"]].describe().round(6))


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


print("\n" + "=" * 80)
print("MAE E RMSE DOS PARÂMETROS")
print("=" * 80)

for nome, ref, est in [
    ("A", "A_REF", "A_EST"),
    ("B", "B_REF", "B_EST"),
    ("C", "C_REF", "C_EST"),
]:
    dados = param[[ref, est]].dropna()

    print(f"\nParâmetro {nome}")
    print(f"MAE  = {mean_absolute_error(dados[ref], dados[est]):.6f}")
    print(f"RMSE = {rmse(dados[ref], dados[est]):.6f}")


print("\n" + "=" * 80)
print("CORRELAÇÕES DOS PARÂMETROS")
print("=" * 80)

print(param[["A_REF", "A_EST"]].corr().iloc[0, 1])
print(param[["B_REF", "B_EST"]].corr().iloc[0, 1])
print(param[["C_REF", "C_EST"]].corr().iloc[0, 1])


print("\n" + "=" * 80)
print("MAIORES ERROS ABSOLUTOS")
print("=" * 80)

param["ABS_ERRO_A"] = (param["A_EST"] - param["A_REF"]).abs()
param["ABS_ERRO_B"] = (param["B_EST"] - param["B_REF"]).abs()
param["ABS_ERRO_C"] = (param["C_EST"] - param["C_REF"]).abs()

print("\nTop 5 A:")
print(param.sort_values("ABS_ERRO_A", ascending=False)[
    ["ITEM", "CO_POSICAO", "A_REF", "A_EST", "ABS_ERRO_A"]
].head(5))

print("\nTop 5 B:")
print(param.sort_values("ABS_ERRO_B", ascending=False)[
    ["ITEM", "CO_POSICAO", "B_REF", "B_EST", "ABS_ERRO_B"]
].head(5))

print("\nTop 5 C:")
print(param.sort_values("ABS_ERRO_C", ascending=False)[
    ["ITEM", "CO_POSICAO", "C_REF", "C_EST", "ABS_ERRO_C"]
].head(5))