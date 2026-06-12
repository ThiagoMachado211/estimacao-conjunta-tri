import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


ARQUIVO = "resultados/simulado/comparacao_parametros_somente_enem_experimento_a.csv"

df = pd.read_csv(ARQUIVO)

df = df.dropna(subset=["B_REF", "B_EST"])

X = df[["B_EST"]]
y = df["B_REF"]

modelo = LinearRegression()
modelo.fit(X, y)

alpha = modelo.intercept_
beta = modelo.coef_[0]

pred = modelo.predict(X)

r2 = r2_score(y, pred)
corr = pearsonr(df["B_REF"], df["B_EST"])[0]

print("\n===================================")
print("AJUSTE DE ESCALA - PARAMETRO B")
print("===================================")

print(f"N itens      : {len(df)}")
print(f"Correlacao   : {corr:.6f}")
print(f"R²           : {r2:.6f}")
print(f"Intercepto   : {alpha:.6f}")
print(f"Inclinacao   : {beta:.6f}")

print("\nTransformacao estimada:")
print(f"B_REF = {alpha:.6f} + {beta:.6f} * B_EST")



plt.figure(figsize=(8,6))

plt.scatter(
    df["B_EST"],
    df["B_REF"]
)

xmin = df["B_EST"].min()
xmax = df["B_EST"].max()

x = np.linspace(xmin, xmax, 100)
y = alpha + beta * x

plt.plot(x, y)

plt.xlabel("B_EST")
plt.ylabel("B_REF")
plt.title("B_REF x B_EST")

plt.grid(True)

plt.show()