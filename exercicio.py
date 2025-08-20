import numpy as np
import matplotlib.pyplot as plt

# População fictícia: horas/dia ouvindo música

rng =np.random.default_rng(1)

pop = np.clip(rng.lognormal(mean=0.7, sigma=0.7, size=200_000), 0, 10)
mu = pop.mean()
sigma = pop.std(ddof=0)

# Médias amostrais (n=30)

n = 30
reps = 5000
idx = rng.integers(0, len(pop), size=(reps, n))
medias = pop[idx].mean(axis=1)

# Curva Normal teórica

x = np.linspace(min(medias), max(medias), 300)
theo_sigma = sigma / np.sqrt(n)
pdf = (1 / (theo_sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu)/theo_sigma)**2)

# Gráfico

plt.hist(medias, bins=50, density=True, alpha=0.6, label="Médias amostrais")
plt.plot(x, pdf, linewidth=2, label="Normal teórica (CLT)")
plt.title(f"Distribuição das médias amostrais (n={n})\nMédia populacional ≈ {mu:.2f}")
plt.xlabel("Média da amostra (horas/dia)")
plt.ylabel("Densidade")
plt.legend()
plt.show()
