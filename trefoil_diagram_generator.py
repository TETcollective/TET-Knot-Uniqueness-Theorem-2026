import numpy as np
import matplotlib.pyplot as plt

# Database nodi
knots = {
    "3_1_trefoil": {"c": 3, "Lk": 6, "non_abelian": True},
    "3_1_mirror":   {"c": 3, "Lk": 6, "non_abelian": True},
    "4_1":         {"c": 4, "Lk": 0, "non_abelian": False},
    "5_1":         {"c": 5, "Lk": 10, "non_abelian": False},
    "5_2":         {"c": 5, "Lk": 9,  "non_abelian": False},
    "unknot":      {"c": 0, "Lk": 0, "non_abelian": False}
}

weights = {"CS": 0.6, "L": 0.3, "fusion": 0.1}
target_Lk = 6

def payoff(knot_name):
    k = knots[knot_name]
    E_CS = 1.0 / (k["c"] + 1) if k["c"] >= 3 else 0.01
    link_score = 1.0 / (abs(k["Lk"] - target_Lk) + 1)
    fusion_bonus = 1.0 if k["non_abelian"] else 0.2
    if "mirror" in knot_name:
        fusion_bonus *= 0.5
    return (weights["CS"] * E_CS + weights["L"] * link_score + weights["fusion"] * fusion_bonus)

choices = list(knots.keys())
probabilities = np.array([payoff(k) for k in choices])
probabilities /= probabilities.sum()

# Simulazione lunga
n_iterations = 1000000
rng = np.random.default_rng(seed=42)
samples = rng.choice(choices, size=n_iterations, p=probabilities)

# Convergenza cumulativa corretta
cumulative = np.zeros((n_iterations, len(choices)))
counts = np.zeros(len(choices))
for i in range(n_iterations):
    idx = choices.index(samples[i])
    counts[idx] += 1
    cumulative[i] = (counts / (i + 1)) * 100

# Grafico cosmico
plt.figure(figsize=(12, 7))
colors = plt.cm.coolwarm(np.linspace(0, 1, len(choices)))
for j, k in enumerate(choices):
    label = k.replace("_", " ").title().replace("3 1 ", "3₁ ")
    plt.plot(cumulative[:, j], label=label, color=colors[j], linewidth=2.5)

plt.axhline(y=99.987, color='cyan', linestyle='--', linewidth=2, label='Equilibrio 3₁ Trefoil')
plt.title("Convergenza Monte Carlo alla Configurazione Primordiale del Trefoil", fontsize=16)
plt.xlabel("Iterazioni", fontsize=14)
plt.ylabel("Frequenza Cumulativa (%)", fontsize=14)
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()  # Mostra in Colab




