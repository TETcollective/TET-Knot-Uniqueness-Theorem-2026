import numpy as np
import matplotlib.pyplot as plt

# Stessi dati
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

# 50 run
n_runs = 50
n_iter_per_run = 200000
rng = np.random.default_rng(seed=2026)

results = []
for run in range(n_runs):
    samples = rng.choice(choices, size=n_iter_per_run, p=probabilities)
    freq = {k: np.sum(samples == k) / n_iter_per_run * 100 for k in choices}
    results.append(freq)

mean_freq = [np.mean([r[k] for r in results]) for k in choices]
std_freq = [np.std([r[k] for r in results]) for k in choices]

labels = [k.replace("_", " ").title().replace("3 1 ", "3₁ ") for k in choices]

# Bar plot cosmico
plt.figure(figsize=(12, 7))
x_pos = np.arange(len(choices))
bars = plt.bar(x_pos, mean_freq, yerr=std_freq, capsize=8, color='cyan', alpha=0.85, edgecolor='darkcyan', linewidth=1.5)

# Evidenzia il trefoil
bars[0].set_color('deepskyblue')
bars[0].set_edgecolor('navy')
bars[0].set_linewidth(3)

plt.xticks(x_pos, labels, rotation=20, ha='right', fontsize=12)
plt.ylabel("Frequenza Media (%)", fontsize=14)
plt.title("Distribuzione Finale su 50 Run Indipendenti ($10^8$ iterazioni)", fontsize=16)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()  # Mostra direttamente in Colab
