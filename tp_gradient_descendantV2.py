
import os
import numpy as np
import matplotlib.pyplot as plt

# Fixer le hasard pour obtenir les memes resultats a chaque execution
np.random.seed(42)

# Create folder for figures
DOSSIER = "figures"
os.makedirs(DOSSIER, exist_ok=True)

def sauvegarder(nom):
    plt.tight_layout()
    plt.savefig(os.path.join(DOSSIER, nom), dpi=300, bbox_inches="tight")
    plt.close()


# Donnees
age = np.array([60, 48, 12, 30], dtype=float)
puissance = np.array([4, 7, 4, 5], dtype=float)
km = np.array([110, 90, 20, 45], dtype=float)

# Normalisation de y : division par 100 pour convertir
# les pourcentages en valeurs entre 0 et 1 et faciliter l'apprentissage.
y = np.array([22, 31, 45, 43], dtype=float).reshape(-1, 1) / 100

m = len(y)


# Q1 : Modele lineaire et calcul du gradient
X1 = np.column_stack([
    age / 100,
    km / 150,
    np.ones(m)
])  # Scaling lineaire sur 100 et 150

theta = np.random.uniform(-0.7, 0.7, (3, 1))
alpha = 0.004

for i in range(2):
    erreur = X1 @ theta - y
    theta = theta - alpha * (X1.T @ erreur) / m
    print("Iteration", i + 1, ":", theta.ravel())


# Fonctions de performance
def mae(X, theta):
    return np.mean(np.abs(X @ theta - y))


def mse(X, theta):
    return np.mean((X @ theta - y)**2)


def entrainer(X, theta, alpha, n):
    theta = theta.copy()  # Pour ne pas modifier la valeur originale de theta

    for _ in range(n):
        theta -= alpha * X.T @ (X @ theta - y) / m

    return theta


# Q2 : Hyperparametres
alphas = [
    0.001, 0.002, 0.004, 0.006, 0.008,
    0.01, 0.02, 0.03, 0.05, 0.07,
    0.1, 0.15, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.8, 1.0, 1.2,
    1.4, 1.5
]

iterations = [100, 500, 1000, 5000, 10000]


# Q3 : Tableau de performance
T = np.zeros((len(iterations), len(alphas)))
T_MSE = np.zeros((len(iterations), len(alphas)))

# Meme initialisation pour comparer correctement les combinaisons
theta_initial = np.random.uniform(-0.7, 0.7, (3, 1))

for i, n in enumerate(iterations):
    for j, a in enumerate(alphas):
        th = entrainer(X1, theta_initial, a, n)

        T[i, j] = mae(X1, th)
        T_MSE[i, j] = mse(X1, th)


print("\nQ3 : Tableau MAE")
print("Colonnes alpha :", alphas)
print("Lignes iterations :", iterations)
print(T)

print("=================")

print("\nQ3 : Tableau MSE")
print("Colonnes alpha :", alphas)
print("Lignes iterations :", iterations)
print(T_MSE)


# Q4 : Performance selon alpha
for i, n in enumerate(iterations):
    plt.plot(alphas, T[i], marker="o", label=f"{n} iterations")

plt.xlabel("Alpha")
plt.ylabel("MAE")
plt.title("Q4 : Performance selon alpha")
plt.legend()
sauvegarder("Q4_performance_alpha.png")


# Q5 : Performance selon le nombre d'iterations
for j, a in enumerate(alphas):
    plt.plot(iterations, T[:, j], marker="o", label=f"alpha={a}")

plt.xlabel("Nombre d'iterations")
plt.ylabel("MAE")
plt.title("Q5 : Performance selon les iterations")
plt.legend()
sauvegarder("Q5_performance_iterations.png")


# Q6 : Ajouter des niveaux aux hyperparametres
alphas6 = alphas
iterations6 = iterations

T6 = np.zeros((len(iterations6), len(alphas6)))

# Meme initialisation pour toutes les combinaisons
theta_initial6 = np.random.uniform(-0.7, 0.7, (3, 1))

for i, n in enumerate(iterations6):
    for j, a in enumerate(alphas6):
        th = entrainer(X1, theta_initial6, a, n)
        T6[i, j] = mae(X1, th)


print("\nQ6 : Tableau avec plus de niveaux")
print(T6)


# Graphique supplementaire Q6 : MAE selon alpha
for i, n in enumerate(iterations6):
    plt.plot(alphas6, T6[i], marker="o", label=f"{n} iterations")

plt.xlabel("Alpha")
plt.ylabel("MAE")
plt.title("Q6 : Performance selon alpha")
plt.legend(fontsize=8)
sauvegarder("Q6_performance_alpha.png")


# Graphique supplementaire Q6 : MAE selon iterations
for j, a in enumerate(alphas6):
    plt.plot(iterations6, T6[:, j], marker="o", label=f"alpha={a}")

plt.xlabel("Nombre d'iterations")
plt.ylabel("MAE")
plt.title("Q6 : Performance selon les iterations")
plt.legend(fontsize=8)
sauvegarder("Q6_performance_iterations.png")


# Q7 : Anomalie du pretraitement
print("\nQ7 : Les variables sont divisees par des constantes fixes.")
print("age / 100, km / 150 et y / 100.")
print("Cela permet de travailler avec des valeurs plus petites.")


# Q8 : Nouvelle initialisation
theta0 = np.random.uniform(-0.7, 0.7, (3, 1))
theta8 = entrainer(X1, theta0, 0.01, 10000)

print("\nQ8 : MAE avec nouvelle initialisation =", mae(X1, theta8))


# Q9 : Modele quadratique a trois variables
x1 = age / 100
x2 = puissance / 10
x3 = km / 150

X2 = np.column_stack([
    x1**2, x2**2, x3**2,
    x1*x2, x1*x3, x2*x3,
    x1, x2, x3, np.ones(m)
])

print("\nQ9 : Matrice quadratique")
print(X2)


# Q10 : Initialisation des parametres quadratiques
theta2 = np.random.uniform(-0.7, 0.7, (10, 1))

print("\nQ10 : Parametres initiaux")
print(theta2.ravel())


# Q11 : Performance du modele quadratique
T2 = np.zeros((len(iterations), len(alphas)))

for i, n in enumerate(iterations):
    for j, a in enumerate(alphas):
        th = entrainer(X2, theta2, a, n)
        T2[i, j] = mae(X2, th)

print("\nQ11 : Tableau MAE quadratique")
print(T2)


# Q12 : Comparaison des modeles selon alpha
plt.plot(alphas, T[-1], marker="o", label="Lineaire")
plt.plot(alphas, T2[-1], marker="s", label="Quadratique")

plt.xlabel("Alpha")
plt.ylabel("MAE")
plt.title("Q12 : Comparaison selon alpha")
plt.legend()
sauvegarder("Q12_comparaison_alpha.png")


# Q13 : Comparaison selon le nombre d'iterations
plt.plot(iterations, T[:, 1], marker="o", label="Lineaire")
plt.plot(iterations, T2[:, 1], marker="s", label="Quadratique")

plt.xlabel("Nombre d'iterations")
plt.ylabel("MAE")
plt.title("Q13 : Comparaison selon les iterations")
plt.legend()
sauvegarder("Q13_comparaison_iterations.png")


print("\nToutes les figures ont ete sauvegardees dans le dossier figures.")
