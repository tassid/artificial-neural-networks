# %% [markdown]
# MLP: comparação de ativações (Tanh vs ReLU) com mesma inicialização de pesos
# Funciona em Colab e PyCharm. Requer: numpy, matplotlib, scikit-learn.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ---------- Configurações ----------
DATASET = "moons"   # "moons" ou "iris"
N_NEURONS = 3       # poucos neurônios para diferenças mais claras
SEED_DATA = 7       # semente do dataset
SEED_WEIGHTS = 13   # semente dos pesos (mesma para ambas as ativações)
MAX_ITER = 5000
ALPHA = 1e-4

def make_data(dataset: str):
    if dataset.lower() == "moons":
        X, y = make_moons(n_samples=300, noise=0.25, random_state=SEED_DATA)
        title = "Moons (não linear)"
    elif dataset.lower() == "iris":
        iris = load_iris()
        # Usar apenas 2 features para plot 2D (pétala comprimento/largura)
        X = iris.data[:, 2:4]
        # Duas classes para fronteira binária clara (setosa vs não-setosa)
        y = (iris.target == 0).astype(int)
        title = "Iris (2D: pétalas; binário setosa vs. outros)"
    else:
        raise ValueError("DATASET deve ser 'moons' ou 'iris'")
    return X, y, title

def build_mlp(activation: str):
    # Mesma random_state -> mesma inicialização de pesos
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(N_NEURONS,),
                      activation=activation,
                      alpha=ALPHA,
                      max_iter=MAX_ITER,
                      random_state=SEED_WEIGHTS)
    )

def plot_decision_boundary(model, X, y, title, savepath):
    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.30)            # sem definir cores específicas
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')  # cores padrão do matplotlib
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(savepath, dpi=160)
    plt.show()

def run(dataset: str):
    X, y, ds_title = make_data(dataset)
    mlp_tanh = build_mlp("tanh")
    mlp_relu = build_mlp("relu")
    mlp_tanh.fit(X, y)
    mlp_relu.fit(X, y)

    plot_decision_boundary(mlp_tanh, X, y,
        f"MLP ({N_NEURONS} neurônios, ativação Tanh) — {ds_title}",
        f"mlp_tanh_{dataset}.png")
    plot_decision_boundary(mlp_relu, X, y,
        f"MLP ({N_NEURONS} neurônios, ativação ReLU) — {ds_title}",
        f"mlp_relu_{dataset}.png")

    print("Acurácias:",
          "Tanh =", mlp_tanh.score(X, y),
          "| ReLU =", mlp_relu.score(X, y))

if __name__ == "__main__":
    # Troque DATASET para "iris" se quiser comparar no Iris 2D
    run(DATASET)
