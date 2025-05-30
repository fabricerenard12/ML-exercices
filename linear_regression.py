import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score

def train(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Crée et entraîne un modèle de régression linéaire.
    Retourne le modèle entraîné.
    """

    # TODO: Effectuer la régression linéaire sur l'ensemble d'entraînement
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
    # raise NotImplementedError()

def eval(model: LinearRegression, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
    """
    Évalue le modèle sur l'ensemble de validation et renvoie les métriques.
    """

    # TODO: Effectuer la validation du modèle sur l'ensemble de validation
    y_pred = model.predict(X_val)
    return {
        "rmse": root_mean_squared_error(y_val, y_pred),
        "mse": mean_squared_error(y_val, y_pred),
        "r2":  r2_score(y_val, y_pred),
    }
    # raise NotImplementedError()

def plot(model: LinearRegression, X: np.ndarray, y: np.ndarray):
    plt.figure(figsize=(7, 5))
    plt.scatter(X, y, s=10, alpha=0.4, label="Données brutes")
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color="red", lw=2, label="Droite de régression")
    plt.xlabel("Revenu médian")
    plt.ylabel("Prix médian d'un maison")
    plt.title("Le marché immobilier californien - régression linéaire simple")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # raise NotImplementedError()

def main(test_ratio: float = 0.2, random_state: int = 0):
    data = fetch_california_housing()
    X = data.data[:, [data.feature_names.index("MedInc")]]
    y = data.target

    # TODO: Séparer votre jeu de données en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, shuffle=True
    )

    # TODO: Instancier votre modèle entraîné et effectuer la validation du modèle
    model = train(X_train, y_train)
    metrics = eval(model, X_val, y_val)

    print(f"Validation RMSE : {metrics['rmse']:.4f}")
    print(f"Validation MSE : {metrics['mse']:.4f}")
    print(f"Validation R²  : {metrics['r2']:.4f}")
    print(f"Coefficient     : {model.coef_[0]:.4f}")
    print(f"Intercept       : {model.intercept_:.4f}")

    plot(model, X, y)
    # raise NotImplementedError()

if __name__ == "__main__":
    main()