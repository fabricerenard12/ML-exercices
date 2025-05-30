import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def train(model_class, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
    # TODO: Implémenter l'entraînement du modèle
    model = model_class(**kwargs)
    model.fit(X_train, y_train)
    return model

    raise NotImplementedError()

def eval(model, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
    # TODO: Implémenter la validation du modèle
    y_pred = model.predict(X_val)
    return {
        "rmse": root_mean_squared_error(y_val, y_pred),
        "mse": mean_squared_error(y_val, y_pred),
        "r2":  r2_score(y_val, y_pred),
        "y_pred": y_pred
    }

def plot(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red", lw=2, label="y = ŷ")
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Valeurs prédites")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def create_model(name, model_class, X_train, y_train, X_val, y_val, **kwargs):
    print(f"\n----- {name} -----")
    model = train(model_class, X_train, y_train, **kwargs)
    results = eval(model, X_val, y_val)

    print(f"Validation RMSE : {results['rmse']:.4f}")
    print(f"Validation MSE  : {results['mse']:.4f}")
    print(f"Validation R²   : {results['r2']:.4f}")

    plot(y_val, results["y_pred"], f"Prédictions vs Réel - {name}")

def main(test_ratio: float = 0.2, random_state: int = 0):
    data = fetch_california_housing()
    X = data.data
    y = data.target

    # TODO: Séparer votre jeu de données en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, shuffle=True
    )

    # TODO: Instancier, entraîner et valider chaque modèle à partir de la fonction create_model
    create_model("Régression Linéaire", LinearRegression, X_train, y_train, X_val, y_val)
    create_model("Arbre de Décision", DecisionTreeRegressor, X_train, y_train, X_val, y_val, random_state=random_state)
    create_model("Forêt Aléatoire", RandomForestRegressor, X_train, y_train, X_val, y_val, random_state=random_state, n_estimators=100)
    create_model("Gradient Boosting", GradientBoostingRegressor, X_train, y_train, X_val, y_val, random_state=random_state, n_estimators=100)

if __name__ == "__main__":
    main()
