import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy import stats

import winsound
from colorama import Fore, Style


class DataLoader:
    def __init__(self, path):
        self.path = path
        self.columns = None

    def load_data(self):
        data = pd.read_csv(self.path)
        data = data.dropna()
        data = data[data["cena"] > 0]
        return data

    def remove_outliers(self, data):
        data = data[data["godina proizvodnje"] >= 2002]
        data = data.drop(
            data[(data["godina proizvodnje"] <= 2005) & (data["cena"] > 10000)].index
        )
        data = data.drop(
            data[(data["godina proizvodnje"] <= 2007) & (data["cena"] > 20000)].index
        )
        data = data.drop(
            data[(data["godina proizvodnje"] <= 2010) & (data["cena"] > 22000)].index
        )
        data = data.drop(
            data[(data["godina proizvodnje"] >= 2010) & (data["cena"] < 2000)].index
        )
        data = data.drop(
            data[(data["godina proizvodnje"] >= 2015) & (data["cena"] < 4000)].index
        )
        data = data.drop(
            data[(data["godina proizvodnje"] >= 2017) & (data["cena"] < 6000)].index
        )
        data = data.drop(
            data[(data["godina proizvodnje"] >= 2019) & (data["cena"] < 8000)].index
        )
        data = data.drop(
            data[(data['snaga motora'] >= 175) & (data['cena'] < 2000)].index
        )
        data = data.drop(
            data[(data["snaga motora"] >= 200) & (data["cena"] < 3000)].index
        )
        data = data.drop(
            data[(data["snaga motora"] >= 250) & (data["cena"] < 3500)].index
        )
        data = data.drop(
            data[(data["snaga motora"] >= 300) & (data["cena"] < 6500)].index
        )
        data = data.drop(
            data[(data["snaga motora"] >= 400) & (data["cena"] < 12000)].index
        )
        data = data[data["kilometraža"] < 375000]
        data = data.drop(
            data[(data["kilometraža"] > 300000) & (data["cena"] > 30000)].index
        )
        data = data.drop(
            data[(data['kilometraža'] > 325000) & (data['cena'] > 25000)].index
        )
        data = data.drop(
            data[(data["kilometraža"] > 350000) & (data["cena"] > 20000)].index
        )
        return data

    def preprocess(self, data, train):
        if train:
            data = self.remove_outliers(data)
            data["cena"] = np.log(data["cena"])

        data.drop("kubikaža", axis=1, inplace=True)
        data.drop("model", axis=1, inplace=True)
        data.drop("broj vrata", axis=1, inplace=True)
        data.drop("gorivo", axis=1, inplace=True)

        if not train:
            encoded_make = pd.get_dummies(data["marka"], prefix="marka").reindex(
                columns=self.columns_make, fill_value=0
            )
            encoded_body = pd.get_dummies(
                data["karoserija"], prefix="karoserija"
            ).reindex(columns=self.columns_body, fill_value=0)
            encoded_transmission = pd.get_dummies(
                data["menjač"], prefix="menjač"
            ).reindex(columns=self.columns_transmission, fill_value=0)
            encoded_ac = pd.get_dummies(data["klima"], prefix="klima").reindex(
                columns=self.columns_ac, fill_value=0
            )
        else:
            encoded_make = pd.get_dummies(data["marka"], prefix="marka")
            encoded_body = pd.get_dummies(data["karoserija"], prefix="karoserija")
            encoded_transmission = pd.get_dummies(data["menjač"], prefix="menjač")
            encoded_ac = pd.get_dummies(data["klima"], prefix="klima")
            self.columns_make = encoded_make.columns
            self.columns_body = encoded_body.columns
            self.columns_transmission = encoded_transmission.columns
            self.columns_ac = encoded_ac.columns

        df_encoded = pd.concat(
            [data, encoded_body, encoded_make, encoded_transmission, encoded_ac], axis=1
        )

        df_encoded.drop(["marka"], axis=1, inplace=True)
        df_encoded.drop(["karoserija"], axis=1, inplace=True)
        df_encoded.drop(["menjač"], axis=1, inplace=True)
        df_encoded.drop(["klima"], axis=1, inplace=True)
        data = df_encoded
        return data

    def scale_params(self, data):
        self.mean_mileage = data["kilometraža"].mean()
        self.std_mileage = data["kilometraža"].std()
        self.mean_year = data["godina proizvodnje"].mean()
        self.std_year = data["godina proizvodnje"].std()
        self.mean_hp = data["snaga motora"].mean()
        self.std_hp = data["snaga motora"].std()

        data["kilometraža"] = (data["kilometraža"] - data["kilometraža"].mean()) / data[
            "kilometraža"
        ].std()
        data["godina proizvodnje"] = (
            data["godina proizvodnje"] - data["godina proizvodnje"].mean()
        ) / data["godina proizvodnje"].std()
        data["snaga motora"] = (
            data["snaga motora"] - data["snaga motora"].mean()
        ) / data["snaga motora"].std()
        return data


def cross_validate(X_train, Y_train, regressor):
    scores = cross_val_score(
        regressor, X_train, Y_train, cv=5, scoring="neg_mean_absolute_error"
    )
    predicted_values = np.exp(-scores)

    print("ElasticNet Cross-Validation MAE Scores:", predicted_values)
    print("ElasticNet Mean MAE:", predicted_values.mean())


def get_elastic_net_best_estimator(X_train, Y_train):
    elastic_net_param_grid = {
        "alpha": [0.0005, 0.001, 0.002, 0.005, 0.0075, 0.01, 0.025, 0.05],
        "l1_ratio": [0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.4, 0.5, 0.6, 0.8],
    }
    elastic_net_regressor = ElasticNet(random_state=0)
    elastic_net_grid_search = GridSearchCV(
        elastic_net_regressor,
        elastic_net_param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
    )
    elastic_net_grid_search.fit(X_train, Y_train)
    print("Best parameters for ElasticNet:", elastic_net_grid_search.best_params_)
    return elastic_net_grid_search


def get_random_forest_best_estimator(X_train, Y_train):
    random_forest_param_grid = {
        'n_estimators': [225, 250, 275],
        'max_depth': [None, 10, 15],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [1, 3, 5],
        'max_features': ['sqrt', 'log2', None],
    }
    random_forest_regressor = RandomForestRegressor(random_state=0)
    random_forest_grid_search = GridSearchCV(
        random_forest_regressor,
        random_forest_param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        verbose=2,
    )
    random_forest_grid_search.fit(X_train, Y_train)
    print("Best parameters for Random Forest:", random_forest_grid_search.best_params_)
    return random_forest_grid_search


def get_xgboost_best_estimator(X_train, Y_train):
    xgboost_param_grid = {
        "n_estimators": [100, 200, 300, 1000],
        "learning_rate": [0.01, 0.05, 0.1, 0.15],
        "max_depth": [3, 4, 5, 6, 7],
        "subsample": [0.8, 0.9, 1.0],
    }

    xgboost_regressor = XGBRegressor(random_state=0, verbosity=1)
    xgboost_grid_search = GridSearchCV(
        xgboost_regressor,
        xgboost_param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        verbose=2,
    )
    xgboost_grid_search.fit(X_train, Y_train)
    print("Best parameters for XGBoost:", xgboost_grid_search.best_params_)

    return xgboost_grid_search


def prep_data_for_test(data_loader, X_val):
    X_val["kilometraža"] = (
        X_val["kilometraža"] - data_loader.mean_mileage
    ) / data_loader.std_mileage
    X_val["godina proizvodnje"] = (
        X_val["godina proizvodnje"] - data_loader.mean_year
    ) / data_loader.std_year
    X_val["snaga motora"] = (
        X_val["snaga motora"] - data_loader.mean_hp
    ) / data_loader.std_hp
    return X_val


def test(regressor, X_val, Y_val, label):
    y_pred = np.exp(regressor.predict(X_val))
    mae = mean_absolute_error(Y_val, y_pred)
    print(f"MAE for {label}: {mae}")


def main():
    data_loader = DataLoader("vehicles/vehicles.csv")
    data = data_loader.load_data()

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    train_data = data_loader.preprocess(train_data, True)
    train_data = data_loader.scale_params(train_data)

    X_train = train_data.drop("cena", axis=1)
    Y_train = train_data["cena"]

    print(Fore.GREEN + "Fitting ElasticNet model...")
    # Best: alpha 0.0005, l1 0.05
    # elastic_net_grid_search = get_elastic_net_best_estimator(X_train, Y_train)
    # elastic_net_regressor = elastic_net_grid_search.best_estimator_
    elastic_net_regressor = ElasticNet(alpha=0.0005, l1_ratio=0.05, random_state=0)
    elastic_net_regressor.fit(X_train, Y_train)
    print("ElasticNet model fitted successfully.")

    print('-------------------------------')

    print("Fitting RandomForestRegressor model...")
    # Best: n_estimators 275, max_depth: None (default), min_samples_leaf 1 (default), max_features None (default), min_samples_split 10
    # random_forest_grid_search = get_random_forest_best_estimator(X_train, Y_train)
    # random_forest_regressor = random_forest_grid_search.best_estimator_
    random_forest_regressor = RandomForestRegressor(n_estimators= 275, min_samples_split=10, random_state=0)
    random_forest_regressor.fit(X_train, Y_train)
    print("RandomForestRegressor model fitted successfully.")

    print('-------------------------------')

    print("Fitting XGBRegressor model...")
    # Best parameters for XGBoost: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 1000, 'subsample': 0.9}
    # MAE for XGBoost regressor: 1693.20137255967
    # xgboost_grid_search = get_xgboost_best_estimator(X_train, Y_train)
    # xgboost_regressor = xgboost_grid_search.best_estimator_
    xgboost_regressor = XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.1, subsample=0.9, random_state=0)
    xgboost_regressor.fit(X_train, Y_train)
    print("XGBRegressor model fitted successfully.")
    
    print(Style.RESET_ALL)

    # cross_validate(X_train, Y_train, elastic_net_regressor)
    # cross_validate(X_train, Y_train, random_forest_regressor)
    # cross_validate(X_train, Y_train, xgboost_regressor)

    test_data = data_loader.preprocess(test_data, False)
    x_test = test_data.drop("cena", axis=1)
    y_test = test_data["cena"]

    x_test = prep_data_for_test(data_loader, x_test)
    test(elastic_net_regressor, x_test, y_test, label='Elastic net regressor')
    test(random_forest_regressor, x_test, y_test, label='Random forest regressor')
    test(xgboost_regressor, x_test, y_test, label="XGBoost regressor")
    winsound.Beep(880, 450)


if __name__ == "__main__":
    main()
