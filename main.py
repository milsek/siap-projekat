import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNet
from scipy import stats


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
        data = data.drop(data[(data['godina proizvodnje'] <= 2005) & (data['cena'] > 5000)].index)
        data = data.drop(data[(data['godina proizvodnje'] <= 2007) & (data['cena'] > 7500)].index)
        data = data.drop(data[(data['godina proizvodnje'] >= 2010) & (data['cena'] < 4000)].index)
        data = data.drop(data[(data['godina proizvodnje'] >= 2015) & (data['cena'] < 7000)].index)
        data = data.drop(data[(data['godina proizvodnje'] >= 2019) & (data['cena'] < 12000)].index)
        data = data.drop(data[(data['snaga motora'] > 200) & (data['cena'] < 3000)].index)
        data = data.drop(data[(data['snaga motora'] > 250) & (data['cena'] < 3500)].index)
        data = data.drop(data[(data['snaga motora'] > 300) & (data['cena'] < 6500)].index)
        data = data[data["kilometraža"] < 375000]
        data = data.drop(data[(data['kilometraža'] > 250000) & (data['cena'] > 20000)].index)
        data = data.drop(data[(data['kilometraža'] > 200000) & (data['cena'] > 25000)].index)
        data = data[np.abs(stats.zscore(data["cena"])) <= 3]
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
            encoded_ac = pd.get_dummies(
                data["klima"], prefix="klima"
            ).reindex(columns=self.columns_ac, fill_value=0)
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


def test(data_loader, regressor, X_val, Y_val):
    # scale
    X_val["kilometraža"] = (
        X_val["kilometraža"] - data_loader.mean_mileage
    ) / data_loader.std_mileage
    X_val["godina proizvodnje"] = (
        X_val["godina proizvodnje"] - data_loader.mean_year
    ) / data_loader.std_year
    X_val["snaga motora"] = (
        X_val["snaga motora"] - data_loader.mean_hp
    ) / data_loader.std_hp

    y_pred = np.exp(regressor.predict(X_val))
    mae = mean_absolute_error(Y_val, y_pred)
    print(f"MAE: {mae}")


def main():
    data_loader = DataLoader("vehicles/vehicles.csv")
    data = data_loader.load_data()

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    train_data = data_loader.preprocess(train_data, True)
    train_data = data_loader.scale_params(train_data)

    X_train = train_data.drop("cena", axis=1)
    Y_train = train_data["cena"]

    alpha = 0.1
    l1_ratio = 0.45
    elastic_net_regressor = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0)
    elastic_net_regressor.fit(X_train, Y_train)

    test_data = data_loader.preprocess(test_data, False)
    x_test = test_data.drop("cena", axis=1)
    y_test = test_data["cena"]

    test(data_loader, elastic_net_regressor, x_test, y_test)


if __name__ == "__main__":
    main()
