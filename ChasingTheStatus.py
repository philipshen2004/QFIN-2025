from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


class TradingAlgorithm:

    def __init__(self):
        self.positions: Dict[
            str, int] = {}  # self.positions is a dictionary that will keep track of your position in each product
        # E.g. {"ABC": 2, "XYZ": -5, ...}
        # This will get automatically updated after each call to getOrders()

        #
        # TODO: Initialise any other variables that you want to use to keep track of things between getOrders() calls here
        #
        self.position_limit = 100
        self.UECMid = []
        self.SoberMid = []
        self.SMIFMid = []
        self.FAWAMid = []
        self.UECStd200 = 0
        self.UECMean = []
        self.UECMomentum = 0
        self.SoberMomentum = 0
        self.buyLag = 0



    def calculate_momentum(self, prices, window):
        """
        Calculate the momentum indicator as the percentage change over the specified window.
        Momentum = (Current Price / Price 'window' days ago) - 1.

        For the first 'window' elements, momentum will be None.
        """
        prices = np.array(prices)
        lnreturn = 0
        momentum = 0
        if len(prices) > window:
            lnreturn = (prices[-1] - prices[-window]) / prices[-window]
            if lnreturn > 0.01:
                return 20
            elif lnreturn < -0.01:
                return -20
        return 0


    def calculate_gbm_parameters(self, prices):
        """
        Calculate drift and volatility using the historical prices.
        """
        prices = np.array(prices)
        log_returns = np.log(prices[1:] / prices[:-1])
        volatility = np.std(log_returns)
        drift = np.mean(log_returns) - 0.5 * volatility ** 2
        return drift, volatility

    def forecast_price(self, current_price, days, drift, volatility):
        """
        Forecast the future price based on the GBM model.
        """
        brownian_motion = np.random.normal(0, 1, days)
        forecast = current_price * np.exp((drift * np.linspace(0, days, days)) + (volatility * brownian_motion))
        return np.mean(forecast)  # Return the expected price

    def decide_position(self, prices):
        """
        Decide whether to buy, sell, or hold based on the GBM model.
        """
        # Ensure we have enough data to calculate parameters
        if len(prices) < 10:
            return 0  # Insufficient data, hold

        current_price = prices[-1]
        drift, volatility = self.calculate_gbm_parameters(prices)

        # Forecast future price
        forecasted_price = self.forecast_price(current_price, days=10, drift=drift, volatility=volatility)

        # Define thresholds
        z_score = norm.ppf(0.95)  # 95% confidence interval
        lower_bound = forecasted_price / (1 + z_score * volatility)
        upper_bound = forecasted_price * (1 + z_score * volatility)

        # Make a decision
        if current_price < lower_bound:
            return min(10, self.position_limit)  # Buy 10 units or max allowed
        elif current_price > upper_bound:
            return max(-10, -self.position_limit)  # Sell 10 units or max allowed
        else:
            return 0  # Hold

    def sell(self, prices, days):
        if len(prices) > days:
            if prices[-1] < prices[-days]:
                return -20

        return 0

    def pairtradeShift(self, prices1, days):
        if len(prices1) > days:
            lnreturn = prices1[-1] / prices1[-days]
            if lnreturn > 0:
                return 1
            if lnreturn < 0:
                return -1

        return 0



    # This method will be called every timestamp with information about the new best bid and best ask for each product
    def getOrders(self, current_data: Dict[str, Dict[str, float]], order_data: Dict[str, int]) -> Dict[str, int]:
        # current_data is a dictionary that holds the current timestamp, best bid and best ask for each product
        # E.g. {"ABC": {"Timestamp": 134, "Bid": 34, "Ask" 38}, "XYZ": {"Timestamp": 134, "Bid": 1034, "Ask": 1038}, ...}

        # order_data is a dictionary that holds the quantity you will order for each product in this current timestamp
        # Intially the quantity for each product is set to 0 (i.e. no buy or sell orders will be sent if order_data is returned as it is)
        # To buy ABC for quantity x -> order_data["ABC"] = x (This will buy from the current best ask)
        # To sell ABC for quantity x -> order_data["ABC"] = -x (This will sell to the current best bid)

        #
        # TODO: Process new data and populate order_data here
        #

        UECPrice = ((current_data["UEC"]["Bid"] + current_data["UEC"]["Ask"]) / 2)
        SoberPrice = ((current_data["SOBER"]["Bid"] + current_data["SOBER"]["Ask"]) / 2)
        FAWAPrice = ((current_data["FAWA"]["Bid"] + current_data["FAWA"]["Ask"]) / 2)
        SMIFPrice = ((current_data["SMIF"]["Bid"] + current_data["SMIF"]["Ask"]) / 2)

        self.UECMid.append(UECPrice)
        self.SoberMid.append(SoberPrice)
        self.FAWAMid.append(FAWAPrice)
        self.SMIFMid.append(SMIFPrice)

        order_data['UEC'] = self.calculate_momentum(self.UECMid, 50)
        order_data['SOBER'] = self.sell(self.SoberMid, 10)
        order_data['FAWA'] = self.pairtradeShift(self.SMIFMid, 70)

        return order_data


# Leave this stuff as it is
team_algorithm = TradingAlgorithm()


def getOrders(current_data, positions):
    team_algorithm.positions = positions
    order_data = {product: 0 for product in current_data}
    return team_algorithm.getOrders(current_data, order_data)