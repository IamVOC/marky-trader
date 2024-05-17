from gym_trading_env.downloader import download
import datetime

download(
exchange_names = ["binance"],
symbols= ["BTC/USDT"],
timeframe= "1h",
dir = "data",
since = datetime.datetime(2019, 1, 1),
until = datetime.datetime(2024, 1, 1),
)