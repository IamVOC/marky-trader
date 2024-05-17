import numpy as np
import pandas as pd
from scipy.stats import norm


def confidence_interval(data, confidence_level=0.95):
  mean = np.mean(data)
  std = np.std(data)
  margin_of_error = std * norm.ppf((1 + confidence_level) / 2)
  lower_bound = mean - margin_of_error
  upper_bound = mean + margin_of_error
  return lower_bound, upper_bound