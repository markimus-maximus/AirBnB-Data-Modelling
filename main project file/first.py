from DataHandler import DataHandler
import numpy as np
import pandas as pd
from pathlib import Path

dataset = DataHandler.csv_to_dataframe(Path('AirbnbDataSci/tabular_data/AirBnBData.csv'))
print(dataset)