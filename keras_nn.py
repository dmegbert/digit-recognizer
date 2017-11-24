import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten
from sklearn.model_selection import train_test_split
from subprocess import check_output

print(check_output(['ls', 'data']).decode('utf-8'))
