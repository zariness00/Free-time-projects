import pandas as pd

read_file = pd.read_excel("/Users/zoryawka/Desktop/Coding/Free-time-projects/Python_ML/Raisins-classification/dataset/Raisin_Dataset.xlsx")
read_file.to_csv ("Raisin_Dataset.csv", index = None, header=True) 