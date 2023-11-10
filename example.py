from cluster import TBCluster
import pandas as pd

# Path: example.py

TB = TBCluster()

df = pd.read_csv('data.csv')
clusters = TB.cluster(df)

print(clusters)

