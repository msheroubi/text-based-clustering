# Text-Based Clustering

Create clusters of text data by finding the cosine similarity between vectors using the sentence-transformers tokenizer.

## Requirements

```
pip install -U sentence-transformers
pip install pandas
```

## Usage

```
from cluster import TBCluster
import pandas as pd

TB = TBCluster()

BATCH_SIZE = 20000
THRESHOLD = 0.6

df = pd.read_csv('data.csv')

clusters = TB.cluster(df)
# clusters = TB.cluster(df, cluster_batch_size=BATCH_SIZE, threshold=THRESHOLD)

print(clusters)
```
threshold (0 to 1) controls how close the clusters are. Bring down cluster_batch_size if running into performance issues, recommended to leave at default.

