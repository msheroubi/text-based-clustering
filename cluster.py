import pandas as pd
import random
from sentence_transformers import SentenceTransformer, util
import torch
import logging

logging.basicConfig(filename='job.log', encoding='utf-8', level=logging.DEBUG)

class TBCluster:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def cluster(self, df, cluster_batch_size=20000, threshold=0.6):
        logging.info("Initializing LLM...")
        model = self.model

        try:
            df = df.dropna().astype(str).reset_index(drop=True)
            
            text = df['text']
            index = df['index']
            NUM_BATCHES = len(text) // cluster_batch_size
            logging.info(f"Running clustering in {NUM_BATCHES} batches.")

            embeddings = model.encode(text, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
            out = {}

            if NUM_BATCHES > 1:
                for x in range(0, NUM_BATCHES):
                    start = x * cluster_batch_size
                    end = (x + 1) * cluster_batch_size
                    clusters = util.community_detection(embeddings[start:end], min_community_size=1, threshold=threshold)
                    for i, cluster in enumerate(clusters):
                        idx = random.getrandbits(32)
                        for sentence_id in cluster:
                            id = start + sentence_id
                            if idx in out:
                                out[idx].append(id)
                            else:
                                out[idx] = [id]
                cluster_embeddings = []
                hashes = []
                for idx, cluster in out.items():
                    cluster_embeddings.append(embeddings[cluster[0]])
                    hashes.append(idx)

                batch_clusters = util.community_detection(torch.stack(cluster_embeddings), min_community_size=1, threshold=threshold)
                map_clusters = {}
                for i, cluster in enumerate(batch_clusters):
                    for sentence_id in cluster:
                        if hashes[sentence_id] not in map_clusters:
                            map_clusters[hashes[sentence_id]] = i

                for key, val in out.copy().items():
                    for x, y in out.copy().items():
                        if x in map_clusters and key in map_clusters and key != x:
                            if map_clusters[key] == map_clusters[x]:
                                if key in out:
                                    out[key].extend(y)
                                    del out[x]
            else:
                clusters = util.community_detection(embeddings, min_community_size=1, threshold=threshold)
                for i, cluster in enumerate(clusters):
                    idx = random.getrandbits(32)
                    for sentence_id in cluster:
                        if idx in out:
                            out[idx].append(sentence_id)
                        else:
                            out[idx] = [sentence_id]
            data = []
            for idx, cluster in out.items():
                for id in cluster:
                    data.append([idx, index[id], text[id]])

            out_df = pd.DataFrame(data, columns = ['cluster', 'index', 'text'])
            return out_df
        except Exception as E:
            logging.warning("Returning empty dataframe...")
            logging.warning(E)
            data = []
            out_df = pd.DataFrame(data, columns = ['cluster', 'index', 'text'])
            return out_df