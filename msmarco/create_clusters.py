from collections import defaultdict
import pickle
import pandas as pd
import faiss
import numpy as np
from tqdm import tqdm
import os


CLUSTERS_NUM = 3
N_ITER = 20
EMBEDDINGS_LOC = "./msmarco_clusters_test"
passages_file = "./passages_embeddings_baai1.5small.tsv"
#passages_file = "./queries_eval_embeddings_baai1.5small.tsv"
query_file = "./queries_eval_embeddings_baai1.5small.tsv"

def read(file):
    # Read the entire TSV file into a DataFrame
    #df = pd.read_csv(file, sep='\t', header=None, encoding='utf-8')
    
    df = pd.concat([chunk for chunk in tqdm(pd.read_csv(file, sep='\t', header=None, encoding='utf-8', chunksize = 1000), desc='Loading data')])

    # Extract the columns
    ids = df[0].tolist()
    passages = df[1].tolist()
    
    # Use vectorized string operations to split each embedding string into columns,
    # then convert them to float32 (or float64 if needed)
    chunksize = 10000

    # Split the Series into roughly equal-sized chunks
    chunks = np.array_split(df[2], len(df[2]) // chunksize + 1)
    embeddings_list = []

    # Process each chunk with a progress bar
    for chunk in tqdm(chunks, desc="Processing embeddings"):
        # For each chunk, split the strings and convert to float32
        chunk_embeddings = chunk.str.split(" ", expand=True).astype(np.float32).values
        embeddings_list.append(chunk_embeddings)

    # Combine all chunks back into a single NumPy array
    embeddings = np.vstack(embeddings_list)
    embeddings = np.ascontiguousarray(embeddings)
    return ids, passages, embeddings

ids, passages, embeddings = read(passages_file)

query_ids, queries, query_embeddings = read(query_file)

print(embeddings.shape)

#exit()

num_gpus = faiss.get_num_gpus()
print("Number of GPUs available:", num_gpus)

niter = N_ITER
verbose = True
d = embeddings.shape[1]
kmeans = faiss.Kmeans(d, CLUSTERS_NUM, niter=niter, verbose=verbose)
kmeans.train(embeddings)

print(kmeans.centroids.shape)

D, I = kmeans.index.search(embeddings, 1)

print(D.shape)
print(I.shape)


index = faiss.IndexFlatL2(d)

# Add embeddings to the index
index.add(embeddings)

# Search for the nearest 5 neighbors
k = 5  # number of nearest neighbors
distances, indices = index.search(query_embeddings[0].reshape(1, -1), k)

print("Indices of nearest neighbors:", indices)
print("Distances to nearest neighbors:", distances)

doc_emb_map = defaultdict(dict)
clustered_embs = [[] for _ in range(CLUSTERS_NUM)]
embs = embeddings
for i in range(len(embs)):
    cluster = I[i][0]
    if len(I[i]) != 1:
        print(f"Error in embedding {i}, len {len(I[i])}")
    clustered_embs[cluster].append(embs[i])
    emb_id = len(clustered_embs[cluster]) - 1
    doc_emb_map[cluster][emb_id] = i

querytexts = queries


centroids = kmeans.centroids


os.makedirs(EMBEDDINGS_LOC, exist_ok=True)

with open(f'{EMBEDDINGS_LOC}/centroids.pkl', 'wb') as file:
    pickle.dump(centroids, file)

with open(f'{EMBEDDINGS_LOC}/embeddings_list.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

for i in range(CLUSTERS_NUM):
    with open(f'{EMBEDDINGS_LOC}/cluster_{i}.pkl', 'wb') as f:
        pickle.dump(clustered_embs[i], f)

with open(f'{EMBEDDINGS_LOC}/doc_emb_map.pkl', 'wb') as f:
    pickle.dump(doc_emb_map, f)

with open(f'{EMBEDDINGS_LOC}/doc_list.pkl', 'wb') as f:
    pickle.dump(passages, f)

np.savetxt(f'{EMBEDDINGS_LOC}/query.csv', querytexts, fmt="%s")
np.savetxt(f'{EMBEDDINGS_LOC}/query_emb.csv', query_embeddings, delimiter=",")
