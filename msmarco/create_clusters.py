import pandas as pd


def cluster(file, output, ncentroids):
    count = 0
    ids_comp = []
    passages_comp = []
    embeddings_comp = []
    # Read the TSV file in chunks (each chunk is a DataFrame)
    for chunk in pd.read_csv(file, sep='\t', header=None, chunksize=128, encoding='utf-8'):
        # Process each chunk (for example, display the chunk)
        # convert chunk to list of strings
        ids = chunk[0].tolist()
        passages = chunk[1].tolist()
        embeddings = chunk[2].tolist()

        for i in range(len(embeddings)):
            embeddings[i] = [float(x) for x in embeddings[i].split(" ")]

        #print(ids[0])
        #print(passages[0])
        #print(embeddings[0])

        ids_comp.extend(ids)
        passages_comp.extend(passages)
        embeddings_comp.extend(embeddings)

    print(len(embeddings_comp))
    print(len(embeddings_comp[7878]))
