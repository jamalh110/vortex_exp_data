import os
from FlagEmbedding import FlagModel
import pandas as pd

passages_file = './collection.tsv'
queries_file = './queries.eval.tsv'

model = FlagModel(
        'BAAI/bge-small-en-v1.5',
        devices="cuda:0",   # if you don't have a GPU, you can use "cpu"
    )

def compute_embeddings(file, output, batch_size):

    count = 0
    ids_comp = []
    passages_comp = []
    embeddings_comp = []
    # Read the TSV file in chunks (each chunk is a DataFrame)
    for chunk in pd.read_csv(file, sep='\t', header=None, chunksize=batch_size, encoding='utf-8'):
        # Process each chunk (for example, display the chunk)
        # convert chunk to list of strings
        ids = chunk[0].tolist()
        passages = chunk[1].tolist()

        ids_comp.extend(ids)
        passages_comp.extend(passages)

        #print(ids)
        #print(passages)
        # get the embeddings
        embeddings = model.encode(passages)
        embeddings_comp.extend(embeddings)
        if count%100 == 0:
            print(count*batch_size)
        count+=1
    
    # Save the embeddings to a file
    with open(output, 'w') as f:
        batch_lines = []
        count = 0
        for i in range(len(ids_comp)):
            line = str(ids_comp[i]) + '\t' + passages_comp[i] + '\t' + ' '.join([str(x) for x in embeddings_comp[i]]) + '\n'
            batch_lines.append(line)
            count += 1
            if count % 1000 == 0:
                f.write(''.join(batch_lines))
                batch_lines = []
                print(count)
        if batch_lines:
            f.write(''.join(batch_lines))

    
compute_embeddings(queries_file, '/mydata/msmarco/queries_eval_embeddings.tsv', 64)
compute_embeddings(passages_file, '/mydata/msmarco/passages.tsv', 64)
