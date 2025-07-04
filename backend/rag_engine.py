import numpy as np
import pandas as pd
from chromadb import PersistentClient
import chromadb.utils.embedding_functions as embedding_functions
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.data_loader import load_all_data
from preprocessing.feature_engineering import prepare_features

data = load_all_data()
train_data = data["train"]
train_data = prepare_features(train_data, data["oil"], data["holidays"], data["transactions"], data["stores"])
# train_data.dropna(inplace=True)

train_data['text'] = (
    "Store " + train_data['store_nbr'].astype(str) +
    ", Family: " + train_data['family'].astype(str) +
    ", Sales: " + train_data['sales'].astype(str)
)
corpus = train_data['text'].tolist()

print("corpus:" , len(corpus))
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_path = os.path.join(os.path.dirname(__file__), "..", "chromadb")

chroma_client = PersistentClient(path=chroma_path)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="sales_data",
    embedding_function=sentence_transformer_ef
)


if collection.count() == 0:
    print("Embedding and populating ChromaDB...")
    BATCH_SIZE = 5000

    for i in tqdm(range(0, len(corpus), BATCH_SIZE), desc="Encoding & Inserting"):
        batch_corpus = corpus[i:i + BATCH_SIZE]
        batch_meta = train_data[['store_nbr', 'family']].iloc[i:i + BATCH_SIZE].to_dict(orient='records')
        batch_ids = [f"doc_{j}" for j in range(i, i + len(batch_corpus))]

        # Embedding
        batch_embeddings = embedding_model.encode(
            batch_corpus,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()

        # Add to Chroma
        collection.add(
            documents=batch_corpus,
            embeddings=batch_embeddings,
            ids=batch_ids,
            metadatas=batch_meta
        )

    print(f"Added {collection.count()} documents to ChromaDB.")
else:
    print(f"ChromaDB already populated with {collection.count()} documents.")


__all__ = ["train_data", "embedding_model", "collection", "corpus"]
