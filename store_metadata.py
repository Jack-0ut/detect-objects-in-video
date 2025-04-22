import os
import pickle
import numpy as np
import hashlib
from collections import Counter
from sentence_transformers import SentenceTransformer

class MetadataStore:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", save_path="metadata.pkl"):
        self.model = SentenceTransformer(embedding_model)
        self.save_path = save_path

        self.metadata = []
        self.hash_to_idx = {}
        self.embeddings = []  # np.array of shape (N, 384)

        self._load()

    def _object_signature(self, objects):
        counts = Counter(obj["class"] for obj in objects)
        return "|".join(f"{k}:{counts[k]}" for k in sorted(counts))

    def _hash_signature(self, sig):
        return hashlib.md5(sig.encode()).hexdigest()

    def _create_description(self, objects):
        counts = Counter(obj["class"] for obj in objects)
        return ", ".join(f"{counts[k]} {k}{'s' if counts[k] > 1 else ''}" for k in sorted(counts))

    def add_frame(self, frame_number: int, timestamp: str, inference_time: float, objects: list):
        sig = self._object_signature(objects)
        h = self._hash_signature(sig)

        if h in self.hash_to_idx:
            idx = self.hash_to_idx[h]
            self.metadata[idx]["frames"].append(frame_number)
            return False

        desc = self._create_description(objects)
        emb = self.model.encode(desc)
        self.embeddings.append(emb)

        self.metadata.append({
            "description": desc,
            "signature": sig,
            "frames": [frame_number],
            "timestamp": timestamp,
            "inference_time_ms": inference_time,
            "objects": objects,
        })
        self.hash_to_idx[h] = len(self.metadata) - 1
        return True

    def search(self, query: str, k=5):
        if not self.embeddings:
            return []

        query_emb = self.model.encode(query)
        emb_matrix = np.array(self.embeddings)
        distances = np.linalg.norm(emb_matrix - query_emb, axis=1)
        top_k_indices = distances.argsort()[:k]
        return [self.metadata[i] for i in top_k_indices]

    def save(self):
        with open(self.save_path, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "hash_to_idx": self.hash_to_idx,
                "embeddings": np.array(self.embeddings),
            }, f)

    def _load(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, "rb") as f:
                data = pickle.load(f)
                self.metadata = data["metadata"]
                self.hash_to_idx = data["hash_to_idx"]
                self.embeddings = data["embeddings"].tolist()  # back to list of vectors
