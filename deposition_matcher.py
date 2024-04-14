import math
import random
from pinecone import Pinecone
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.readers.file import PDFReader
from pathlib import Path
from llama_index.core import Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import NodeParser
from whisper_func import VideoToTextModel
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader

class DepositionMatcher:
    def __init__(self, pinecone_index_name, pinecone_api_key, togetherai_api_key):
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index_name)
        self.embedder = TogetherEmbedding(
            model_name="togethercomputer/m2-bert-80M-8k-retrieval", api_key=togetherai_api_key
        )
        self.vector_store = PineconeVectorStore(pinecone_index=self.index)

    def query(self, video_path, max_doc=5, max_window=3):
        segs = self.query_text("pranav", top_k=100, filter={"video_path": {"$eq": video_path}}, include_values=True)
        segs = sorted(segs["matches"], key=lambda x:x["metadata"]["start"])
        
        matches = []
        for seg in segs:
            res = self.index.query(vector=seg["values"], top_k=max_doc, include_values=True, include_metadata=True, filter={"file_type": {"$eq": "application/pdf"}})
            max_match = max(res["matches"], key=lambda x:x["score"])
            matches.append({
                "start": float(seg["metadata"]["start"]),
                "end": float(seg["metadata"]["end"]),
                "utterance": seg["metadata"]["text"],
                "file": max_match["metadata"]["file_name"],
                "page": int(max_match["metadata"]["page_label"]),
                "score": float(max_match["score"])
            })
        
        return sorted(matches, key=lambda x:x["score"])[::-1][:max_window]


    def vectorize_text(self, text):
        return self.embedder.get_text_embedding(text)
    
    def query_text(self, text, top_k=3, include_values=False, filter=None):
        query_vector = self.vectorize_text(text)
        return self.index.query(vector=query_vector, top_k=top_k, include_values=include_values, include_metadata=True, filter=filter)

