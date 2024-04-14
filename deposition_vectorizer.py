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

class DepositionVectorizer:
    def __init__(self, pinecone_index_name, pinecone_api_key, togetherai_api_key):
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index_name)
        self.embedder = TogetherEmbedding(
            model_name="togethercomputer/m2-bert-80M-8k-retrieval", api_key=togetherai_api_key
        )
        self.vector_store = PineconeVectorStore(pinecone_index=self.index)
        
    def vectorize_and_upsert_pdf(self, pdf_path):
        res = self.query_text("a", top_k=1, filter={"file_path": {"$eq": pdf_path}})
        if (len(res["matches"]) > 0):
          return
        documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
        node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20, include_metadata=True)
        nodes = node_parser.get_nodes_from_documents(documents)
        vectors = [{"id": node.node_id, "values": self.vectorize_text(node.get_content()), "metadata": {**node.metadata, "text": node.get_content()}} for node in nodes]
        self.index.upsert(vectors=vectors)
        
    def vectorize_and_upsert_video(self, video_path, window=5, overlap=1):
        print("vectorize + upsert")

        res = self.query_text("a", top_k=1, filter={"video_path": {"$eq": video_path}})
        if (len(res["matches"]) > 0):
          print("dup ski")
          return

        video_to_text = VideoToTextModel()
        text = video_to_text.sentence_transcribe(video_path)
        vectors = []
        prev_seg_ind_start = -1

        max_duration = int(math.ceil(text[-1]["end"]))
        for t0 in range(0, max_duration, window-overlap):
            print(f"t: {t0}")

            seg_ind_start = -1
            for j, s in enumerate(text):
                if s['start'] >= t0:
                    seg_ind_start = j
                    break
            
            if seg_ind_start <= max(-1, prev_seg_ind_start):
                print("skip", seg_ind_start)
                continue
            
            prev_seg_ind_start = seg_ind_start
            seg_ind_end = seg_ind_start
            while (text[seg_ind_end]['end']-text[seg_ind_start]['start']) < window and seg_ind_end < len(text)-1:
                seg_ind_end += 1

            seg_text = "\n".join([seg["text"] for seg in text[seg_ind_start:seg_ind_end+1]])
            
            print("VEC: ", seg_ind_start, seg_ind_end, seg_text)

            vector = {
                "id": str(random.random()).replace("0.", ""), 
                "values": self.vectorize_text(seg_text),
                "metadata": {
                    "start": text[seg_ind_start]["start"],
                    "end": text[seg_ind_end]["end"],
                    "video_path": video_path,
                    "text": seg_text 
                },
            }
            vectors.append(vector)

        if len(vectors) > 0:
            self.index.upsert(vectors=vectors)
        else:
            print("no vectors")
      
    def vectorize_text(self, text):
        return self.embedder.get_text_embedding(text)
    
    def query_text(self, text, top_k=3, filter=None):
        query_vector = self.vectorize_text(text)
        return self.index.query(vector=query_vector, top_k=top_k, include_values=False, include_metadata=True, filter=filter)

