import json
from deposition_vectorizer import DepositionVectorizer
from deposition_matcher import DepositionMatcher
from contradictions import ContradictionFinder
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

def main():
    deposition_vectorizer = DepositionVectorizer(
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        togetherai_api_key=os.getenv("TOGETHERAI_API_KEY"),
    )

    deposition_matcher = DepositionMatcher(
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        togetherai_api_key=os.getenv("TOGETHERAI_API_KEY"),
    )

    # deposition_vectorizer.vectorize_and_upsert_pdf("document.pdf")
    # deposition_vectorizer.vectorize_and_upsert_video("deposition.mp4", window=7, overlap=3)
    
    res = deposition_matcher.query("deposition.mp4", max_doc=5, max_window=10)
    print(json.dumps(res, indent=4))
    # contradiction_finder = ContradictionFinder(deposition_vectorizer, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
    # await contradiction_finder.find_contradictions("fridman_altman_podcast_sample.mp4")
    
    # deposition_vectorizer.index.delete(delete_all=True)

    # await deposition_vectorizer.vectorize_and_upsert_pdf("Zipcar_ Refining the Business Model.pdf")
    # res = deposition_vectorizer.query("pranav.3gp", max_doc=5, max_window=5)
    # print(json.dumps(res, indent=4, sort_keys=True))
    # res = await deposition_vectorizer.query_text("How does GPT-7 relate to proofs?")
    # print(res)

if __name__ == "__main__":
    main()
