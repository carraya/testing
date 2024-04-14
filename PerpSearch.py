import instructor
from pydantic import BaseModel
from anthropic import Anthropic
from dotenv import load_dotenv
import os
from serpapi import GoogleSearch
from tavily import TavilyClient

load_dotenv()


class SearchQuery(BaseModel):
    text: str


class Content(BaseModel):
    url: str
    content: str


class PerpSearch:

    def __init__(self, keywords=None):
        self.keywords = keywords
        self.client = instructor.from_anthropic(Anthropic())
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def getSourcesFromSegments(self, segments, start_time, end_time):
        full_text = []
        for segment in segments:
            if segment["start"] >= start_time and segment["end"] <= end_time:
                full_text.append(segment["text"])
        if not full_text:
            return None
        return self.getSources(" ".join(full_text))

    def getSources(self, full_text):
        resp = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": "Given a text, return a concise Google search query.",
                },
                {"role": "user", "content": full_text},
            ],
            response_model=SearchQuery,
            temperature=0.2,
        )

        assert isinstance(resp, SearchQuery)

        query = resp.text

        response = self.tavily.search(query=query)
        context = [
            {"url": obj["url"], "content": obj["content"]}
            for obj in response["results"]
        ]

        # for i in range(len(context)):
        #     context[i] = self.client.messages.create(
        #         # model="claude-3-sonnet-20240229",
        #         model="claude-3-haiku-20240307",
        #         max_tokens=1024,
        #         messages=[
        #             {
        #                 "role": "system",
        #                 "content": "Given a dictionary with keys 'url' and 'content', remove unrelated text from the content and return the cleaned dictionary.",
        #             },
        #             {"role": "user", "content": str(context[i])},
        #         ],
        #         response_model=Content,
        #         temperature=0.2,
        #     )
        #     context[i] = {"url": context[i].url, "content": context[i].content}

        response = {"query": query, "context": context}
        return response

        # # Use SerpApi to perform a Google search based on the query
        # params = {
        #     "q": query,
        #     "location": "United States",
        #     "hl": "en",
        #     "gl": "us",
        #     "google_domain": "google.com",
        #     "api_key": os.getenv("SERPAPI_KEY"),  # Replace with your actual SerpApi key
        # }

        # search = GoogleSearch(params)
        # results = search.get_dict()

        # # Return the most relevant results
        # return results["organic_results"]


# Testing
if __name__ == "__main__":
    perp = PerpSearch()
    sources = perp.getSources(
        "So it's not possible once the GPT gets like GPT-7, we'll just be instantaneously be able to see, you know, here's the proof from our theorem. It seems to me like you want to be able to allocate more compute to harder problems."
    )
    print(sources)
