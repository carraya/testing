import openai
import instructor
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

client = instructor.from_openai(openai.OpenAI())


class User(BaseModel):
    name: str
    biography: str
    age: int


user_stream = client.chat.completions.create_partial(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Create a user with a biography and age"},
    ],
    response_model=User,
)

for user in user_stream:
    print(user)
