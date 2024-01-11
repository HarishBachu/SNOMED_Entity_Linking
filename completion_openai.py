import openai
from openai import OpenAI

MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0
# Temperature 0.2 is recommended for deterministic and focused output that is
# 'more likely to be correct and efficient'. Instead, 0 is used for consistency.

with open('openai.key', 'r') as file:
    api_key = file.read().rstrip()

def create_chat_completion(prompts, model=MODEL, temperature=TEMPERATURE, **kwargs):
    client = OpenAI(api_key = api_key)
    response = client.chat.completions.create(messages=[
      {
        "role" : "user", 
        "content" : "Hi There"
      }
    ], model=model, temperature=temperature, **kwargs)
    return response["choices"][0]["message"]["content"]

# client = OpenAI(
#     # This is the default and can be omitted
#     api_key=os.environ.get("OPENAI_API_KEY"),
# )

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Say this is a test",
#         }
#     ],
#     model="gpt-3.5-turbo",
# )

