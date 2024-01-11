from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from prompts import convert_chat_to_message

MODEL = "claude-2.1"
MAX_TOKENS = 300

with open('anthropic.key', 'r') as file:
  api_key = file.read().rstrip()

def create_chat_completion(prompts, model=MODEL, max_tokens=MAX_TOKENS):
  # print(prompts)
  client = Anthropic(api_key=api_key)
  response = client.completions.create(
    model=MODEL,
    max_tokens_to_sample=MAX_TOKENS,
    prompt=f"{HUMAN_PROMPT} {convert_chat_to_message(prompts)}{AI_PROMPT}",
  )
  return response.completion