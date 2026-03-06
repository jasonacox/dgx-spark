from openai import OpenAI

API_KEY = "not-needed"
BASE_URL = "http://spark.local:8000/v1"
MODEL_NAME = None  # Use default model

# Connect to your local nanochat service
print(f"Connecting to nanochat at {BASE_URL}...")
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# Determine available models
print("Available models:")
models = client.models.list()
for model in models.data:
    print(f" - {model.id}")
if MODEL_NAME is None:
    MODEL_NAME = models.data[0].id
print(f"\nUsing model: {MODEL_NAME}")

# Streaming response
prompt = "What is the capital of France?"
print("\nStreaming response:")
print("Prompt:", prompt, "\nResponse: ", end="", flush=True)
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": prompt}
    ],
    stream=True
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# Non-streaming response
prompt = "Hello!"
print("\n\nNon-streaming response:")
print("Prompt:", prompt, "\nResponse: ", end="", flush=True)
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": prompt}
    ],
    stream=False
)
print(response.choices[0].message.content)
print("\n✓ Test completed successfully.")

