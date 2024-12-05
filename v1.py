import ollama
model_name = 'llama3:latest'  # Correct model name
response = ollama.chat(model=model_name, messages=[{"role": "user", "content": "Hello, how are you?"}])
print(response)
