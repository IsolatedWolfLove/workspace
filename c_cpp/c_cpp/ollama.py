import ollama

response = ollama.chat(
    model='llama3.2-vision',
    messages=[{
        'role': 'user',
        'content': 'What is in this image?',
        'images': ['/home/ccy/workspace/deeplearning/learning_conda/src/load_model_cpp/loss/0.png']
    }]
)

print(response)