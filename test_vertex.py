from google import genai

client = genai.Client(
    vertexai=True,
    project="pdf-rag-project-488220",
    location="us-central1"
)

response = client.models.embed_content(
    model="text-embedding-004",
    contents=["hello world"]
)

print("SUCCESS")