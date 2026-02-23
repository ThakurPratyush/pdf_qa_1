from google import genai

client = genai.Client(api_key="AIzaSyD2jgDDsZHeQGvavXz5T6O5kUjrFglmmOg")

for m in client.models.list():
    print(m.name)