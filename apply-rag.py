#!.venv/bin/python3
import pandas as pd
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import requests as req
import os
from openai import OpenAI

os.environ["TOKENIZERS_PARALLELISM"] = "false"

df = pd.read_csv('./test_data.csv')
data = df.to_dict('records')

encoder = SentenceTransformer('all-MiniLM-L6-v2') # Model to create embeddings

# create the vector database client
qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance

# Create collection to store peoples
collection = "people"
qdrant.recreate_collection(
    collection_name=collection,
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
        distance=models.Distance.COSINE
    )
)

# vectorize!
qdrant.upload_points(
    collection_name=collection,
    points=[
        models.PointStruct(
            id=idx,
            vector=encoder.encode(doc["description"]).tolist(),
            payload=doc,
        ) for idx, doc in enumerate(data) # data is the variable holding all the peoples
    ]
)

user_prompt = "Please tell me the name of people that has description similar with 'only contain magazine baby'?"

hits = qdrant.search(
    collection_name=collection,
    query_vector=encoder.encode(user_prompt).tolist(),
    limit=3
)
# for hit in hits:
#   print(hit.payload, "score:", hit.score)

# define a variable to hold the search results
search_results = [hit.payload for hit in hits]

client = OpenAI(
    base_url="http://localhost:8080/v1", # "http://<Your api-server IP>:port"
    api_key = "sk-no-key-required"
)

completion = client.chat.completions.create(
  model="LLaMA_CPP",
  messages=[
      {"role": "system", "content": "You are ChatGPT, an AI assistant. Your top priority is help me find the name of a person by giving her description."},
      {"role": "user", "content": user_prompt},
      {"role": "assistant", "content": str(search_results)}
  ]
)

print(completion.choices[0].message.content)

