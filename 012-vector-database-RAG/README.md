# Vector Database
Database using vector embedding as key

## Understanding Vector Database
1. Vector Library
    - Faiss, Annoy, NMSLIB, ScaNN
2. Vector-only Database
    - Pinecone, Weaviate, Milvus, Chroma, Qdrant, Vespa
3. Vector Database - Feature Addition
    - ElasticSearch, PostgreSQL, MongoDB, Neo4j

</br></br>

# How a Vector Database Works

## KNN retreive and Limitations
KNN is intuitive and accurate because it examines all data

**Practice Dataset**
```bash
!wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
!tar -xf sift.tar.gz
!mkdir data/sift1M -p
!mv sift/* data/sift1M
```

**Load Practice Dataset**
```py
import time
import psutil

import faiss
from faiss.contrib.datasets import DatasetSIFT1M

def get_memory_usage_mb():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)

ds = DatasetSIFT1M()

xq = ds.get_queries() # data for using
xb = ds.get_database() # saved vector data
gt = ds.get_groundtruth() # ground truth label
```

**Changes in index/search time and memory usages as data grows**
```py
k = 1
d = xq.shape[1]
nq = 1000
xq = xq[:nq]

for i in range(1, 10, 2):
    start_memory = get_memory_usage_mb()
    start_indexing = time.time()
    index = faiss.IndexFlatL2(d)
    index.add(xb[:(i+1)*100000])
    end_indexing = time.time()
    end_memory = get_memory_usage_mb()

    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()
    print(f"데이터: {(i+1)*100000}개")
    print(f"색인: {(end_indexing - start_indexing) * 10000 :.3f} ms {(end_memory - start_memory):.3f} MB 검색: {(t1 - t0) * 1000 / nq :.3f} ms")
```
```
데이터: 200000개
색인: 562.561 ms (98.000 MB 검색: 1.649 ms)
데이터: 400000개
색인: 958.750 ms (97.465 MB 검색: 3.270 ms)
데이터: 600000개
색인: 1465.695 ms (97.621 MB 검색: 4.880 ms)
데이터: 800000개
색인: 1952.639 ms (97.566 MB 검색: 6.512 ms)
데이터: 1000000개
색인: 2458.615 ms (97.633 MB 검색: 8.160 ms)
```
But, since it examines all vectors, the amount of computation increases proportionally to the number of data, making it slow, so it is not scalable

## ANN retrieve
**Approximate Nearest Neighbor**
Focus on narrowing the retrieve range when retrieving by storing the embedding vector in a structure that can be retrieved quickly
- Inverted File Index(IVF), HNSW(Hierachical Navigable Small World)

## Navigable Small World(NSW)
In order to create random connections that reduce the number of search steps, NSW use a method to store vectors in a randomly shuffled order in the searchable small world.
- A local minimum problem may occur

## Hierarchy
Stores vectors by arranging graphs in multiple layers using a level-by-level hierarchical structure.

</br></br>

# Key parameters of HNSW Index

## Paramter: `m`
Minimum number of conntections to conntect to a vector
- The more connected edges there are, the denser the graph becomes, which improves the quality (recall) of the search.
- However, it increases memory usage and increases indexing/retrieve time.
```py
import numpy as np

k = 1
d = xq.shape[1]
nq = 1000
xq = xq[:nq]

for m in [8, 16, 32, 64]:
    index = faiss.IndexHNSWFlat(d, m)
    time.sleep(3)
    start_memory = get_memory_usage_mb()
    start_index = time.time()
    index.add(xb)
    end_memory = get_memory_usage_mb()
    end_index = time.time()
    print(f"M: {m} - Indexing time: {end_index - start_index} s, Memory usage: {end_memory - start_memory} MB")

    to = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()

    recall_at_1 = np.equal(I, gt[:nq, :1]).sum() / float(nq)
    print(f"{(t1 - t0) * 1000.0 / nq:.3f} ms per query, R@1 {recall_at_1:.3f}")
```
```
M: 8 - Indexing time: 10.826627492904663 s, Memory usage: 598.71875 MB
2121.793 ms per query, R@1 0.684
M: 16 - Indexing time: 13.72940731048584 s, Memory usage: 621.51171875 MB
2138.547 ms per query, R@1 0.774
M: 32 - Indexing time: 26.921372652053833 s, Memory usage: 736.640625 MB
2168.507 ms per query, R@1 0.891
M: 64 - Indexing time: 37.020442485809326 s, Memory usage: 1011.65234375 MB
2208.576 ms per query, R@1 0.928
```

## Parameter: `ef_construction`
The number of candidates to store to select the M closest ones during the indexing process.
```py
k = 1
d = xq.shape[1]
nq = 1000
xq = xq[:nq]

for ef_construction in [40, 80, 160, 320]:
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = ef_construction
    start_memory = get_memory_usage_mb()
    start_index = time.time()
    index.add(xb)
    end_memory = get_memory_usage_mb()
    end_index = time.time()
    print(f"efConstruction: {ef_construction} - Indexing time: {end_index - start_index} s, Memory usage: {end_memory - start_memory} MB")

    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()

    recall_at_1 = np.equal(I, gt[:nq, :1]).sum() / float(nq)
    print(f"{(t1 - t0) * 1000.0 / nq:.3f} ms per query, R@1 {recall_at_1:.3f}")
```
```
efConstruction: 40 - Indexing time: 26.81549048423767 s, Memory usage: 748.63671875 MB
0.019 ms per query, R@1 0.909
efConstruction: 80 - Indexing time: 33.101176261901855 s, Memory usage: 736.4296875 MB
0.014 ms per query, R@1 0.873
efConstruction: 160 - Indexing time: 68.20233583450317 s, Memory usage: 736.23828125 MB
0.016 ms per query, R@1 0.883
efConstruction: 320 - Indexing time: 134.2638123035431 s, Memory usage: 736.45703125 MB
0.018 ms per query, R@1 0.903
```

## Parameter: `ef_search`
The number of candidates to store to select the K closest ones during the retrieve process
```py
for ef_search in [16, 32, 64, 128]:
    index.hnsw.efSearch = ef_search
    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()

    recall_at_1 = np.equal(I, gt[:nq, :1]).sum() / float(nq)
    print(f"{(t1 - t0) * 1000.0 / nq:.3f} ms per query, R@1 {recall_at_1:.3f}")
```
```
0.018 ms per query, R@1 0.903
0.028 ms per query, R@1 0.963
0.049 ms per query, R@1 0.988
0.086 ms per query, R@1 0.994
```

</br></br>

# Pinecone

## Pinecone Client
```py
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
# Connect account & Create index
pinecone_api_key = ""

pc = Pinecone(api_key=pinecone_api_key)
pc.create_index("llm-book", spec=ServerlessSpec("aws", "us-east-1"), dimension=768)

index = pc.Index('llm-book') # Load Index

# Create embeddings
sentence_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
klue_dp_train = load_dataset('klue', 'dp', split='train[:100]')

embeddings = sentence_model.encode(klue_dp_train['sentence'])
embeddings = embeddings.tolist()

# Data processing
# Data format for pinecone: {"id": document ID(str), "values": embeddings(List[float]), "metadata": metadata(dict)}
insert_data = []
for idx, (embedding, text) in enumerate(zip(embeddings, klue_dp_train['sentence'])):
    insert_data.append({"id": str(idx), "values": embedding, "metadata": {'text': text}})

# Save embedding to index
upsert_response = index.upsert(vectors=insert_data, namespace='llm-book-sub')

# Retrieve index
query_response = index.query(
    namespace='llm-book-sub',
    top_k=10,
    include_values=True,
    include_metadata=True,
    vector=embeddings[0]
)

query_response

# document update & delete
new_text = 'new text for updating'
new_embedding = sentence_model.encode(new_text).tolist()

# update
update_response = index.update(
    id='existing_document_id',
    valeus=new_embedding,
    set_metadata={'text': new_text},
    namespace='llm-book-sub'
)

# delete
delete_response = index.delete(ids=['existing_document_id'], namespace='llm-book-sub')
```

## Llama Index
**Use different vector database in LlamaIndex**
```py
# Pinecone setting
from pinecone import Pinecone

pc = Pinecone(api_key=pinecone_api_key)
pc.create_index(
    "quickstart", dimension=1536, metric="euclidean", spec=ServerlessSpec("aws", "us-east-1"))
pinecone_index = pc.Index("quickstart")

# Connect pinecone index to LlamaIndex
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
```

</br></br>

# Implementation Multi-modal Retrieve using Pinecone

## Dataset
```py
from datasets import load_dataset

dataset = load_dataset("poloclub/diffusiondb", "2m_first_1k", split='train')

example_index = 867
original_image = dataset[example_index]['image']
original_prompt = dataset[example_index]['prompt']
print(original_prompt)
```

## Image Explanation using GPT-4o¶
```py
import base64
import requests
from io import BytesIO

def make_base64(image):
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def generate_description_from_image_gpt4(prompt, image64):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {client.api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    response_oai = requests.post("https://api.openai.com/v1/chat/completions",
                                 headers=headers, json=payload)
    result = response_oai.json()['choices'][0]['message']['content']
    return result

from openai import OpenAI
client = OpenAI()

# Image explanation
image_base64 = make_base64(original_image)
described_result = generate_description_from_image_gpt4("Describe provided image", image_base64)

described_result
```
```
'The image depicts a majestic lion with an elaborate and colorful mane that appears to be adorned with vibrant, peacock-like feathers. The lion is centered in a lush, natural setting with a soft-focus background featuring greenery and purple flowers. The artwork combines elements of realism with fantastical features, creating a visually striking and imaginative scene.'
```

## Save prompt
```py
index_name = "llm-multimodal"
try:
    pc.create_index(
        name=index_name,
        dimension=512,
        metric='cosine',
        spec=ServerlessSpec('aws', 'us-east-1')
    )
    print(f"Index '{index_name}' created successfully.")
except Exception as e:
    print(f"Failed to create index: {e}")
    
index = pc.Index(index_name)
```

**prompt text to embedding vector**
```py
import torch

from tqdm.auto import trange
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPTextModelWithProjection

device = "cuda" if torch.cuda.is_available() else "cpu"

text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

tokens = tokenizer(dataset['prompt'], padding=True, return_tensors='pt', truncation=True)
batch_size = 16
text_embs = []
for start_idx in trange(0, len(dataset), batch_size):
    with torch.no_grad():
        outputs = text_model(
            input_ids = tokens['input_ids'][start_idx: start_idx+batch_size],
            attention_mask = tokens['attention_mask'][start_idx: start_idx+batch_size])
        text_emb_tmp = outputs.text_embeds
    text_embs.append(text_emb_tmp)
text_embs = torch.cat(text_embs, dim=0)
text_embs.shape # torch.Size([1000, 512])
``` 

**Text embedding to pinecone index**
```py
input_data = []

for id_int, emb, prompt in zip(range(0, len(dataset)), text_embs.tolist(), dataset['prompt']):
    input_data.append(
        {
            "id": str(id_int),
            "values": emb,
            "metadata": {
                "prompt": prompt
            }
        }
    )

index.upsert(
    vectors=input_data
)
```

## Retrieve image embeddings
**similar prompt retrieve using image embedding**
```py
from transformers import AutoProcessor, CLIPVisionModelWithProjection

vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

inputs = processor(images=original_image, return_tensors="pt")

outputs = vision_model(**inputs)
image_embeds = outputs.image_embeds

search_results = index.query(
    vector=image_embeds[0].tolist(),
    top_k=3,
    include_values=False,
    include_metadata=True
)

search_idx = int(search_results['matches'][0]['id'])
```
```
search_results
>>>
{'matches': [{'id': '918',
              'metadata': {'prompt': 'cute fluffy bunny cat lion hybrid mixed '
                                     'creature character concept, with long '
                                     'flowing mane blowing in the wind, long '
                                     'peacock feather tail, wearing headdress '
                                     'of tribal peacock feathers and flowers, '
                                     'detailed painting, renaissance, 4 k '},
              'score': 0.37472102,
              'values': []},
             {'id': '817',
              'metadata': {'prompt': 'cute fluffy baby cat lion hybrid mixed '
                                     'creature character concept, with long '
                                     'flowing mane blowing in the wind, long '
                                     'peacock feather tail, wearing headdress '
                                     'of tribal peacock feathers and flowers, '
                                     'detailed painting, renaissance, 4 k '},
              'score': 0.372388244,
              'values': []},
             {'id': '867',
              'metadata': {'prompt': 'cute fluffy baby cat rabbit lion hybrid '
                                     'mixed creature character concept, with '
                                     'long flowing mane blowing in the wind, '
                                     'long peacock feather tail, wearing '
                                     'headdress of tribal peacock feathers and '
                                     'flowers, detailed painting, renaissance, '
                                     '4 k '},
              'score': 0.372347265,
              'values': []}],
 'namespace': '',
 'usage': {'read_units': 6}}
```

## Generate image using DALL-E 3
**Func. Genrate and save image using prompt**
```py
from PIL import Image

def generate_image_dalle3(prompt):
    response_oai = client.images.generate(
        model="dall-e-3",
        prompt=str(prompt),
        size="1024x1024",
        quality="standard",
        n=1,
    )
    result = response_oai.data[0].url
    return result

def get_generated_image(image_url):
    generated_image = requests.get(image_url).content
    image_filename = 'gen_img.png'
    with open(image_filename, 'wb') as image_file:
        image_file.write(generated_image)
    return Image.open(image_filename)

# GPT-4o prompt
gpt_described_image_url = generate_image_dalle3(described_result)
gpt4o_prompt_image = get_generated_image(gpt_described_image_url)

# Original prompt
original_prompt_image_url = generate_image_dalle3(original_prompt)
original_prompt_image = get_generated_image(original_prompt_image_url)

# Retrieved prompt
searched_prompt_image_url = generate_image_dalle3(dataset[search_idx]['prompt'])
searched_prompt_image = get_generated_image(searched_prompt_image_url)
```