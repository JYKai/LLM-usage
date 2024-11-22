# Understanding Text Embeddings

## Advantages of sentence embedding

**Computing word-to-word similarity using sentence embedding**
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

smodel = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
dense_embeddings = smodel.encode(['School', 'Study', 'Exercise'])

cosine_similarity(dense_embeddings)
```
```
array([[0.9999999 , 0.46143383, 0.5178716 ],
       [0.46143383, 1.        , 0.44198984],
       [0.5178716 , 0.44198984, 0.99999994]], dtype=float32)
```
- Using character embedding, we can determine whether different texts are similar or related to each other as if they were understood by humans.

## One-hot encoding

**Limitations of one-hot encoding**
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

word_dict = {
    "school": np.array([[1, 0, 0]]),
    "study": np.array([[0, 1, 0]]),
    "workout": np.array([[0, 0, 1]])
}

cosine_school_study = cosine_similarity(word_dict['school'], word_dict['study']) # array([[0.]])
cosine_school_workout = cosine_similarity(word_dict['school'], word_dict['workout']) # array([[0.]])
```
- It can prevent unintended relationships between categorical data
- It is not possible to find similarity between words

## Bag of words
Converts documents into numbers by using the assuption that 'if there are many similar words, it is a similar sentence or document'

Sometimes, the frequency with which a word appers dose not help much in understanding the meaning of a document

## TF-IDF
**Term Frequency-Inverse Document Frequency(TF-IDF)**  
To address the 'words that appear in every document' problem, the importance of words that appear in many documents is reduced

| TF-IDF(w) = TF(w) x log(N/DF(w))
- TF(w) : number of times a specific word w appears in a specific document
- DF(w) : number of documents a specific word w

## word2vec
A word embedding method that uses information about the 'frequency with which words appear together' to imply the meaning of a word.

### CBOW(Continuous Bag of Words)
A method of predicting the word in the middle using information about surrounding words

### Skip-gram
A method to predict surrounding words using information about the middle word

</br></br>

# Method of Sentence Embedding

## Two ways to compute relationships between sentences

### BERT
1. bi-encoder
    - Input each sentence independently into BERT model, and calculate the similarity between the sentence embedding vectors, which are the output results of the model
2. cross-encoder
    - Two sentences are input together into BERT model, and the model directly outputs the relationship between the two sentences as a value between 0 and 1
    - Since it only calculates the similarity between the two sentences entered as input, there is a problem that the same calculation must be repeated if you want to know the similarity between other sentences and the search query

### Bi-encoder structure

**Bi-encoder** 
```python
from sentence_transformers import SentenceTransformer, models

# BERT model
word_embedding_model = models.Transformer('klue/roberta-base')

# Input pooling layer demension
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

# Combine two modules
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```
```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: RobertaModel 
  (1): Pooling({
  'word_embedding_dimension': 768, 
  'pooling_mode_cls_token': False, 
  'pooling_mode_mean_tokens': True, 
  'pooling_mode_max_tokens': False, 
  'pooling_mode_mean_sqrt_len_tokens': False, 
  'pooling_mode_weightedmean_tokens': False, 
  'pooling_mode_lasttoken': False, 'include_prompt': True
  })
)
```
- `pooling_mode_cls_token`
    - The output embedding of the [CLS] token, which is the first token of the model, is used as the sentence embedding.
- `pooling_mode_mean_tokens`
    - The average of the output embeddings of all input tokens.
    ```python
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] # last layer output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    ```
- `pooling_mode_max_tokens`
    - Find the maximum value in the sentence length direction from the output embedding of all input tokens and use it as the sentence embedding
    ```python
    def max_poolint(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, 1)[0]
    ```

## Create text and image embeddings with Sentence-Transformers

**Korean sentence embedding model**
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

embs = model.encode(['잠이 안 옵니다',
                     '졸음이 옵니다',
                     '기차가 옵니다'])

cos_scores = util.cos_sim(embs, embs)
cos_scores
```
```
tensor([[1.0000, 0.6410, 0.1887],
        [0.6410, 1.0000, 0.2730],
        [0.1887, 0.2730, 1.0000]])
```

**Computation of image and text embeddings using CLIP**
```python
from PIL import Image
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('clip-ViT-B-32')

img_embs = model.encode([Image.open('dog.jpg'), Image.open('cat.jpg')])
text_embs = model.encode(['A dog on grass', 'Brown cat on yellow background'])

cos_scores = util.cos_sim(img_embs, text_embs)
cos_scores
```
```
tensor([[0.2722, 0.1796],
        [0.2220, 0.3563]])
```

</br></br>

# Semantic Search

## Implementation of semantic search

```python
# Load dataset
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

klue_mrc_dataset = load_dataset('klue', 'mrc', split='train')
sentence_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# Procs of semantic search
query = "이번 연도에는 언제 비가 많이 올까?"
query_embedding = sentence_model.encode([query])
distances, indices = index.search(query_embedding, 3)

for idx in indices[0]:
    print(klue_mrc_dataset['context'][idx][:50])

# >>>
# 올여름 장마가 17일 제주도에서 시작됐다. 서울 등 중부지방은 예년보다 사나흘 정도 늦은 
# 연구 결과에 따르면, 오리너구리의 눈은 대부분의 포유류보다는 어류인 칠성장어나 먹장어, 그
# 연구 결과에 따르면, 오리너구리의 눈은 대부분의 포유류보다는 어류인 칠성장어나 먹장어, 그

# Limitation of semantic search
query = klue_mrc_dataset[3]['context']
query_embedding = sentence_model.encode([query])
distances, indices = index.search(query_embedding, 3)

for idx in indices[0]:
    print(klue_mrc_dataset['context'][idx][:50])

# >>>
# 미국 세인트루이스에서 태어났고, 프린스턴 대학교에서 학사 학위를 마치고 1939년에 로체스
# 1950년대 말 매사추세츠 공과대학교의 동아리 테크모델철도클럽에서 ‘해커’라는 용어가 처음
# 1950년대 말 매사추세츠 공과대학교의 동아리 테크모델철도클럽에서 ‘해커’라는 용어가 처음
```

- Sometimes, you may get search results that are not very relevant

## Use Sentence-Transformers in LlamaIndex
```python
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name='snunlp/KR-SBERT-V40K-klueNLI-augSTS')
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)

# Use local model
# service_context = ServiceContext.from_defaults(embed_model="local")

text_list = klue_mrc_dataset[:100]['context']
documents = [Document(text=t) for t in text_list]

index_llama = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context
)
```

</br></br>

# Improve Performance

## Keyward search : BM25
Unlike TF-IDF, it takes into account the saturation effect on the word frequency term and the influence of the document length
```python
import math
import numpy as np
from typing import List
from collections import defaultdict
from transformers import AutoTokenizer, PreTrainedTokenizer

class BM25:
    def __init__(self, corpus: List[List[str]], tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.tokenized_corpus = self.tokenizer(corpus, add_special_tokens=False)['input_ids']
        self.n_docs = len(self.tokenized_corpus)
        self.avg_doc_lens = sum(len(lst) for lst in self.tokenized_corpus) / len(self.tokenized_corpus)
        self.idf = self._calculate_idf()
        self.term_freqs = self._calculate_term_freqs()

    def _calculate_idf(self):
        idf = defaultdict(float)
        for doc in self.tokenized_corpus:
            for token_id in set(doc):
                idf[token_id] += 1
        for token_id, doc_frequency in idf.items():
            idf[token_id] = math.log(((self.n_docs - doc_frequency + 0.5) / (doc_frequency + 0.5)) + 1)
        return idf

    def _calculate_term_freqs(self):
        term_freqs = [defaultdict(int) for _ in range(self.n_docs)]
        for i, doc in enumerate(self.tokenized_corpus):
            for token_id in doc:
                term_freqs[i][token_id] += 1
        return term_freqs

    def get_scores(self, query: str, k1: float=1.2, b: float=0.75):
        query = self.tokenizer([query], add_special_tokens=False)['input_ids'][0]
        scores = np.zeros(self.n_docs)
        for q in query:
            idf = self.idf[q]
            for i, term_freq in enumerate(self.term_freqs):
                q_frequency = term_freq[q]
                doc_len = len(self.tokenized_corpus[i])
                score_q = idf * (q_frequency * (k1 + 1)) / ((q_frequency) + k1 * (1 - b + b * (doc_len / self.avg_doc_lens)))
                scores[i] += score_q
        return scores

    def get_top_k(self, query: str, k: int):
        scores = self.get_scores(query)
        top_k_indices = np.argsort(scores)[-k:][::-1]
        top_k_scores = scores[top_k_indices]
        return top_k_scores, top_k_indices
```
```python
# check score
tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')

bm25 = BM25(['안녕하세요', '반갑습니다', '안녕 서울'], tokenizer)
bm25.get_scores('안녕')

# array([0.44713859, 0.        , 0.52354835])


# limitation of BM25
bm25 = BM25(klue_mrc_dataset['context'], tokenizer)

query = '이번 연도에는 언제 비가 많이 올까?'
_, bm25_search_ranking = bm25.get_top_k(query, 100)

for idx in bm25_search_ranking[:3]:
    print(klue_mrc_dataset['context'][idx][:50])

# 출력 결과
# 갤럭시S5 언제 발매한다는 건지언제는 “27일 판매한다”고 했다가 “이르면 26일 판매한다
# 인구 비율당 노벨상을 세계에서 가장 많이 받은 나라, 과학 논문을 가장 많이 쓰고 의료 특
# 올여름 장마가 17일 제주도에서 시작됐다. 서울 등 중부지방은 예년보다 사나흘 정도 늦은 


# Procs of BM25 search result
query = klue_mrc_dataset[3]['question']
_, bm25_search_ranking = bm25.get_top_k(query, 100)

for idx in bm25_search_ranking[:3]:
    print(klue_mrc_dataset['context'][idx][:50])

# 출력 결과
# 미국 세인트루이스에서 태어났고, 프린스턴 대학교에서 학사 학위를 마치고 1939년에 로체스
# ;메카동(メカドン)
# :성우 : 나라하시 미키(ならはしみき)
# 길가에 버려져 있던 낡은 느티나
# ;메카동(メカドン)
# :성우 : 나라하시 미키(ならはしみき)
# 길가에 버려져 있던 낡은 느티나
```

## Reciprocal Rank Funsion(RRF)
The score is calculated by using the ranking at each point.

```python
from collections import defaultdict

def reciprocal_rank_fusion(rankings: List[List[int]], k=5):
    rrf = defaultdict(float)
    for ranking in rankings:
        for i, doc_id in enumerate(ranking, 1):
            rrf[doc_id] += 1.0 / (k + i)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)

# check
rankings = [[1, 4, 3, 5, 6], [2, 1, 3, 6, 4]]
reciprocal_rank_fusion(rankings)
```
```
[(1, 0.30952380952380953),
 (3, 0.25),
 (4, 0.24285714285714285),
 (6, 0.2111111111111111),
 (2, 0.16666666666666666),
 (5, 0.1111111111111111)]
```

## Hybrid search
```python
def dense_vector_search(query: str, k: int):
    query_embedding = sentence_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return distances[0], indices[0]

def hybrid_search(query, k=20):
    _, dense_search_ranking = dense_vector_search(query, 100)
    _, bm25_search_ranking = bm25.get_top_k(query, 100)

    results = reciprocal_rank_fusion([dense_search_ranking, bm25_search_ranking], k=k)
    return results
```
```python
query = "이번 년도에는 언제 비가 많이 올까?"
print("Search query sentence: ", query)

results = hybrid_search(query)
for idx, score in results[:3]:
    print(klue_mrc_dataset['context'][idx][:50])

print('=' * 80)

query = klue_mrc_dataset[3]['question']
print("Search query sentence: ", query)

results = hybrid_search(query)
for idx, score in results[:3]:
    print(klue_mrc_dataset['context'][idx][:50])
```
```
Search query sentence:  이번 년도에는 언제 비가 많이 올까?
index 0 :  올여름 장마가 17일 제주도에서 시작됐다. 서울 등 중부지방은 예년보다 사나흘 정도 늦은 
index 232 :  갤럭시S5 언제 발매한다는 건지언제는 “27일 판매한다”고 했다가 “이르면 26일 판매한다
index 260 :  “‘마일드세븐’이나 ‘아사히’ 없어도 자영업자나 소비자한테 전혀 지장 없습니다. 담배, 맥
================================================================================
Search query sentence:  로버트 헨리 딕이 1946년에 매사추세츠 연구소에서 개발한 것은 무엇인가?
index 3 :  미국 세인트루이스에서 태어났고, 프린스턴 대학교에서 학사 학위를 마치고 1939년에 로체스
index 326 :  1950년대 말 매사추세츠 공과대학교의 동아리 테크모델철도클럽에서 ‘해커’라는 용어가 처음
index 327 :  1950년대 말 매사추세츠 공과대학교의 동아리 테크모델철도클럽에서 ‘해커’라는 용어가 처음
```