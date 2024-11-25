# Two ways to improve search performance

1. Improve performance by additionally training with the dataset you want to use the sentence embedding model for

2. Add a cross-encoder to improve search performance

</br></br>

# Creating a langugae model as an embedding model

## Contrastive learning
A learning method that brings related or similar data closer together and moves related or dissimilar data futher apart

## Prepare training

**Creating embedding model using pre-trained language model**
```python
from datasets import load_dataset
from torch.utils.data import DataLoader
from sentence_transformers import InputExample # format for managing data in Sentence-Transformers
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

transformer_model = models.Transformer('klue/roberta-base')

pooling_layer = models.Pooling(
    transformer_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)
embedding_model = SentenceTransformer(modules=[transformer_model, pooling_layer])

klue_sts_train = load_dataset('klue', 'sts', split='train')
klue_sts_test = load_dataset('klue', 'sts', split='validation')

# split train, validation datasets using training data
klue_sts_train = klue_sts_train.train_test_split(test_size=0.1, seed=42)
klue_sts_train, klue_sts_eval = klue_sts_train['train'], klue_sts_train['test']

# Normalization similarity score to 0 ~ 1 -> IndexExample
def prepare_sts_examples(dataset):
    examples = []
    for data in dataset:
        examples.append(
            InputExample(
                texts=[data['sentence1'], data['sentence2']],
                label=data['labels']['label'] / 5.0)
        )
    return examples

train_examples = prepare_sts_examples(klue_sts_train)
eval_examples = prepare_sts_examples(klue_sts_eval)
test_examples = prepare_sts_examples(klue_sts_test)

# make dataset for training
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Prepare evaluation object for validation
eval_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(eval_examples)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_examples)

test_evaluator(embedding_model)
# 0.36460670798564826
```

## Training an embedding model with similar sentence data

**Training embedding model**
```python
from sentence_transformers import losses

num_epochs = 4
model_name = 'klue/roberta-base'
model_save_path = 'output/training_sts_' + model_name.replace('/', '-')
train_loss = losses.CosineSimilarityLoss(model=embedding_model)

embedding_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=eval_evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=100,
    output_path=model_save_path
)
```

**Evaluate trained embedding model performace**
```python
trained_embedding_model = SentenceTransformer(model_save_path)
test_evaluator(trained_embedding_model)

# 0.8891355260276683
```

</br></br>

# Fine-tuning Embedding Model
## Prepare training
```py
# Dataset check
from datasets import load_dataset
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

klue_mrc_train = load_dataset('klue', 'mrc', split='train')

# Load basic embedding model
sentence_model = SentenceTransformer('Noahyun/klue-roberta-base-klue-sts')

# Preprocess data
klue_mrc_train = load_dataset('klue', 'mrc', split='train')
klue_mrc_test = load_dataset('klue', 'mrc', split='validation')

df_train = klue_mrc_train.to_pandas()
df_test = klue_mrc_test.to_pandas()

df_train = df_train[['title', 'question', 'context']]
df_test = df_test[['title', 'question', 'context']]

# Add irrelevant context
def add_ir_context(df):
    irrelevant_contexts = []
    for idx, row in df.iterrows():
        title = row['title']
        irrelevant_contexts.append(df.query(f"title != '{title}'").sample(n=1)['context'].values[0])
    df['irrelevant_context'] = irrelevant_contexts
    return df

df_train_ir = add_ir_context(df_train)
df_test_ir = add_ir_context(df_test)

# Make data for evaluation performance
examples = []
for idx, row in df_test_ir.iterrows():
    examples.append(
        InputExample(texts=[row['question'], row['context']], label=1)
    )
    examples.append(
        InputExample(texts=[row['question'], row['irrelevant_context']], label=0)
    )

# Evaluation
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    examples
)
evaluator(sentence_model)

# 0.8217415827013919
```

## Fine-tuning using MNR loss
**Multiple Negatives Ranking(MNR)**  
- Good to use when there are only related sentences
- Train the model by using unrelated data from other data in one batch of data
```py
# Datasets
from sentence_transformers import datasets, losses
train_samples = []
for idx, row in df_train_ir.iterrows():
    train_samples.append(InputExample(texts=[row['question'], row['context']]))

# Remove duplicates
batch_size = 16
loader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=batch_size)

# Load MNR loss
loss = losses.MultipleNegativesRankingLoss(sentence_model)

# Fine-tuning
epochs = 1
save_path = './klue_mrc_mnr'

sentence_model.fit(
    train_objectives=[(loader, loss)],
    epochs=epochs,
    warmup_steps=100,
    output_path=save_path,
    show_progress_bar=True
)

# Evaluate
evaluator(sentence_model)

# 0.8594708084199976
```

</br></br>

# Reorder the rankings
**cross-encoder**
```py
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator

cross_model = CrossEncoder('klue/roberta-small', num_labels=1)

# evaluation cross encoder
ce_evaluator = CECorrelationEvaluator.from_input_examples(examples)
ce_evaluator(cross_model) # -0.025302776035606954

# prepare dataset
train_samples = []
for idx, row in df_train_ir.iterrows():
    train_samples.append(InputExample(texts=[row['question'], row['context']], label=1))
    train_samples.append(InputExample(texts=[row['question'], row['irrelevant_context']], label=0))

# train cross encoder
train_batch_size = 16
num_epochs = 1
model_save_path = 'output/training_mrc'

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

cross_model.fit(
    train_dataloader=train_dataloader,
    epochs=num_epochs,
    warmup_steps=100,
    output_path=model_save_path
)

# Evaluate
ce_evaluator(cross_model) # 0.8648947632389092
```

</br></br>

# RAG Implementation with bi-encoder and cross-encoder
```py
import time
import faiss
from datasets import load_dataset

# dataset sampling for test
klue_mrc_test = load_dataset('klue', 'mrc', split='validation')
klue_mrc_test = klue_mrc_test.train_test_split(test_size=1000, seed=42)['test']

# Implement function to store and retrieve embeddings
def make_embedding_index(sentence_model, corpus):
    embeddings = sentence_model.encode(corpus)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def find_embedding_top_k(query, sentence_model, index, k):
    embedding = sentence_model.encode([query])
    distances, indices = index.search(embedding, k)
    return indices

# Rerank orders
def make_question_context_pairs(question_idx, indices):
    return [[klue_mrc_test['question'][question_idx], klue_mrc_test['context'][idx]] for idx in indices]

def rerank_top_k(cross_model, question_idx, indices, k):
    input_examples = make_question_context_pairs(question_idx, indices)
    relevance_scores = cross_model.predict(input_examples)
    reranked_indices = indices[np.argsort(relevance_scores)[::-1]]
    return reranked_indices

# Metric: hit rate
def evaluate_hit_rate(datasets, embedding_model, index, k=10):
    start_time = time.time()
    predictions = []
    for question in datasets['question']:
        predictions.append(find_embedding_top_k(question, embedding_model, index, k)[0])
    total_prediction_count = len(predictions)
    hit_count = 0
    questions = datasets['question']
    context = datasets['context']
    for idx, prediction in enumerate(predictions):
        for pred in prediction:
            if context[pred] == context[idx]:
                hit_count += 1
                break

    end_time = time.time()
    return hit_count / total_prediction_count, end_time - start_time
```

**Retrieval**
```py
import time
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

# Retrieval using base embedding model
base_embedding_model = SentenceTransformer('Noahyun/klue-roberta-base-klue-sts')
base_index = make_embedding_index(base_embedding_model, klue_mrc_test['context'])

evaluate_hit_rate(klue_mrc_test, base_embedding_model, base_index, 10)
# (0.87, 3.5601565837860107)

# Retrieval using fine-tuned embedding model
finetuned_embedding_model = SentenceTransformer('Noahyun/klue-roberta-base-klue-sts-mrc')
finetuned_index = make_embedding_index(finetuned_embedding_model, klue_mrc_test['context'])

evaluate_hit_rate(klue_mrc_test, finetuned_embedding_model, finetuned_index, 10)
# (0.946, 3.6130285263061523)

# Retrieval using combination with fine-tuned embedding model and cross encoder
## Evaluate metric including order rerank
def evaluate_hit_rate_with_rerank(datasets, embedding_model, cross_model, index, bi_k=30, cross_k=10):
    start_time = time.time()
    predictions = []
    for question_idx, question in enumerate(tqdm(datasets['question'])):
        indices = find_embedding_top_k(question, embedding_model, index, bi_k)[0]
        predictions.append(rerank_top_k(cross_model, question_idx, indices, k=cross_k))
    total_prediction_count = len(predictions)
    hit_count = 0
    questions = datasets['question']
    contexts = datasets['context']
    for idx, prediction in enumerate(predictions):
        for pred in prediction:
            if contexts[pred] == contexts[idx]:
                hit_count += 1
                break
    end_time = time.time()
    return hit_count / total_prediction_count, end_time - start_time, predictions

cross_model = SentenceTransformer('shangrilar/klue-roberta-small-cross-encoder')

hit_rate, cosumed_time, predictions = evaluate_hit_rate_with_rerank(klue_mrc_test, finetuned_embedding_model, cross_model, finetuned_index)
hit_rate, cosumed_time
# (0.973)
```