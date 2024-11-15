# Text2SQL 데이터셋

## 대표적인 Text2SQL 데이터셋
- WikiSQL
    - 하나의 테이블만 사용하고 SELECT 문에 컬럼을 1개만 사용하거나 조건(WHERE)절에 최대 3개의 조건만 사용하는 등 비교적 쉬운 SQL 문으로 구성되어 있다.

- Spider
    - 좀 더 현실적인 문제 해결을 위해 구축된 데이터셋으로 ORDER BY, GROUP BY, HAVING, JOIN 등 비교적 복잡한 SQL 문도 포함하고 있다.

SQL을 생성하기 위해서는 크게 두 가지 데이터가 필요하다.
1. 어떤 데이터가 있는지 알 수 있는 데이터베이스 정보(테이블과 컬럼)
2. 어떤 데이터를 추출하고 싶은지 나타낸 요청사항(request 또는 question)

## 한국어 데이터셋
- NL2SQL
    - AI 허브에서 구축한 자연어 기반 질의 검색 생성 데이터

## 합성 데이터 활용
현재는 한국어 Text2SQL 작업을 위해 활용할 수 있는 데이터셋이 없기 때문에 실습에서 활용할 데이터셋을 GPT-3.5, GPT-4를 활용해 생성 

>[데이터 셋 위치](https://huggingface.co/datasets/shangrilar/ko_text2sql)

**데이터셋 구성**
- db_id
    - 테이블이 포함된 데이터베이스의 아이디
    - 동일한 db_id를 갖는 테이블은 같은 도메인을 공유한다.
- context
    - SQL 생성에 사용할 테이블 정보
- question
    - 데이터 요청사항
- answer
    - 요청에 대한 SQL 정답

</br></br>

# 성능 평가 파이프라인 준비하기
머신러닝 모델을 학습시킬 때는 학습이 잘 진행된 것인지 판단할 수 있도록 성능 평가 방식을 미리 정해야 한다.

## Text2SQL 평가 방식
LLM이 생성한 SQL이 데이터 요청을 잘 해결하는지 GPT-4를 사용해 확인한다.

**GPT를 활용한 성능 평가 파이프라인 준비**  
- 평가 데이터셋 구축
- LLM이 SQL을 생성할 때 사용할 프롬프트
- GPT 평가에 사용할 프롬프트와 GPT-4 API 요청을 빠르게 할 수 있는 코드 작성

## 평가 데이터셋 구축
112개의 데이터
[링크](https://huggingface.co/datasets/shangrilar/ko_text2sql/viewer/default/test)

## SQL 생성 프롬프트
LLM이 SQL을 생성하도록 하기 위해서는 지시사항과 데이터를 포함한 프롬프트를 준비해야 한다.
- LLM의 경우 학습에 사용한 프롬프트 형식을 추론할 때도 동일하게 사용해야 결과 품질이 좋기 때문에 준비한 프롬프트 형식은 이어지는 모델의 미세 조정할 때도 동일하게 적용한다.

```python
def make_prompt(ddl, question, query=''):
    prompt = f"""당신은 SQL을 생성하는 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결할 수 있는 SQL 쿼리를 생성하세요.
### DDL:
{ddl}

### Question:
{question}

### SQL:
{query}"""
    return prompt
```

## GPT-4 평가 프롬프트와 코드 준비
GPT-4를 사용해 평가를 수행한다면 반복적으로 GPT-4 API 요청을 보내야 한다.
- 이번에 평가로 사용할 112개의 데이터의 수준이라면 for 문을 통해 반복적인 요청을 수행해도 시간이 오래걸리지 않지만, 평가 데이터셋을 더 늘린다면 시간이 오래걸린다.
    - OpenAI가 openai-cookbook 깃허브 저장소에서 제공하는 코드를 활용해 요청 제한을 관리하면서 비동기적으로 요청을 보낼 수 있다.

```python
def make_requests_for_gpt_evaluation(df, filename, dir='requests'):
    if not Path(dir).exists():
        Path(dir).mkdir(parents=True)
    prompts = []
    for idx, row in df.iterrows():
        prompts.append("""Based on below DDL and Question, evaluate gen_sql can resolve Question. If gen_sql and gt_sql do equal job, return "yes" else "no". Output Json Format: {"resolve_yn": ""}""" + f"""

DDL: {row['context']}
Question: {row['question']}
gt_sql: {row['answer']}
gen_sql: {row['gen_sql']}"""
)

    jobs = [{"model": "gpt-4-turbo", "response_format": {"type": "json_object"}, "messages": [{"role": "system", "content": prompt}]} for prompt in prompts]
    with open(Path(dir, filename), "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")
```

# 미세 조정 수행하기

## 기초 모델 평가하기
기초 모델을 불러와 프롬프트에 대한 결과를 생성한다.

```python
def make_inference_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

model_id = 'beomi/Yi-Ko-6B'
hf_pipe = make_inference_pipeline(model_id)

example = """당신은 SQL을 생성하는 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결할 수 있는 SQL 쿼리를 생성하세요.

### DDL:
CREATE TABLE players (
    player_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    date_joined DATETIME NOT NULL,
    last_login DATETIME
);

### Question:
사용자 이름에 'admin'이 포함되어 있는 계정의 수를 알려주세요.

### SQL:
"""

hf_pipe(example, do_sample=False, return_full_text=False, max_length=1024, truncation=True)
```

## 미세 조정 수행
모델의 미세 조정에는 autotrain-advanced 라이브러리를 활용한다.
- 허깅페이스에서 trl 라이브러리를 한 번 더 추상화해 개발한 라이브러리다.

