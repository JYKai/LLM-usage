# 언어 모델 추론 이해하기

## 언어 모델이 언어를 생성하는 방법
언어 모델은 입력한 텍스트 다음에 올 토큰의 확률을 계산하고 그중에서 가장 확률이 높은 토큰을 입력 텍스트에 추가하면서 한 토큰씩 생성한다.

**언어 모델이 텍스트 생성을 마치는 이유**  
1. 다음 토큰으로 생성 종료를 의미하는 특수 토큰(예: EOS)을 생성하는 경우
2. 사용자가 최대 길이로 설정한 길이에 도달할 경우

위의 경우에 해당하기 전까지는 새로운 토큰을 추가한 텍스트를 다시 모델에 입력으로 넣는 과정을 반복한다.

언어 모델은 입력 텍스트를 기반으로 바로 다음 토큰만 예측하는 자기 회귀적(auto-regressive) 특성을 갖는다.
- 입력 텍스트(프롬프트)의 경우 이미 작성된 텍스트이기 때문에 한 번에 하나씩 토큰을 처리할 필요 없이 동시에 병렬적으로 처리할 수 있다. 즉, 프롬프트가 길어도 다음 토큰 1개를 생성하는 시간과 비슷한 시간이 걸린다.
    - 이런 이유로 추론 과정을 프롬프트를 처리하는 단계인 사전 계산 단계(prefill phase)와 이후에 한 토큰씩 생성하는 디코딩 단계(decoding phase)로 구분한다.

이러한 처리 과정에서 동일한 토큰이 반복해서 입력으로 들어가면 동일한 연산을 반복적으로 수행하기 때문에 비효율적이다.

## 중복 연산을 줄이는 KV 캐시
KV(Key-Value) 캐시는 셀프 어텐션 연산 과정에서 동일한 입력 토큰에 대해 중복 계산이 발생하는 비효율을 줄이기 위해 먼저 계산했던 키와 값 결과를 메모리에 저장해 활용하는 방법을 말한다.

**KV 캐시 저장을 위한 메모리 사용량**  
- KV 캐시 메모리 = 2바이트(fp16) x 2(키와 값) x (레이어 수) x (토큰 임베딩 차원) x (최대 시퀀스 길이) x (배치 크기)

## GPU 구조와 최적의 배치 크기
**서빙이 효율적인지 판단하는 큰 기준**  
1. 비용
2. 처리량(throughput)
    - 시간당 처리한 요청 수(query/s)
3. 지연 시간(latency)
    - 하나의 토큰을 생성하는 데 걸리는 시간(token/s)

즉, 적은 비용으로 더 많은 요청을 처리하면서 생성한 다음 토큰을 빠르게 전달할 수 있다면 효율적인 서빙이라고 할 수 있다.

## KV 캐시 메모리 줄이기
트랜스포머 모델이 셀프 어텐션을 수행할 때는 한 번의 어텐션 연산만 수행하는 것이 아니라 멀티 헤드 어텐션을 사용하기 때문에 KV 캐시에 더 많은 메모리를 사용하고 KV 캐시에서 더 많은 데이터를 불러와 계산하기 때문에 그만큼 속도가 느려진다.

**멀티 쿼리 어텐션(Multi-Query Attention, MQA)**  
멀티 쿼리 어텐션 방식은 여러 헤드의 쿼리 벡터가 하나의 키와 값 벡터를 사용한다. 즉, 멀티 쿼리 어텐션은 하나의 키와 값 벡터만 저장하기 때문에 KV 캐시를 저장하는 데 훨씬 적은 메모리를 사용한다. 하지만, 성능이 떨어지는 문제가 있다.

**그룹 쿼리 어텐션(Grouped-Query Attention, GQA)**  
그룹된 쿼리 벡터당 1개의 키와 값 벡터를 사용함으로써 결과적으로 멀티 헤드 어텐션에 비해 더 적은 키와 값 벡터를 사용한다.

멀티 쿼리 어텐션의 경우 멀티 헤드 어텐션과 비교했을 때 성능 저하가 뚜렷하여, 기존의 학습 데이터로 추가 학습(uptraining)을 수행한다.
- 기존 학습 데이터의 10%까지 사용해도 여전히 성능 차이가 발생한 연구 결과가 있다.

그룹 쿼리 어텐션을 사용한 경우 추가 학습을 하지 않더라도 멀티 헤드 어텐션과 성능 차이가 크지 않고, 기존 학습 데이터의 약 5%만 사용해 추가 학습을 수행해도 성능 차이가 거의 없다는 연구 결과가 있다.

</br></br>

# 양자화로 모델 용량 줄이기
양자화란 부동소수점 데이터를 더 적은 메모리를 사용하는 정수 형식으로 변환해 GPU를 효율적으로 사용하는 방법을 말한다.
- FP32 -> FP16, BF16

16비트 파라미터는 보통 8, 4, 3비트로 양자화하는데, 최근에는 4비트로 모델 파라미터를 양자화하고 계산은 16비트로 하는 **W4A16(Weight 4bits and Activation 16bits)**을 주로 활용한다.

양자화는 양자화를 수행하는 시점에 따라 학습 후 양자화(Post-Training Quantization, PTQ)와 양자화 학습(Quantization-Aware Training, QAT)으로 나뉜다.
- LLM은 학습 후 양자화를 주로 활용한다.
- 양자화 방식
    - 비츠앤바이츠(bits-and-bytes)
    - GPTQ(GPT Quantization)
    - AWQ(Activation-aware Weight Quantization)

## 비츠앤바이츠
비츠앤바이츠 라이브러리는 크게 두 가지 양자화 방식을 제공한다.
1. 8비트로 연산을 수행하면서도 성능 저하가 거의 없이 성능을 유지하는 8비트 행렬 연산
2. 4비트 정규 분포 양자화 방식

비츠앤바이츠의 경우 입력 x의 값 중 크기가 큰 이상치가 포함된 열은 별도로 분리해서 16비트 그대로 계산했다.
- 입력에서 값이 큰 경우 중요한 정보를 담고 있다고 판단해 정보가 손실되지 않도록 양자화하지 않고 그대로 연산만

정상 범위에 있는 열을 양자화할 때 벡터 단위(입력의 행, 모델의 열)로 절대 최댓값을 찾고 그 값을 기준으로 양자화를 수행한다.

비츠앤바이츠가 지원하는 8비트, 4비트 양자화를 사용하려면 모델을 불러올 때 양자화 설정을 전달하면 된다.
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8-bit quantization model
bnb_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model_8bit = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", quantization_config=bnb_config_8bit)

# 4-bit quantization model
bnb_config_4bit = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')
model_4bit = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", quantization_config=bnb_config_4bit, low_cpu_mem_usage=True)
```

## GPTQ
양자화 이전의 모델에 입력 x를 넣었을 때와 양자화 이후의 모델에 입력 x를 넣었을 때 오차가 가장 작아지도록 모델의 양자화를 수행한다.
- 직관적으로 봤을 때 양자화 전과 후의 결과 차이가 작다면 훌륭한 양자화라고 볼 수 있다.

GPTQ는 양자화를 위한 작은 데이터셋을 준비하고 그 데이터셋을 활용해 모델 연산을 수행하면서 양자화 이전과 유사한 결과가 나오도록 모델을 업데이트한다.
- GPTQ 양자화를 수행하기 위해서는 시간이 걸리는데, 논문에 따르면 175B 모델을 양자화하는 데 A100 GPU로 4시간이 걸린다고 한다.
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)
```

## AWQ
모든 파라미터가 동등하게 중요하지는 않으며 특별히 중요한 파라미터의 정보를 유지하면 양자화를 수행하면서도 성능 저하를 막을 수 있다는 아이디어에서 출발.

특별히 중요한 파라미터는 어떻게 찾을 수 있을까?  
1. 모델 파라미터의 값이 크다면 중요하다고 예상할 수 있다.
2. 입력 데이터의 활성화 값이 큰 채널의 파라미터가 중요하다고 가정할 수 있다.

활성화 값을 기준으로 중여한 1% 파라미터의 정보만 지키면 모델의 성능이 유지된다는 사실을 발견했다.
- 하지만 모델 파라미터에 서로 다른 데이터 타입이 섞여 있는 경우 한 번에 일괄적으로 연산하기 어렵기 때문에 연산이 느려지고 하드웨어 효율성이 떨어지는 문제가 발생한다.

양자화로 인해 정보가 손실되는 것을 막기 위해 MIT 연구진은 중요한 파라미터에만 1보다 큰 값(스케일러)을 곱하는 방식으로 문제를 해결했다.
- 스케일러가 2일 때까지는 성능이 향상되지만 2를 넘어가는 경우 성능이 다시 하락한다는 것을 확인했다.
- 스케일러가 클 경우 파라미터에서 가장 큰 수에 맞춰 나머지 파라미터도 양자화되기 때문에 나머지 파라미터가 좁은 범위로 변환되면서 정보 소실이 발생할 수 있다.

AWQ는 모델의 활성화 값 분포를 통해 중요한 파라미터를 결정하고 양자화를 수행하기 때뭉네 양자화에 많은 시간이 걸리지 않고 기존 모델의 성능을 거의 유지할 수 있어 활발히 활용되고 있다.
```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name_or_path = "TheBloke/zephyr-7B-beta-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True, trust_remote_code=False, safetensors=True)
```

**GGUF 형식**  
GGUF(Georgi Gerganov Unified Format)
- GGUF 형식의 양자화 모델은 GPU는 물론 애플의 실리콘 칩을 포함한 다양한 CPU에서 모델 추론 가능
- 모델의 일부는 CPU에 두고 일부만 GPU에서 실행 가능 &rarr; 온디바이스 환경에 유리
- 하나의 파일에 추론을 위한 모든 정보를 담을 수 있음 &rarr; 배포 과정 간소화

</br></br>

# 지식 증류 활용하기
지식 증류(knowledge distillation)란, 더 크고 성능이 높은 선생 모델의 생성 결과를 활용해 더 작고 성능 낮은 학생 모델을 만드는 방법을 말한다.
- 학생 모델은 선생 모델의 생성 결과를 모방하는 방식으로 학습한다.

최근, sLLM의 학습 데이터 구축에 GPT-4와 같은 대형 모델을 활용하는 경우가 일반적이다.
- 허깅페이스가 제퍼를 개발할 때는 지시 데이터셋의 구축과 선호 데이터셋의 구축에 모두 LLM을 사용했다.