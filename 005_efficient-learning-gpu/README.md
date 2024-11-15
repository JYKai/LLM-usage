# GPU에 올라가는 데이터 살펴보기
기본적으로 GPU에는 딥러닝 모델 자체가 올라간다.   
딥러닝 모델은 수많은 행렬 곱셈을 위한 파라미터의 집합이다.

## 딥러닝 모델의 데이터 타입
컴퓨터는 일반적으로 소수 연산을 위하 32비트 부동소수점(float32)을 사용한다.
- LLM 모델의 용량은 모델을 몇 비트의 데이터 형식으로 표현하는지에 따라 달라진다.
- 최근에는 주로 16비트로 수를 표현하는 fp16 또는 bf16(brain float16)을 주로 사용한다.
    - fp16은 표현할 수 있는 수의 범위가 좁아 딥러닝 연산 과정에서 수를 제대로 표현하지 못하는 문제가 발생한다.
    - bf16은 fp32와 같은 크기의 지수 부분을 사용하여 표현의 범위를 같게 했다.

딥러닝 모델 크기 = 파라미터 수 x 파라미터 당 비트(또는 바이트) 수

## 양자화로 모델 용량 줄이기
더 적은 비트로 모델을 표현하기 위해 양자화(quantization)기술이 개발됐다.
- 양자화 기술에서는 더 적은 비트를 사용하면서도 원본 데이터의 정보를 최대한 소실 없이 유지하는 것이 핵심 과제이다.

## GPU 메모리 분해하기

**GPU에 저장되는 데이터**  
- 모델 파라미터
- 그레이디언트
- 옵티마이저 상태
- 순전파 상태 : 역전파를 수행하기 위해 저장하고 있는 값

**fp16, AdamW와 같은 옵티마이저를 사용할 때 학습에 필요한 최소 메모리**  
- 모델 파라미터 : 2byte * 파라미터 수(B, 10억개) = N
- 그레이디언트 : 2byte * 파라미터 수(B, 10억개) = N
- 옵티마이저 상태 : 2byte * 파라미터 수(B, 10억개) * 2(상태 수) = 2N
- 순전파 상태(배치 크기, 시퀀스 길이, 잠재 상태 크기 등등)

**메모리 사용량 측정을 위한 함수**
```python
def print_gpu_utilization():
    if torch.cuda.is_available():
        used_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory usage: {used_memory:.3f} GB")
    else:
        print("Change Runtime type to GPU.")
```

**그레이디언트와 옵티마이저 상태의 메모리 사용량을 계산하는 함수**  
```python
from transformers import AdamW
from torch.utils.data import DataLoader

# Check gradient memory usage
def estimate_memory_of_gradients(model):
    total_memory = 0
    for param in model.parameters():
        if param.grad is not None:
            total_memory += param.grad.nelement() * param.grad.element_size()
    return total_memory

# Check optimizer state memory usage
def estimate_memory_of_optimizer(optimizer):
    total_memory = 0
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                total_memory += v.nelement() * v.element_size()
    return total_memory
```

</br>
</br>

# 단일 GPU 효율적으로 활용하기

## 그레이디언트 누적
제한된 메모리 안에서 배치 크기를 키우는 것과 동일한 효과를 얻는 방법으로 딥러닝 모델을 학습시킬 때 각 배치마다 모델을 업데이트하지 않고 여러 배치의 학습 데이터를 연산한 후 모델을 업데이트 하는 방법이다.
- 적은 GPU 메모리로도 더 큰 배치 크기와 같은 효과를 얻을 수 있지만, 추가적인 순전파 및 역전파 연산을 수행하기 때문에 학습 시간이 증가된다.

```python
def train_model(model, dataset, training_args):
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataloader = DataLoader(dataset, batch_size=training_args.per_device_train_batch_size)
    optimizer = AdamW(model.parameters())
    model.train()
    gpu_utilization_printed = False

    for step, batch in enumerate(train_dataloader, start=1):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / training_args.gradient_accumulation_steps # 4일 경우, 손실을 4로 나눠서 역전파를 수행
        loss.backward()

        if step % training_args.gradient_accumulation_steps == 0: # 배치 크기가 4배로 커진 것과 동일한 효과
            optimizer.step()
            gradients_memory = estimate_memory_of_gradients(model)
            optimizer_memory = estimate_memory_of_optimizer(optimizer)
            if not gpu_utilization_printed:
                print_gpu_utilization()
                gpu_utilization_printed = True
            optimizer.zero_grad()

    print(f"Optimizer state memory usage: {optimizer_memory / (1024 ** 3):.3f} GB")
    print(f"Gradient memory usage: {gradients_memory / (1024 ** 3):.3f} GB")
```
- `gradient_accumulation_steps` 파라미터 조절을 통한 gradient accumulation.

## 그레이디언트 체크포인팅
순전파의 계산 결과를 모두 저장하지 않고 일부만 저장해 학습 중 GPU 메모리의 사용량을 줄이는 학습 방법이다.
- 추가적인 순전파 계산이 필요하기 때문에 메모리 사용량은 줄지만 학습 시간이 증가한다.

# 분산 학습과 ZeRO

## 분산 학습
분산 학습은 GPU를 여러 개 활용해 딥러닝 모델을 학습시키는 것을 말한다.
- 모델 학습 속도를 높이는 것
- 1개의 GPU로 학습이 어려운 모델을 다루는 것

**데이터 병렬화**  
모델이 작아 하나의 GPU에 올릴 수 있는 경우 여러 GPU에 각각 모델을 올리고 학습 데이터를 병렬로 처리해 학습 속도를 높이는 방법

**모델 병렬화**  
하나의 GPU에 올리기 어려운 큰 모델의 경우 모델을 여러 개의 GPU에 나눠서 올리는 방식
- 파이프라인 병렬화 : 모델의 층(layer)별로 나눠 GPU에 올리는 방법
- 텐서 병렬화 : 한 층의 모델도 나눠서 GPU에 올리는 방법

## 데이터 병렬화에서 중복 저장 줄이기(ZeRO)
하나의 모델을 하나의 GPU에 올리지 않고 마치 모델 병렬화처럼 모델을 나눠 여러 GPU에 올리고 각 GPU에서는 자신의 모델 부분의 연산만 수행하고 그 상태를 저장하면 메모리를 효율적으로 사용하면서 속도도 빠르게 유지할 수 있다는 것이 ZeRO의 컨셉이다.


# 효율적인 학습 방법(PEFT): LoRA

## 모델 파라미터의 일부만 재구성해 학습하는 LoRA
LoRA는 모델 파라미터를 재구성(reparameterization)해 더 적은 파라미터를 학습함으로써 GPU 메모리 사용량을 줄인다.
- 행렬을 더 작은 2개의 행렬의 곱으로 표현해 전체 파라미터를 수정하는 것이 아니라 더 작은 2개의 행렬을 수정하는 것을 의미한다.
- 학습하는 파라미터의 수가 줄어들면 모델 업데이트에 사용하는 옵티마이저 상태의 데이터가 줄어드는데, LoRA를 통해 GPU 메모리 사용량이 줄어드는 부분은 바로 그레이디언트와 옵티마이저 상태를 저장하는 데 필요한 메모리가 줄어든다.

## LoRA 설정 살펴보기
모델 학습에 LoRA를 적용할 때 결정해야 할 사항은 크게 세 가지다.
1. 파라미터에 더할 행렬 A, B를 만들 때 차원 r을 몇으로 할지 정해야 한다.
    - r을 작게 설정하면 학습시켜야 하는 파라미터 수가 줄어들기 때문에 GPU 메모리 사용량을 더 줄일 수 있다. 하지만 학습 데이터의 패턴을 충분히 학습하지 못할 수 있다.
    - 적절한 실험을 통해 값을 찾아야 한다.

2. 추가한 파라미터를 기존 파라미터에 얼마나 많이 반영할지 결정하는 알파를 설정해야 한다.
    - LoRA에서는 행렬 A와 B 부분을 (알파/r)만큼의 비중으로 기존 파라미터 W에 더해준다.
    - 알파가 커질수록 새롭게 학습한 파라미터의 중요성을 크게 고려한다고 볼 수 있다.
    - 학습 데이터에 따라 적절한 알파 값도 달라지기 때문에 적절한 실험을 통해 설정해야 한다.

3. 모델에 있는 많은 파라미터 중에서 어떤 파라미터를 재구성할지 결정해야 한다.
    - 일반적으로 셀프 어텐션 연산의 쿼리, 키, 값 가중치와 피드 포워드 층의 가중치와 같이 선형 연산의 가중치를 재구성한다.
    - 특정 가중치에만 적용할 수도 있고 전체 선형 층에 적용할 수도 있다. 보통은 전체 선형 층에 적용할 경우 성능이 가장 좋다고 알려져 있지만 이 또한 실험이 필요하다.

## 코드로 LoRA 학습 사용하기
허깅페이스는 peft 라이브러리를 통해 LoRA와 같은 효율적인 학습 방식을 쉽게 활용할 수 있는 기능을 제공한다.
- peft 라이브러리에서 LoraConfig 클래스를 사용하면 LoRA를 적용할 때 사용할 설정을 정의할 수 있다.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def load_model_and_tokenizer(model_id, peft=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if peft is None:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map={"": 0})
    elif peft == 'loar':
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map={"": 0})
        lora_config = LoraConfig(
                        r=8,
                        lora_alpha=32,
                        target_modules=["query_key_value"],
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    print_gpu_utilization()

    return model, tokenizer
```
- `lora_config` 설정 후 `get_peft_model`을 통해 모델 적용
```
trainable params: 1,572,864 || all params: 1,333,383,168 || trainable%: 0.11796039111242178
GPU memory usage: 2.602 GB
GPU memory usage: 4.732 GB
Optimizer state memory usage: 0.006 GB
Gradient memory usage: 0.003 GB
```

</br></br>

# 효율적인 학습 방법(PEFT): QLoRA

## 4비트 양자화와 2차 양자화
학습된 모델 파라미터는 거의 정규 분포에 가깝다고 알려져 있다. 따라서, 입력이 정규 분포라는 가정을 활용하면 모델의 성능을 거의 유지하면서도 빠른 양자화가 가능해진다.

## 페이지 옵티마이저
그레디언트 체크포인팅 과정에서 발생할 수 있는 OOM 에러를 방지하기 위해 페이지 옵티마이저를 활용한다.

페이지 옵티마이저란, 엔비디아의 통합 메모리를 통해 GPU가 CPU 메모리(RAM)를 공유하는 것을 말한다.

```python
def load_model_and_tokenizer(model_id, peft=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if peft is None:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map={"":0})

    elif peft == 'lora':
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map={"":0})
        lora_config = LoraConfig(
                        r=8,
                        lora_alpha=32,
                        target_modules=["query_key_value"],
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    elif peft == 'qlora': # QLoRA
        lora_config = LoraConfig(
                        r=8,
                        lora_alpha=32,
                        target_modules=["query_key_value"],
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM"
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
            
    
    print_gpu_utilization()
    return model, tokenizer
```

```
GPU memory usage: 1.167 GB
Optimizer state memory usage: 0.012 GB
Gradient memory usage: 0.006 GB
```