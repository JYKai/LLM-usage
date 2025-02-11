# 딥러닝과 언어 모델링
LLM은 다음에 올 단어가 무엇일지 예측하면서 문장을 하나씩 만들어 가는 방식으로 텍스트를 생성한다.
- 언어모델(Language model) : 다음에 올 단어를 예측하는 모델을 말한다.

## 데이터의 특징을 스스로 추출하는 딥러닝
딥러닝이 머신러닝과 가장 큰 차이를 보이는 지점은 **'데이터의 특징을 누가 뽑는가?'** 이다.

## 임베딩: 딥러닝 모델이 데이터를 표현하는 방식
데이터의 의미와 특징을 포착해 숫자로 표현한 것을 **임베딩(embedding)** 이라고 부른다.
- 임베딩은 거리를 계산할 수 있기 때문에 검색 및 추천, 클러스터링 및 분류, 이상치 탐지 등과 같은 작업에 활용될 수 있다.
- 딥러닝 모델은 데이터를 통해 학습하는 과정에서 그 데이터를 가장 잘 이해할 수 있는 방식을 함께 배우며, 데이터의 의미를 숫자로 표현한 것이 바로 임베딩이다.

단어를 임베딩으로 변환한 것을 일컬어 단어 임베딩(word embedding)이라고 한다.

## 언어 모델링: 딥러닝 모델의 언어 학습법
언어 모델링이란, 모델이 입력받은 텍스트의 다음 단어를 예측해 텍스트를 생성하는 방식을 말한다.  

**전이 학습(transfer learning)**  
하나의 문제를 해결하는 과정에서 얻은 지식과 정보를 다른 문제를 풀 때 사용하는 방식  
- 사전 학습에 사용한 이미지와 현재 풀고자 하는 과제의 이미지가 다르더라도 선이나 점 같은 특징을 파악하는 능력은 공통적으로 필요하다.
- 사전 학습 모델을 미세 조정해 풀고자 하는 과제를 흔히 다운스트림(downstream)과제라고 부른다.
- 전이 학습은 학습 데이터가 적은 경우에 특히 유용하다.

</br>

# 언어 모델이 챗GPT가 되기까지

## RNN에서 트랜스포머 아키텍처로
작은 단위(단어)의 데이터가 연결되고, 그 길이가 다양한 데이터의 형태를 '시퀀스(sequence)'라 한다.  

RNN은 입력하는 텍스트를 순차적으로 처리해서 다음 단어를 예측한다.
- '하나의 잠재 상태(hidden state)에 지금까지의 입력 텍스트의 맥락을 압축'한다. 즉, 텍스트의 맥락을 압축하고 다음 단어를 예측한다.
- 지속적으로 단어를 잠재 상태에 압축하다 보면 먼저 입력한 단어의 의미가 점차 희석되어 입력이 길어지는 경우 의미를 충분히 담지 못하고 성능이 떨어지는 문제가 발생한다.  

트랜스포머 아키텍처는 맥락을 모두 참조하는 어텐션 연산을 사용하여 RNN의 문제를 대부분 해결했다.
- 맥락 데이터를 그대로 모두 활용해 다음 단어를 예측한다.
- 맥락을 압축하지 않고 그대로 활용하기 때문에 성능을 높일 수 있지만, 입력 텍스트가 길어지면 메모리 사용량이 증가한다. 또한, 매번 다음 단어를 예측할 때마다 맥락 데이터를 모두 확인해야 하기 때문에 입력이 길어지면 예측에 걸리는 시간도 증가한다.

## GPT 시리즈로 보는 모델 크기와 성능의 관계
왜 모델의 크기가 커지고 학습 데이터가 많을수록 모델의 성능이 높아질까?
- 언어 모델의 경우 학습 데이터와 언어 모델의 결과가 모두 '생성된 언어'다. 따라서, 언어 모델이 학습하는 과정을 학습 데이터를 압축하는 과정으로 해석할 수 있다.
    - 여기에서 압축이란 공통되고 중요한 패턴을 남기는 손실 압축을 말한다.

## 챗GPT의 등장
GPT-3는 그저 사용자의 말을 이어서 작성하는 능력밖에 없었다. 이러한 GPT를 현재의 챗GPT로 바꾼 것은 아래의 기술이다.
1. 지도 미세 조정(supervised fine-tuning)
- LLM이 생성하는 답변을 사용자의 요청 의도에 맞추는 것을 정렬(alignment)라 한다. 지도 미세 조정은 정렬을 위한 가장 핵심적인 학습 과정으로서, 언어 모델링으로 사전 학습한 언어 모델을 지시 데이터셋(instruction dataset)으로 추가 학습한 것을 뜻한다.
    - 지시 데이터셋은 사용자가 요청 또는 지시한 사항과 그에 대한 적절한 응답을 정리한 데이터셋이다.

2. RLHF(Reinforcement Learning from Human Feedback)
- 두 가지 답변 중 사용자가 더 선호하는 답변을 선택한 데이터셋을 구축했는데, 이를 선호 데이터셋(preference dataset)이라고 한다.
- 선호 데이터셋으로 LLM의 답변을 평가하는 리워드 모델을 만들고 LLM이 점점 더 높은 점수를 받을 수 있도록 추가 학습하는데, 이때 강화 학습을 사용하므로 이 기술을 일컬어 RLHF라고 불렀다.

## LLM 애플리케이션의 시대가 열리다.

### 지식 사용법을 획기적으로 바꾼 LLM
LLM이 사회에 큰 영향을 미치고 있는 이유는 '다재다능함' 때문이다.
- LLM은 언어 이해와 생성 두 측면 모두에서 뛰어나고, 사용자의 요청에 맞춰 다양한 작업을 수행하는 '다재다능함'을 가졌다.

### sLLM: 더 작고 효율적인 모델 만들기
상업용 모델은 오픈소스 LLM에 비해 모델이 크고 범용 텍스트 생성 능력이 뛰어나다. 하지만, 오픈소스 LLM은 원하는 도메인의 데이터, 작업을 위한 데이터로 자유롭게 추가 학습할 수 있다는 장점이 있다.
- 추가 학습을 하는 경우 모델 크기가 작으면서도 특정 도메인 데이터나 작업에서 높은 성능을 보이는 모델을 만들 수 있는데 이를 sLLM이라고 한다.
- 2024년 4월 메타는 라마-3 모델을 오픈소스로 공개하면서 sLLM 연구를 리드하고 있다.

### 더 효율적인 학습과 추론을 위한 기술
LLM은 많은 연산량을 빠르게 처리하기 위해 다른 딥러닝 모델과 마찬가지로 GPU를 사용한다. 이런 배경에서 LLM을 학습하고 추론할 때 GPU를 더 효율적으로 사용해 적은 GPU 자원으로도 LLM을 활용할 수 있도록 돕는 연구가 활발히 진행되고 있다.

1. 양자화(Quantization)  
모델 파라미터를 더 적은 비트로 표현

2. LoRA(Low Rank Adaptation)  
모델 전체를 학습하는 것이 아니라 모델의 일부만 학습

3. 무거운 어텐션 연산을 개선해 효율적인 학습과 추론을 가능하게 하는 연구

### LLM의 환각 현상을 대처하는 검색 증강 생성(RAG) 기술
'환각 현상'이란 LLM이 잘못된 정보나 실제로 존재하지 않는 정보를 만들어 내는 현상을 말한다.
- 기본적으로 LLM은 학습 데이터를 압축해 그럴듯한 문장을 만들 뿐 어떤 정보가 사실인지, 가짓인지 학습한 적은 없어 특정 정보가 사실인지 판단할 능력은 없다.
- 또한, 학습 데이터를 압축하는 과정에서 비교적 드물게 등장하는 정보는 소실될 텐데, 그런 정보의 소실이 부정확한 정보를 생성하는 원인이 될 수도 있다.

이러한 환각 현상을 줄이기 위해 검색 증강 생성(Retrieval Augmented Generation, RAG)을 사용한다.
- RAG는 프롬프트에 LLM이 답변할 때 필요한 정보를 미리 추가함으로써 잘못된 정보를 생성하는 문제를 줄인다.

</br>

# LLM의 미래: 인식과 행동의 확장
1. Multi-modal LLM
    - 더 다양한 형식의 데이터를 입력으로 받을 수 있고 출력으로도 여러 형태의 데이터를 생성할 수 있도록 발전시키는 연구
    
2. Agent LLM
    - 텍스트 생성 능력을 사용해 계획을 세우거나 의사결정을 내리고 필요한 행동까지 수행하는 연구
    - 에이전트 프레임워크: 단순히 텍스트를 생성하는 기능 외에 스스로 판단하고 행동하는 에이전트의 '두뇌'로 사용하려는 시도도 늘고 있다. = AutoGPT

3. 트렌스포머 아키텍처를 새로운 아키텍처로 변경
    - 오디오나 비디오와 같이 긴 입력을 효율적으로 처리하려는 연구