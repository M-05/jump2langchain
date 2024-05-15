# jump2langchain

`Document Loader`
```python
from langchain_community.document_loaders import ArxivLoader

docs = ArxivLoader(query="2403.05568", load_max_docs=2).load()
with open("./data/langchainPaper.txt", "w") as f:
    f.write(docs[0].page_content[480:])
```


`CharacterTextSplitter` : 기본적으로 "\n\n" 을 기준으로 문자 단위로 텍스트를 분할하고, 청크의 크기를 문자 수로 측정
```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50,
                                      length_function=len,
                                      is_separator_regex=False,)
split_docs = TextLoader("data/langchainPaper.txt").load_and_split(text_splitter)
```

`HuggingFaceEmbeddings` : 텍스트 데이터를 숫자로 이루어진 벡터로 변환하는 과정
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2" # maximum sequence length of 2048
              # 'jhgan/ko-sbert-nli', # 최대 시퀀스 길이는 128 토큰
              # 'BM-K/KoSimCSE-roberta'
    model_kwargs={'device':'mps'}, # 'cpu', 'gpu', 'cuda'
    encode_kwargs={'normalize_embeddings':True},
)

```

`Faiss` : 밀집 벡터의 효율적인 유사도 검색과 클러스터링을 위한 라이브러리  
> Facebook AI Similarity Search
```python
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

vectorstore = FAISS.from_documents(split_docs,
                                   embedding = embeddings_model,
                                   distance_strategy = DistanceStrategy.COSINE
                                  )
retriever = vectorstore.as_retriever(),
```

`GPT4All` : 언어 모델을 사용하여 프롬프트에 대한 응답을 생성하는 LLMChain을 구현  
`StreamingStdOutCallbackHandler()` : 답변을 받기 위한  콜백
```python
from langchain_community.llms import GPT4All
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

local_path = (
    "./ggufModels/EEVE-Korean-Instruct-10.8B-v1.0-Q5_K_M.gguf"
)
llm = GPT4All(
    model=local_path,
    callbacks=[StreamingStdOutCallbackHandler()],
    backend="mps", # GPU 설정
    streaming=True,
    verbose=True,
)
```

`PromptTemplate` : LLMs에 메시지를 전달하기 전에 문장 구성을 편리하게 만들어주는 기능  
`RunnablePassthrough` : 데이터를 전달하는 역할  
`StrOutputParser` : 모델의 출력을 문자열 형태로 파싱하여 최종 결과를 반환
```python
from langchain_core.prompts import PromptTemplate

template = """
당신은 영어를 가르치는 10년차 영어 선생님입니다. 상황에 [FORMAT]에 영어 회화를 작성해 주세요.

상황:
{question}

FORMAT:
- 영어 회화:

- 한글 해석:
"""

prompt = PromptTemplate.from_template(template)

chain = (
        { "context": retriever,
          "question": RunnablePassthrough()} | 
        prompt | 
        llm | 
        StrOutputParser()
            )
query = "저는 식당에 가서 음식을 주문하고 싶어요"
print(chain.invoke({"question": query}))
```





---

`RunnablePassthrough` : 데이터를 전달하는 역할
```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

runnable = RunnableParallel(
    passed=RunnablePassthrough(),    
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    modified=lambda x: x["num"] + 1,
)

runnable.invoke({"num": 1})
> {'passed': {'num': 1}, 'extra': {'num': 1, 'mult': 3}, 'modified': 2}
```


