import os
# llama-parse is async-first, running the async code in a notebook requires the use of nest_asyncio
import nest_asyncio

nest_asyncio.apply()
# API access to llama-cloud
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-i9NkjSBiUTwYAfo0G6UvrdAWp7ZkFJ84v0kaUUUWK6suTHvw"

# Using OpenAI API for embeddings/llms
os.environ["GEN_API_KEY"] = "pak-VcI5V05aaf42FbLqO7Fc6FR7KFSz2Gmap97DO1y3-I8"

from genai import Client, Credentials
from genai.extensions.langchain import LangChainEmbeddingsInterface
from genai.schema import TextEmbeddingParameters
from genai.extensions.langchain.chat_llm import LangChainChatInterface
from genai.extensions.langchain import LangChainInterface
from genai.schema import (
    DecodingMethod,
    ModerationHAP,
    ModerationParameters,
    TextGenerationParameters,
    TextGenerationReturnOptions,
)
from llama_index.llms.langchain import LangChainLLM
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_parse import LlamaParse
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.query_engine import RetrieverQueryEngine

from MarkdownElementNodeParser import MarkdownElementNodeParser

credentials = Credentials(api_key=os.environ["GEN_API_KEY"],
                          api_endpoint='https://bam-api.res.ibm.com/v2/text/chat?version=2024-01-10')
client = Client(credentials=credentials)

llm = LangChainChatInterface(
    model_id="meta-llama/llama-2-70b-chat",
    client=client,
    parameters=TextGenerationParameters(
        decoding_method=DecodingMethod.SAMPLE,
        max_new_tokens=200,
        min_new_tokens=10,
        temperature=0.5,
        top_k=50,
        top_p=1,
        return_options=TextGenerationReturnOptions(input_text=False, input_tokens=True),
    ),
    moderations=ModerationParameters(
        # Threshold is set to very low level to flag everything (testing purposes)
        # or set to True to enable HAP with default settings
        hap=ModerationHAP(input=True, output=False, threshold=0.01)
    ),
)

credentials = Credentials(api_key=os.environ["GEN_API_KEY"])
client = Client(credentials=credentials)
embeddings = LangChainEmbeddingsInterface(
    client=client,
    model_id="sentence-transformers/all-minilm-l6-v2",
    parameters=TextEmbeddingParameters(truncate_input_tokens=True),
)

llm = LangChainLLM(llm=llm)
embed_model = LangchainEmbedding(embeddings)

Settings.llm = llm
Settings.embed_model = embeddings

documents = LlamaParse(result_type="markdown").load_data('./NVIDIA Earnings.pdf')

node_parser = MarkdownElementNodeParser(llm=None, num_workers=8)

nodes = node_parser.get_nodes_from_documents(documents)

base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
index_with_obj = VectorStoreIndex(nodes=base_nodes + objects)

raw_index = VectorStoreIndex.from_documents(documents)

base_nodes, node_mappings = node_parser.get_base_nodes_and_mappings(nodes)

index = VectorStoreIndex(nodes=base_nodes)
index_ret = index.as_retriever(top_k=15)

recursive_index = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": index_ret},
    node_dict=node_mappings,
    verbose=False,
)

reranker = FlagEmbeddingReranker(
    top_n=5,
    model="BAAI/bge-reranker-large",
)

recursive_query_engine = RetrieverQueryEngine.from_args(recursive_index,
                                                        node_postprocessors=[reranker], verbose=False)

query = "What is the gross carrying amount of Total Amortizable Intangible Assets for Jan 29, 2023?"

response = recursive_query_engine.query(query)

print(response)
