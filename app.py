import os
import sys
import logging
import nest_asyncio
import pandas as pd
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors.llm_selectors import LLMSingleSelector
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.llms.bedrock import Bedrock

# Initialize the Bedrock model with Anthropic Claude-v2
llm = Bedrock()

# Initialize async environment
nest_asyncio.apply()

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)



# Initialize Anthropic and HuggingFace models
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 2000

def run_query(pdf_file_path, query):
    # Load documents from the PDF file
    documents = SimpleDirectoryReader(pdf_file_path).load_data()

    # Create vector index
    vector_index = VectorStoreIndex.from_documents(documents)

    # Create query engine
    vector_query_engine = vector_index.as_query_engine()
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            QueryEngineTool.from_defaults(
                query_engine=vector_query_engine,
                description="Useful for retrieving specific context related to the PDF."
            )
        ]
    )

    # Run the query
    response = query_engine.query(query)
    

    return response.response

# Example usage
if __name__ == "__main__":
    pdf_file_path = "pdfdata"
    query = "List all possible side effects"
    result = run_query(pdf_file_path, query)
    print(result)
