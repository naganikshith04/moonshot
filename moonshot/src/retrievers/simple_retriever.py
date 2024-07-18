from moonshot.src.connectors.connector import Connector
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

# to take in Prompt Arguments
import importlib

from moonshot.src.retrievers.retriever import Retriever

class SimpleRetriever(Retriever):
    def __init__(self) -> None:
        pass
    
    def retrieve_context(self, prompt):

        query = prompt.connector_prompt.prompt

        PROMPT_TEMPLATE = """
                        Answer the question based only on the following context (from retriever module):
                        Answer the question based only on the following context (from retriever module):

                        {context}

                        ---

                        Answer the question based on the above context: {question}
                        """
        
        # default embedding models used: "text-embedding-ada-002"
        embedding_function = OpenAIEmbeddings(openai_api_key="")
        db = Chroma(persist_directory="chroma", embedding_function=embedding_function)
        results = db.similarity_search_with_relevance_scores(query, k=3)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        edited_prompt = prompt_template.format(context=context_text, question=query)

        prompt.connector_prompt.prompt = edited_prompt

        return prompt