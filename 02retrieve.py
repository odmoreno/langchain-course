import os

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough

from adapters import GoogleEmbeddingsSafe  # <--- nuestro wrapper


load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    print("Retrieving ...")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    query = "what is Pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    print(result.content)

    # Embeddings de Google (no pases la key; usa GOOGLE_API_KEY en el entorno).
    inner = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # o "models/embedding-001"
        output_dimensionality=768,  # alinea con tu índice
    )

    # ¡Envuelve para que siempre devuelva list(...)!
    embeddings = GoogleEmbeddingsSafe(inner)

    vectorstore = PineconeVectorStore(
        embedding=embeddings,
        index_name=os.environ["INDEX_NAME"],
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt)

    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke(input={"input": query})

    print(result)

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | custom_rag_prompt
        | llm
    )

    res = rag_chain.invoke(query)
    print(res)


if __name__ == "__main__":
    main()
