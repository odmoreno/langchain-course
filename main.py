import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub


load_dotenv()


def main():
    print("Hello from langchain-course!")

    # 2) Carga de PDF + split
    pdf_path = 'react.pdf'
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    # 3) Embeddings con Google (text-embedding-004)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # o "models/embedding-001"
        output_dimensionality=768,  # alinea con tu índice
    )

    # 4) Index local con FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    # (opcional) Recargar el índice
    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    # 5) LLM Gemini (chat) + prompt de LangChain Hub
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # 6) Cadena de combinación y retrieval
    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    # 7) Consulta
    res = retrieval_chain.invoke(
        {"input": "Give me the gist of ReAct in 3 sentences"})
    print(res["answer"])


if __name__ == "__main__":
    main()
