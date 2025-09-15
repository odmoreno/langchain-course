
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def main():
    print("Ingesting ...")
    loader = TextLoader("mediumblog1.txt", encoding="utf-8")
    document = loader.load()
    print(f"Document length: {len(document)}")

    print("splitting ... ")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("Ingesting ...")
    PineconeVectorStore.from_documents(
        texts,
        embedding=embeddings,
        index_name=os.environ['INDEX_NAME']
    )
    print("Finish ...")

if __name__ == "__main__":
    main()
