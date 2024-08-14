# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 20:35:42 2024

@author: BridgeAiHub
"""
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb
from chromadb.config import Settings

from tempfile import NamedTemporaryFile
from typing import List

import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document, StrOutputParser
from langchain.chains import LLMChain, RetrievalQAWithSourcesChain
from langchain.schema.embeddings import Embeddings

from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.openai import OpenAIEmbeddings

def process_file(*, file:AskFileResponse) -> List[Document]:
   if file.type != "application/pdf":
      raise TypeError("Only PDF files are supported")
   
   with NamedTemporaryFile() as tempfile:
      tempfile.write(file.content)
      loader = PDFPlumberLoader(tempfile.name)
      documents = loader.load()

   text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 1000,
      chunk_overlap=100
   )

   docs = text_splitter.split_documents(documents)

   for i, doc in enumerate(docs):
      doc.metadata["source"] =  f"source_{i}"
   if not docs:
      raise ValueError("pdf file parsing failed")
   return docs

def create_search_engine(*,docs: List[Document], embeddings: Embeddings) -> VectorStore:
   client = chromadb.EphemeralClient()
   client_settings = Settings(allow_reset=True, anonymized_telemetry=False)
   search_engine = Chroma(client=client, client_settings=client_settings)
   search_engine._client.reset()

   search_engine = Chroma.from_documents(
      client=client,
      documents=docs,
      embedding=embeddings,
      client_settings=client_settings
   )

   return search_engine

@cl.on_chat_start
async def on_chat_start():
  print("Chat started, initializing chain...")

  files = None
  while files is None:
     files = await cl.AskFileMessage(
        content="Please upload your PDF",
        accept=['application/pdf'],
        max_size_mb=5,
     ).send()
  file = files[0]

  msg = cl.Message(content=f"processing {file.name}")
  await msg.send()
  docs = process_file(file=file)
  cl.user_session.set("docs", docs)
  msg.content = f"`{file.name}` processed. Loading ..."
  await msg.update()

  embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

  try:
   search_engine = await cl.make_async(create_search_engine)(
         docs=docs, embeddings=embeddings
      )
  except Exception as e:
   await cl.Message(content=f"Error: {e}").send()
   raise SystemError
  msg.content = f"`{file.name}` loaded. You can now ask questions!"
  await msg.update()

  model = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0, 
    streaming=True
  )
#   prompt = ChatPromptTemplate.from_messages(
#      [("system", "you are a chainlit expert"),
#       ("human", "{question}")
#      ]
#      )

#   chain = LLMChain(
#     llm=model,
#     prompt=prompt,
#     output_parser=StrOutputParser()
#   )
  chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=search_engine.as_retriever(max_tokens_limit=4097),
    )
  cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):

    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain

    response = await chain.acall(
        message.content,
        callbacks=[cl.AsyncLangchainCallbackHandler(stream_final_answer=True)]
    )
    answer = response["answer"]
    sources = response["sources"].strip()

    # Get all of the documents from user session
    docs = cl.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    source_elements = []
    if sources:
        found_sources = []
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=source_elements).send()