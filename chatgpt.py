import os
import sys

import openai
openai.proxy = {
            "http": "http://127.0.0.1:7890",
            "https": "http://127.0.0.1:7890"
        }

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
# from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
# from langchain.llms import OpenAI
from langchain_openai.llms import OpenAI
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma


# Check if constants.py exists in the current working directory
if not os.path.exists("constants.py"):
    raise FileNotFoundError("constants.py not found in the current directory")

import importlib.util
# Load constants.py from the current working directory
spec = importlib.util.spec_from_file_location("constants", "constants.py")
constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constants)

os.environ["OPENAI_API_KEY"] = constants.APIKEY

from oliver_framework.utils.logging import getlogger
logger=getlogger("chatgpt")

model="gpt-3.5-turbo"

from openai import OpenAI
# client = OpenAI()
import httpx
client = OpenAI(timeout=httpx.Timeout(15.0, read=5.0, write=10.0, connect=3.0))

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
  loader = DirectoryLoader("data/")
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:
    index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

# Modify this part to include a direct call to the GPT model
def is_uncertain(answer):
    # Define phrases that indicate uncertainty
    uncertain_phrases = ["i don't know", "not sure", "unsure", "maybe", "don't have that information"]
    # Check if the answer contains any of the uncertain phrases
    return any(phrase in answer.lower() for phrase in uncertain_phrases)

def get_openai_response(prompt, model="gpt-3.5-turbo"):  # Adjust model as needed
    response = client.chat.completions.create(model=model,
    messages=[{"role": "user", "content": prompt}])
    # Assuming the response format aligns with the chat model's response structure
    return response.choices[0].message.content.strip()

def query_both_sources(query, chat_history):
    # Query the custom data via the retriever first to check for an uncertain response
    logger.debug("Going to custom data response")
    custom_data_response = chain({"question": query, "chat_history": chat_history, "timeout": 10})

    # If the custom data's answer is uncertain, then query the GPT model
    if 'answer' in custom_data_response and is_uncertain(custom_data_response['answer']):
        logger.debug("Custom data response is uncertain, going to OpenAI response")
        # Only now do we query the GPT model because the custom data was uncertain
        direct_response = get_openai_response(query, model=model)
        return direct_response
    else:
        # If the custom data's answer is confident enough, use it directly
        logger.debug("Using custom data response.")
        return custom_data_response.get('answer', 'No answer found.')

chat_history = []
while True:
    if not query:
        sys.exit()
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()

    # Update to use the new function that queries both sources
    result_answer = query_both_sources(query, chat_history)
    print(result_answer)

    chat_history.append((query, result_answer))
    query = None
