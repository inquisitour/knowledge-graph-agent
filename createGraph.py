from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import NLTKTextSplitter
import nltk

nltk.download('punkt')

username = "give-username"
password = "give-pwd"

#initiates neo4j graph instance
graph = Neo4jGraph(url="bolt://localhost:7687", username=username, password=password)

openai_api_key = "API-KEY
llm = ChatOpenAI(api_key=openai_api_key, model='gpt-3.5-turbo')
llm_transformer = LLMGraphTransformer(llm=llm)

embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-large")

#create chunks of the document using langchain
text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap = 20) #better chunk size could be determined


with open('output.txt', 'r', encoding='utf-8') as file:
    output = file.read()

chunks = text_splitter.split_text(output)
# with open('final.txt', 'w', encoding='utf-8') as file:
#     for item in chunks:
#         file.write(item +'\n')
# documents = []
documents = [Document(page_content=chunk) for chunk in chunks] #Comment this if creating KG

#Uncomment this to create knowledge graph
# for data in chunks:
#     documents = [Document(page_content=data )]
#     graph_documents = llm_transformer.convert_to_graph_documents(documents)

#     print(f"Nodes:{graph_documents[0].nodes}\n")
#     print(f"Relationships:{graph_documents[0].relationships}")
#     #graph.add_graph_documents(graph_documents)

#Code for querying results -> work in progress

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = text_splitter.split_documents(documents)#Splitting the document into further chunks, not really needed
db = Neo4jVector.from_documents(docs, embeddings, url="bolt://localhost:7687", username=username, password=password)

#Querying
query = "What should I do if I have cataract?"
#knn
results = db.similarity_search(query, k=2)
print("THIS IS THE ANSWER:\n",results[0].page_content)
