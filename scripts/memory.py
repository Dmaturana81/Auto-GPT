from config import Config, Singleton
from typing import Optional, Union
import pinecone
import uuid
from chromadb.utils import embedding_functions
import openai

cfg = Config()
TABLE_COLLECTION = "auto-gpt"

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=cfg.openai_api_key,
                model_name="text-embedding-ada-002"
            )



def get_ada_embedding(text):
    text = text.replace("\n", " ")
    openai.api_key = cfg.openai_api_key
    # print('Before embedding')
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    # print(response)
    return response["data"][0]["embedding"]


def get_text_from_embedding(embedding):
    return openai.Embedding.retrieve(embedding, model="text-embedding-ada-002")["data"][0]["text"]


class PineconeMemory(metaclass=Singleton):
    def __init__(self):
        pinecone_api_key = cfg.pinecone_api_key
        pinecone_region = cfg.pinecone_region
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_region)
        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        table_name = TABLE_COLLECTION
        # this assumes we don't start with memory.
        # for now this works.
        # we'll need a more complicated and robust system if we want to start with memory.
        self.vec_num = 0
        if table_name not in pinecone.list_indexes():
            pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)
        self.index = pinecone.Index(table_name)

    def add(self, data):
        vector = get_ada_embedding(data)
        # no metadata here. We may wish to change that long term.
        resp = self.index.upsert([(str(self.vec_num), vector, {"raw_text": data})])
        _text = f"Inserting data into memory at index: {self.vec_num}:\n data: {data}"
        self.vec_num += 1
        return _text

    def get(self, data):
        return self.get_relevant(data, 1)

    def clear(self):
        self.index.delete(deleteAll=True)
        return "Obliviated"

    def get_relevant(self, data, num_relevant=5):
        """
        Returns all the data in the memory that is relevant to the given data.
        :param data: The data to compare to.
        :param num_relevant: The number of relevant data to return. Defaults to 5
        """
        query_embedding = get_ada_embedding(data)
        results = self.index.query(query_embedding, top_k=num_relevant, include_metadata=True)
        sorted_results = sorted(results.matches, key=lambda x: x.score)
        return [str(item['metadata']["raw_text"]) for item in sorted_results]

    def get_stats(self):
        return self.index.describe_index_stats()


class ChromaMemory(metaclass=Singleton):
    def __init__(
        self,
        collection_name: str = TABLE_COLLECTION,
        embedding_function = openai_ef,
        persist_directory: Optional[str] = None,
    ) -> None:
        """Initialize with Chroma client."""
        try:
            import chromadb
            import chromadb.config
        except ImportError:
            raise ValueError(
                "Could not import chromadb python package. "
                "Please it install it with `pip install chromadb`."
            )
        self._client_settings = chromadb.config.Settings()
        if persist_directory is not None:
            self._client_settings = chromadb.config.Settings(
                chroma_db_impl="duckdb+parquet", persist_directory=persist_directory
            )
        self._client = chromadb.Client(self._client_settings)
        self._embedding_function = embedding_function
        self._persist_directory = persist_directory

        # Check if the collection exists, create it if not
        if collection_name in [col.name for col in self._client.list_collections()]:
            self._collection = self._client.get_collection(name=collection_name)
            # TODO: Persist the user's embedding function
            print(
                f"Collection {collection_name} already exists,"
                " Do you have the right embedding function?"
            )
        else:
            self._collection = self._client.create_collection(
                name=collection_name,
                embedding_function=self._embedding_function
                if self._embedding_function is not None
                else None,
            )
            print(
                f"Collection {collection_name} created,"
                " Do you have the right embedding function?"
            )
        self.vec_num = self._collection.count()
    # def __init__(self):
    #     self.client = chromadb.Client(Settings(
    #     chroma_db_impl="duckdb+parquet",
    #     # persist_directory="/path/to/persist/directory" # Optional, defaults to .chromadb/ in the current directory
    #         ))
    #     dimension = 1536
    #     table_name = "auto-gpt"
    #     # this assumes we don't start with memory.
    #     # for now this works.
    #     # we'll need a more complicated and robust system if we want to start with memory.
    #     self.vec_num = 0
    #     if table_name not in pinecone.list_indexes():
    #         pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)
    #     self.index = pinecone.Index(table_name)

    def add(self, data):
        vector =get_ada_embedding(data)
        # no metadata here. We may wish to change that long term.
        id = str(uuid.uuid1())
        self._collection.add(
             embeddings=[vector], documents=[data], ids=[id]
        )
        _text = f"Inserting data into memory at index: {self.vec_num}:\n data: {data}"
        self.vec_num += 1
        return _text

    def get(self, data):
        return self.get_relevant(data, 1)

    def clear(self):
        self._client.reset()
        return "Obliviated"

    def get_relevant(self, data, num_relevant=5):
        """
        Returns all the data in the memory that is relevant to the given data.
        :param data: The data to compare to.
        :param num_relevant: The number of relevant data to return. Defaults to 5
        """
        query_embedding = get_ada_embedding(data)
        k = num_relevant if num_relevant <= self._collection.count() else self._collection.count() 
        results = self._collection.query(query_embeddings=[query_embedding], n_results=k, where=None, include=['documents']
    ) if self.vec_num > 0 else {'documents':[]}
        print(f"THIS ARE THE RELEVAN MEMORY ####################\n{results['documents']}")
        # sorted_results = sorted(results.matches, key=lambda x: x.score)
        return results['documents']

    def get_stats(self):
        return self._collection.count()
