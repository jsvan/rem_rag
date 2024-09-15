import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import time
import random
from tqdm import tqdm


"""
RAPTOR RAG uses heirarchical clusterization + summarization. These new embedded points are often chosen for question answering,
and provide higher quality material for the text generation. 

What I try to do:
    1) Fill up vector space with cluster summarizations WITHOUT clustering algorithms
    2) I generate questions randomly (monte carlo), and take the KNN of that question, which I then use as a "cluster"
    3) I fill in more vector space by treating every new document as a cluster center, getting summarization for every 
        document I add.

The intent is to mimic how the human brain works -- simultaneous reading and comprehension, and a sleep cycle connecting 
the dots in more unstructured reflection. 

I call it REM Rag (Random Emergent Memories RAG)
Could also be called REMCycle (Reflection Embedded Monte Carlo RAG)



1) Sort documents by year, article ID

READING cycle:
    Every time a document is added:
    
    1) Grab nearest neighbors of doc AND prev. short term memory
    2) As context, generate:
        "What new information does this document provide, and how can I reconcile this new information with what the given context?"
    3) Set Short Term Memory to string of doc + 2*
    4) Save Doc in database
    5) Save 2* in database
    5) if article id number of doc changes, clear short term memory == ""


REM cycle (trigger every year change):
    1) Sample TWO (2) points
    2) Ask: "What is the implicit question at the heart which binds the following passages together? Reply with the question only."
    3) Get nearest neighbors to 2* and generate summary:
            "Summarize the following passages. Write a single unified answer, about 100 words long, and do not enumerate
             over each document in the context, or add your own analysis. Do not refer to "the texts", or "the passages",
             simply combine and distill their information together with as many details as possible."
        or
            "Summarize the following passages. Write a single unified answer, about 100 words long, and do not enumerate
             over each document in the context, or add your own analysis. Do not refer to "the texts", or "the passages",
               simply combine and distill their information together with as many details as possible, and a focus on 
               the insights and wisdom contained within. Mention the years at play in your answer.
            
    4) embed
    

Let's REM cycle 1/10 times, of how many item's we've just read

Every year:
    Loop over years_documents:
        Embed Doc
        Compare Doc to relevant knowledge, summarize
    for range(total_docs_read // 100):
        REM_cycle
            Embed
            Sample
            Generate
            Sample
            Generate
            Embed


if every year I dream for DB.count // 100 (ie, 1% of whats there), I will accumulate about 56k dreams, ie 46%. But 
actually more, because it would include the 1% of previous dreamings too. 

"""

class Store:

    def __init__(self, name="chroma_db_1"):
        with open("./data/openai_key.txt") as f:
            apikey = f.read().strip()
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=apikey,
            model_name="text-embedding-3-small"
        )

        self.chroma_client = chromadb.PersistentClient(path="./data")
        self.collection = self.chroma_client.get_or_create_collection(name=name,
                                                                      embedding_function=openai_ef,
                                                                      metadata={"hnsw:space": "cosine"})
        self.count = self.collection.count()


    """
    returns n nearest neighbors to a str or list of strs.
    I think it may be nice to choose from n*2 NN, from which we randomly sample n items. 
    """
    def nearest(self, texts, n, typ=None):
        w = {} if typ in [None, "all"] else {"question": typ}
        match texts:
            case list():
                return self.collection.query(query_texts=texts,
                                             n_results=n,
                                             where=w)
            case str():
                return self.collection.query(query_texts=[texts],
                                             n_results=n,
                                             where=w)
            case _:
                raise Exception(f"Wrong instance of input for 'texts'. Must be list or str, you entered {type(texts)}.")



    """
    Returns n random vectors 
    """
    def sample(self, neighbors):
        # Fetch all vectors and their corresponding metadata (e.g., IDs)
        ids = random.sample(range(self.count), neighbors)
        return self.collection.get(
            ids=[str(x) for x in ids],
            #where={"style": "style1"},
            #include=["ids", "documents", "metadatas", "distances"]
        )


    def add(self, texts, typ):
        if typ not in ['ground_truth', 'question', 'criticism', 'thought']:
            raise Exception(f"add() takes typ as 'ground_truth', 'question', 'criticism', or 'thought'. Received {typ},")
        if not isinstance(texts, list):
            raise Exception(f"add() takes a type list as input, received a {type(list)}.")
        if len(texts) == 0:
            return
        try:
            ids = [str(hash(t)) for t in texts]
            self.collection.add(
                documents=texts,
                ids=ids,
                # Bug maybe: This is a single dictionary, pointed to n times. I think it should be okay
                # because I expect it to be converted into a json string n times.
                metadatas=[ {"type": typ,
                             "time": int(time.time())
                             }
                          ] * len(texts)
            )
        except Exception as e:
            print(e, e.__doc__)


    def read(self, data):
        ids = [str(x) for x in range(self.count, self.count + len(data['chunk']))]
        documents = data['chunk']
        embeddings = data['embedding']
        jump = 1000
        for i in tqdm(range(0, len(ids), jump)):
            self.collection.add(
                ids=ids[i:i + jump],
                documents=documents[i:i + jump],
                embeddings=embeddings[i:i + jump],
            )
        self.count = self.collection.count()

