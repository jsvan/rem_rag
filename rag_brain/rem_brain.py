
"""
REM Cycle-- Random Emergent Memory (Cycle)

This code creates random combinations of facts and memories and reencodes each dream in the vector store.
The point is to approach a complete unification of analysis of how everything it knows fits together in a cohesive whole.
It's possible to always include the previous created analysis, before combining with new elements, as a bit of a thread of thought.
How many vectors are included in the RAG dream is a hyper parameter.

"""
import utils
from preprocess import FA_data
from tqdm import tqdm

DREAM_PROMPT = "What is the implicit question at the heart which binds the following passages together? Reply with the question only."
SUMMARIZATION_PROMPT = "Summarize the following passages. Write a single unified answer, about 100 words long, and do not enumerate over each document in the context, or add your own analysis. Do not refer to \"the texts\", or \"the passages\", simply combine and distill their information together with as many details as possible, and a focus on the insights and wisdom contained within. Mention the years at play in your answer.\n\n"


class Dreamer:

    def __init__(self, vectorstore, llm):
        self.store = vectorstore
        self.llm = llm

    def rem_cycle(self, neighbors=8, num_dreams=1):
        random_duos = self.store.sample(2 * num_dreams)['documents']
        duo_question_prompts = [f"{DREAM_PROMPT}\n\n{item_a}\n\n{item_b}" for item_a, item_b in utils.grouper(random_duos, 2)]
        questions = self.llm.generate_from_list(duo_question_prompts)
        duo_question_prompts = None  # Permit the garbage collector
        associated_ideas = self.store.nearest(questions, neighbors)['documents']  # maybe [0] too
        associated_ideas = [SUMMARIZATION_PROMPT + '\n\n'.join(duos + idealist) for duos, idealist in zip(random_duos, associated_ideas)]
        summarizations = self.llm.generate_from_list(associated_ideas)
        self.store.add(summarizations)
        return None


    """
    
    Chunk, embed paragraphs.
    
    OpenAI analyze article for topics, entities, then draw conclusions for each listed.
    Embed the conclusions. 
    
    Years end:
        KNN on year's topics, generate, summarize. 
        REM Cycle, generate, summarize. 
    
    
    """
    def process_documents(self):
        documents = FA_data.get( article=True, embedding=True, paragraph=True, article_id=True, year=True, url=True, sort_on='paragraph_year')
        curryear = documents['paragraph_year'][0]
        currarticle = documents['article_id'][0]
        years_named_entities = set()

        for article, embedding, year, url in tqdm(utils.dict_zipper(documents)):
            if year != curryear:
                # process last year's works
                self.years_end()
                curryear = year




    def years_end(self, years_named_entities):
        # First we will do clustering analysis for all recent clusters
        # Next we will do REM
        pass


