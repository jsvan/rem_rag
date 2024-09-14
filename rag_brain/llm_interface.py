from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
import utils



"""
Interfaces with OpenAI's LLM for text generation

"""
class LLM:
    def __init__(self, model="gpt-4o-mini", quantization=None, matroshka=None, similarity="dot"):
        with open("../data/openai_key.txt") as F:
            self.key = F.read().strip()
        self.model = model
        with open("../data/openai_key.txt", 'r') as f:
            apikey = f.read().strip()
        self.client = OpenAI(api_key=apikey)  # os.environ.get("OPENAI_API_KEY"),

    def generate_one(self, input_str):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "user",
                 "content": input_str}
            ],
            model=self.model,
        )
        return chat_completion.choices[0].message

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
    def _generate_minibatch(self, input_list):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "user",
                 "content": input_str}
                for input_str in input_list  # max size 4 (ie, 8,000 tokens == 8 paragraphs, but giving lots of space)
            ],
            model=self.model,
        )
        return [x.message for x in chat_completion.choices]

    def generate_list(self, input_list):
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self._generate_minibatch, minibatch)
                for minibatch in tqdm(utils.grouper(input_list, 4))
            ]

        print("Done submitting futures!")
        generated_text = []
        for future in tqdm(futures, total=len(futures), desc="Building embeddings"):
            generated_text += future.result()
        return