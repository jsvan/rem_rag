import openai
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class RAGTester:
    def __init__(self, api_key, model_name="gpt-4", embed_model="text-embedding-ada-002"):
        # Initialize OpenAI API key and model names
        openai.api_key = api_key
        self.model_name = model_name
        self.embed_model = embed_model

    def load_datasets(self):
        # Load the QasperQA and NarrativeQA datasets
        self.qasper = load_dataset("allenai/qasper")  # Load QasperQA dataset
        self.narrativeqa = load_dataset("narrativeqa")  # Load NarrativeQA dataset

    def generate_embeddings(self, text):
        # Generate embeddings using OpenAI's embedding API
        response = openai.Embedding.create(model=self.embed_model, input=text)
        return np.array(response['data'][0]['embedding'])

    def retrieve_documents(self, query_embedding, context_embeddings, top_k=5):
        # Use cosine similarity to retrieve top-k documents
        similarities = np.dot(context_embeddings, query_embedding) / (
                np.linalg.norm(context_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        return top_k_indices

    def generate_answer(self, question, context):
        # Generate an answer using the OpenAI LLM
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content

    def test_on_dataset(self, dataset_name="qasper", num_samples=10):
        # Select the dataset for testing
        dataset = self.qasper['test'] if dataset_name == "qasper" else self.narrativeqa['test']
        context_embeddings = [self.generate_embeddings(doc['context']) for doc in dataset]

        # Run tests on the specified number of samples
        results = []
        for i in range(num_samples):
            sample = dataset[i]
            query_embedding = self.generate_embeddings(sample['question'])

            # Retrieve relevant documents
            top_k_indices = self.retrieve_documents(query_embedding, context_embeddings)
            retrieved_docs = [dataset[idx]['context'] for idx in top_k_indices]

            # Generate answer using retrieved context
            answer = self.generate_answer(sample['question'], ' '.join(retrieved_docs))
            results.append({"question": sample['question'], "predicted": answer, "actual": sample['answer']})

        return results

    def evaluate(self, results):
        # Evaluate the performance of the algorithm
        y_true = [result['actual'] for result in results]
        y_pred = [result['predicted'] for result in results]

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")