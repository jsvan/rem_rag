from llmtest_base import RAGTester


tester = RAGTester(api_key="YOUR_OPENAI_API_KEY")
tester.load_datasets()
results = tester.test_on_dataset("qasper", num_samples=10)  # Test on 10 samples from QasperQA
tester.evaluate(results)