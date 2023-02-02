from transformers import (
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
)
from datasets import load_from_disk


dataset = load_from_disk("./my_knowledge_dataset/my_knowledge_dataset")
dataset.load_faiss_index(
    "embeddings", "./my_knowledge_dataset/my_knowledge_dataset_hnsw_index.faiss"
)
# dataset.add_faiss_index("embeddings", custom_index=index)

retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq", index_name="custom", indexed_dataset=dataset
)
model = RagSequenceForGeneration.from_pretrained(
    "facebook/rag-token-base", retriever=retriever
)
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")

question = "What is bitcoin?"
input_ids = tokenizer.question_encoder(question, return_tensors="pt")["input_ids"]
generated = model.generate(input_ids, max_length=512)

generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
print(f"{generated_string=}")
