import json
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from torch.utils.data import DataLoader

# Example usage of the function retrieve_docs
def retrieve_docs(query, db):
    # Dummy implementation of document retrieval. Replace with actual implementation.
    results = similarity_search(query, db)
    return results[0][0].page_content

def similarity_search(query, db):
    search_results = db.similarity_search_with_score(query)
    return search_results

def augment_question_with_context(input_file):
    # Initialize the embeddings model
    # Define the path to the pre-trained model you want to use
    modelPath = "documents/MiniLM-L6-v2"
    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device':'cuda'}
    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}
    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )

    db = FAISS.load_local("documents/faiss_index_updated", embeddings, allow_dangerous_deserialization=True)

    '''with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    '''

    # Step 1: Access the dataset from the DataLoader
    test_dataset = input_file.dataset

    for item in test_dataset:
        query = item['question']
        retrieved_docs = retrieve_docs(query, db)
        augmented_query = " context: " + retrieved_docs + query
        item['question'] = augmented_query

    # Step 3: Create a new DataLoader with the modified dataset
    modified_test_dataloader = DataLoader(test_dataset, batch_size=8)

    return modified_test_dataloader
