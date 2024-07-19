import gradio as gr
import json
from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_core.documents import Document
from typing import Iterator, List, Dict
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from umap import UMAP
import numpy as np
import tempfile
from collections import defaultdict

# 1. Data Loading
class CustomArxivLoader(ArxivLoader):
    def lazy_load(self) -> Iterator[Document]:
        documents = super().lazy_load()
        for document in documents:
            yield Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "ArxivId": self.query,
                    "Source": f"https://arxiv.org/pdf/{self.query}.pdf"
                }
            )

def load_documents_from_file(file_path: str) -> List[Document]:
    with open(file_path, "r") as f:
        results = json.load(f)
    
    arxiv_urls = results["collected_urls"]["arxiv.org"]
    arxiv_ids = [url.split("/")[-1].strip(".pdf") for url in arxiv_urls]
    
    loaders = [CustomArxivLoader(query=arxiv_id) for arxiv_id in arxiv_ids]
    merged_loader = MergedDataLoader(loaders=loaders)
    
    return merged_loader.load()

# 2. Topic Modeling
def create_topic_model(umap_params: Dict, bertopic_params: Dict) -> BERTopic:
    umap_model = UMAP(**umap_params)
    representation_model = KeyBERTInspired()
    
    return BERTopic(
        language="english",
        verbose=True,
        umap_model=umap_model,
        representation_model=representation_model,
        **bertopic_params
    )

def process_documents(documents: List[Document], topic_model: BERTopic) -> tuple:
    contents = [doc.page_content for doc in documents]
    topics, _ = topic_model.fit_transform(contents)
    topic_labels = topic_model.generate_topic_labels(nr_words=3, topic_prefix=False, separator=' ')
    
    return topics, topic_labels

# 3. Data Manipulation
def create_docs_matrix(documents: List[Document], topics: List[int], labels: List[str]) -> List[List[str]]:
    return [
        [str(i), labels[topic], doc.metadata['Title']]
        for i, (doc, topic) in enumerate(zip(documents, topics))
    ]

def get_unique_topics(labels: List[str]) -> List[str]:
    return sorted(set(labels))

def remove_topics(state: Dict, topics_to_remove: List[str]) -> Dict:
    documents, topics, labels = state['documents'], state['topics'], state['labels']
    filtered_data = [
        (doc, topic, label)
        for doc, topic, label in zip(documents, topics, labels)
        if label not in topics_to_remove
    ]
    new_documents, new_topics, new_labels = map(list, zip(*filtered_data)) if filtered_data else ([], [], [])
    return {**state, 'documents': new_documents, 'topics': new_topics, 'labels': new_labels}

# 4. Output Generation
def create_markdown_content(state: Dict) -> str:
    documents, topics, labels = state['documents'], state['topics'], state['labels']
    if not documents or not labels:
        return "No data available for download."

    topic_documents = defaultdict(list)
    for doc, topic in zip(documents, topics):
        label = labels[topic]
        topic_documents[label].append(doc)

    content = ["# Arxiv Articles by Topic\n"]
    for topic, docs in topic_documents.items():
        content.append(f"## {topic}\n")
        for document in docs:
            content.append(f"### {document.metadata['Title']}")
            content.append(f"{document.metadata['Summary']}")  

    return "\n".join(content)

# 5. Gradio Interface
def create_gradio_interface():
    with gr.Blocks(theme="default") as demo:
        gr.Markdown("# BERT Topic Article Organizer App")
        gr.Markdown("Organizes arxiv articles in different topics and exports it in a zip file.")

        state = gr.State(value={})

        with gr.Row():
            file_uploader = gr.UploadButton("Click to upload", file_types=["json"], file_count="single")
            reprocess_button = gr.Button("Reprocess Documents")
            download_button = gr.Button("Download Results")

        with gr.Row():
            with gr.Column():
                umap_n_neighbors = gr.Slider(minimum=2, maximum=100, value=15, step=1, label="UMAP n_neighbors")
                umap_n_components = gr.Slider(minimum=2, maximum=100, value=5, step=1, label="UMAP n_components")
                umap_min_dist = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.01, label="UMAP min_dist")
            with gr.Column():
                min_topic_size = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="BERTopic min_topic_size")
                nr_topics = gr.Slider(minimum=1, maximum=100, value="auto", step=1, label="BERTopic nr_topics")
                top_n_words = gr.Slider(minimum=5, maximum=50, value=10, step=1, label="BERTopic top_n_words")
                n_gram_range = gr.Slider(minimum=1, maximum=3, value=1, step=1, label="BERTopic n_gram_range")
                calculate_probabilities = gr.Checkbox(label="Calculate Probabilities", value=False)

        output_matrix = gr.DataFrame(
            label="Processing Result",
            headers=["ID", "Topic", "Title"],
            col_count=(3, "fixed"),
            interactive=False
        )

        with gr.Row():
            topic_dropdown = gr.Dropdown(label="Select Topics to Remove", multiselect=True, interactive=True)
            remove_topics_button = gr.Button("Remove Selected Topics")

        markdown_output = gr.File(label="Download Markdown")

        def update_ui(state: Dict):
            matrix = create_docs_matrix(state['documents'], state['topics'], state['labels'])
            unique_topics = get_unique_topics(state['labels'])
            return matrix, gr.Dropdown(choices=unique_topics, value=[]), unique_topics

        def process_and_update(state: Dict, umap_n_neighbors: int, umap_n_components: int, umap_min_dist: float, 
                               min_topic_size: int, nr_topics: int, top_n_words: int, n_gram_range: int, 
                               calculate_probabilities: bool):
            documents = state.get('documents', [])
            umap_params = {
                "n_neighbors": umap_n_neighbors, 
                "n_components": umap_n_components, 
                "min_dist": umap_min_dist
            }
            bertopic_params = {
                "min_topic_size": min_topic_size, 
                "nr_topics": nr_topics,
                "top_n_words": top_n_words,
                "n_gram_range": (1, n_gram_range),
                "calculate_probabilities": calculate_probabilities
            }
            
            topic_model = create_topic_model(umap_params, bertopic_params)
            topics, labels = process_documents(documents, topic_model)
            
            new_state = {**state, 'documents': documents, 'topics': topics, 'labels': labels}
            matrix, dropdown, unique_topics = update_ui(new_state)
            return new_state, matrix, dropdown, unique_topics

        def load_and_process(file, umap_n_neighbors, umap_n_components, umap_min_dist, 
                             min_topic_size, nr_topics, top_n_words, n_gram_range, calculate_probabilities):
            documents = load_documents_from_file(file.name)
            state = {'documents': documents}
            return process_and_update(state, umap_n_neighbors, umap_n_components, umap_min_dist, 
                                      min_topic_size, nr_topics, top_n_words, n_gram_range, calculate_probabilities)

        file_uploader.upload(
            fn=load_and_process,
            inputs=[file_uploader, umap_n_neighbors, umap_n_components, umap_min_dist, 
                    min_topic_size, nr_topics, top_n_words, n_gram_range, calculate_probabilities],
            outputs=[state, output_matrix, topic_dropdown, topic_dropdown]
        )

        reprocess_button.click(
            fn=process_and_update,
            inputs=[state, umap_n_neighbors, umap_n_components, umap_min_dist, 
                    min_topic_size, nr_topics, top_n_words, n_gram_range, calculate_probabilities],
            outputs=[state, output_matrix, topic_dropdown, topic_dropdown]
        )

        def remove_and_update(state: Dict, topics_to_remove: List[str], umap_n_neighbors: int, umap_n_components: int, 
                              umap_min_dist: float, min_topic_size: int, nr_topics: int, top_n_words: int, 
                              n_gram_range: int, calculate_probabilities: bool):
            new_state = remove_topics(state, topics_to_remove)
            return process_and_update(new_state, umap_n_neighbors, umap_n_components, umap_min_dist, 
                                      min_topic_size, nr_topics, top_n_words, n_gram_range, calculate_probabilities)

        remove_topics_button.click(
            fn=remove_and_update,
            inputs=[state, topic_dropdown, umap_n_neighbors, umap_n_components, umap_min_dist, 
                    min_topic_size, nr_topics, top_n_words, n_gram_range, calculate_probabilities],
            outputs=[state, output_matrix, topic_dropdown, topic_dropdown]
        )

        def create_download_file(state: Dict):
            content = create_markdown_content(state)
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as temp_file:
                temp_file.write(content)
            return temp_file.name

        download_button.click(
            fn=create_download_file,
            inputs=[state],
            outputs=[markdown_output]
        )

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True, show_error=True, max_threads=10, debug=True)