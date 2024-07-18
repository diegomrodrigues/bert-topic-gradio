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
from collections import defaultdict

class CustomArxivLoader(ArxivLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lazy_load(self) -> Iterator[Document]:
        documents = super().lazy_load()

        def update_metadata(documents):
            for document in documents:
                yield Document(
                    page_content=document.page_content,
                    metadata={
                        **document.metadata,
                        "ArxivId": self.query,
                        "Source": f"https://arxiv.org/pdf/{self.query}.pdf"
                    }
                )

        return update_metadata(documents)

def upload_file(file):
    if not ".json" in file.name:
        return "Not Allowed"

    print(f"Processing file: {file.name}")

    with open(file.name, "r") as f:
        results = json.load(f)

    arxiv_urls = results["collected_urls"]["arxiv.org"]

    print(f"Collected {len(arxiv_urls)} arxiv urls from file.")

    arxiv_ids = map(lambda url: url.split("/")[-1].strip(".pdf"), arxiv_urls)

    all_loaders = [CustomArxivLoader(query=arxiv_id) for arxiv_id in arxiv_ids]

    merged_loader = MergedDataLoader(loaders=all_loaders)

    documents = merged_loader.load()

    print(f"Loaded {len(documents)} documents from file.")

    return documents

def process_documents(documents, umap_n_neighbors, umap_n_components, umap_min_dist, min_topic_size, nr_topics):
    if not documents:
        return "No documents to process. Please upload a file first."
    
    contents = [doc.page_content for doc in documents]

    representation_model = KeyBERTInspired()

    umap_model = UMAP(
        n_neighbors=umap_n_neighbors, 
        n_components=umap_n_components, 
        min_dist=umap_min_dist, 
        metric='cosine'
    )

    topic_model = BERTopic(
        language="english",
        verbose=True,
        umap_model=umap_model,
        min_topic_size=min_topic_size,
        representation_model=representation_model,
        nr_topics=nr_topics
    )

    topics, _ = topic_model.fit_transform(contents)

    topic_labels = topic_model.generate_topic_labels(nr_words=3, topic_prefix=False, separator=' ')

    print(f"Generated {len(topic_labels)} topics from data.")
    print("Topic Labels: ", topic_labels)

    return documents, topics.tolist() if isinstance(topics, np.ndarray) else topics, topic_labels

def create_docs_matrix(documents: List[Document], topics: List[int], labels: List[str]) -> List[List[str]]:
    if not documents:
        return []
    results = []
    for i, (doc, topic) in enumerate(zip(documents, topics)):
        label = labels[topic]
        results.append([str(i), label, doc.metadata['Title']])
    return results

def get_unique_topics(labels: List[str]) -> List[str]:
    return list(set(labels))

def remove_topics(documents: List[Document], topics: List[int], labels: List[str], topics_to_remove: List[str]) -> tuple:
    new_documents = []
    new_topics = []
    new_labels = []
    
    for doc, topic, label in zip(documents, topics, labels):
        if label not in topics_to_remove:
            new_documents.append(doc)
            new_topics.append(topic)
            new_labels.append(label)
    
    return new_documents, new_topics, new_labels

def create_markdown_content(documents: List[Document], labels: List[str]) -> str:
    if not documents or not labels:
        return "No data available for download."

    topic_documents = defaultdict(list)
    for doc, label in zip(documents, labels):
        topic_documents[label].append(doc)

    full_text = "# Arxiv Articles by Topic\n\n"

    for topic, docs in topic_documents.items():
        full_text += f"## {topic}\n\n"

        for document in docs:
            full_text += f"### {document.metadata['Title']}\n\n"
            full_text += f"{document.metadata['Summary']}\n\n"

    return full_text

with gr.Blocks(theme="default") as demo:
    gr.Markdown("# Bert Topic Article Organizer App")
    gr.Markdown("Organizes arxiv articles in different topics and exports it in a zip file.")

    state = gr.State(value=[])

    with gr.Row():
        file_uploader = gr.UploadButton(
            "Click to upload", 
            file_types=["json"], 
            file_count="single"
        )
        reprocess_button = gr.Button("Reprocess Documents")
        download_button = gr.Button("Download Results")

    with gr.Row():
        with gr.Column():
            umap_n_neighbors = gr.Slider(minimum=2, maximum=100, value=15, step=1, label="UMAP n_neighbors")
            umap_n_components = gr.Slider(minimum=2, maximum=100, value=5, step=1, label="UMAP n_components")
            umap_min_dist = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.01, label="UMAP min_dist")
        with gr.Column():
            min_topic_size = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="BERTopic min_topic_size")
            nr_topics = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="BERTopic nr_topics")

    with gr.Row():
        output_matrix = gr.DataFrame(
            label="Processing Result",
            headers=["ID", "Topic", "Title"],
            col_count=(3, "fixed"),
            interactive=False
        )

    with gr.Row():
        topic_dropdown = gr.Dropdown(
            label="Select Topics to Remove",
            multiselect=True,
            interactive=True
        )
        remove_topics_button = gr.Button("Remove Selected Topics")

    markdown_output = gr.File(label="Download Markdown", visible=False)

    def update_ui(documents, topics, labels):
        matrix = create_docs_matrix(documents, topics, labels)
        unique_topics = get_unique_topics(labels)
        return matrix, unique_topics

    def process_and_update(state, umap_n_neighbors, umap_n_components, umap_min_dist, min_topic_size, nr_topics):
        documents = state if state else []
        new_documents, new_topics, new_labels = process_documents(documents, umap_n_neighbors, umap_n_components, umap_min_dist, min_topic_size, nr_topics)
        matrix, unique_topics = update_ui(new_documents, new_topics, new_labels)
        return [new_documents, new_topics, new_labels], matrix, unique_topics

    file_uploader.upload(
        fn=lambda file: upload_file(file), 
        inputs=[file_uploader],
        outputs=[state]
    ).then(
        fn=process_and_update,
        inputs=[state, umap_n_neighbors, umap_n_components, umap_min_dist, min_topic_size, nr_topics],
        outputs=[state, output_matrix, topic_dropdown]
    )

    reprocess_button.click(
        fn=process_and_update,
        inputs=[state, umap_n_neighbors, umap_n_components, umap_min_dist, min_topic_size, nr_topics],
        outputs=[state, output_matrix, topic_dropdown]
    )

    def remove_and_update(state, topics_to_remove, umap_n_neighbors, umap_n_components, umap_min_dist, min_topic_size, nr_topics):
        documents, topics, labels = state
        new_documents, new_topics, new_labels = remove_topics(documents, topics, labels, topics_to_remove)
        return process_and_update([new_documents, new_topics, new_labels], umap_n_neighbors, umap_n_components, umap_min_dist, min_topic_size, nr_topics)

    remove_topics_button.click(
        fn=remove_and_update,
        inputs=[state, topic_dropdown, umap_n_neighbors, umap_n_components, umap_min_dist, min_topic_size, nr_topics],
        outputs=[state, output_matrix, topic_dropdown]
    )

    def create_download_file(state):
        documents, _, labels = state
        content = create_markdown_content(documents, labels)
        return gr.File(value=content, visible=True, filename="arxiv_articles_by_topic.md")

    download_button.click(
        fn=create_download_file,
        inputs=[state],
        outputs=[markdown_output]
    )

demo.launch(share=True, show_error=True, max_threads=10, debug=True)