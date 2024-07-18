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

    return {
        "topics": topics.tolist() if isinstance(topics, np.ndarray) else topics,  # Convert to list if it's a numpy array
        "labels": topic_labels,
        "documents": documents
    }

def create_docs_matrix(state: Dict) -> List[List[str]]:
    if not state or 'documents' not in state:
        return []
    results = []
    for i, (doc, topic) in enumerate(zip(state['documents'], state['topics'])):
        label = state['labels'][topic]
        results.append([str(i), label, doc.metadata['Title']])
    return results

def remove_documents(state: Dict, rows_to_remove: List[int]) -> Dict:
    if not state or 'documents' not in state:
        return state
    
    new_documents = [doc for i, doc in enumerate(state['documents']) if i not in rows_to_remove]
    new_topics = [topic for i, topic in enumerate(state['topics']) if i not in rows_to_remove]
    
    return {
        "documents": new_documents,
        "topics": new_topics,
        "labels": state['labels']
    }

def remove_topics(state: Dict, topics_to_remove: List[str]) -> Dict:
    if not state or 'documents' not in state:
        return state
    
    new_documents = []
    new_topics = []
    
    for doc, topic in zip(state['documents'], state['topics']):
        if state['labels'][topic] not in topics_to_remove:
            new_documents.append(doc)
            new_topics.append(topic)
    
    return {
        "documents": new_documents,
        "topics": new_topics,
        "labels": state['labels']
    }

def create_markdown_content(state: Dict) -> str:
    if not state or 'documents' not in state or 'topics' not in state or 'labels' not in state:
        return "No data available for download."

    topic_documents = defaultdict(list)
    for doc, topic in zip(state['documents'], state['topics']):
        topic_label = state['labels'][topic]
        topic_documents[topic_label].append(doc)

    full_text = "# Arxiv Articles by Topic\n\n"

    for topic, documents in topic_documents.items():
        full_text += f"## {topic}\n\n"

        for document in documents:
            full_text += f"### {document.metadata['Title']}\n\n"
            full_text += f"{document.metadata['Summary']}\n\n"

    return full_text

with gr.Blocks(theme="default") as demo:
    gr.Markdown("# Bert Topic Article Organizer App")
    gr.Markdown("Organizes arxiv articles in different topics and exports it in a zip file.")

    state = gr.State(value={})

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
            interactive=True
        )

    with gr.Row():
        remove_docs_button = gr.Button("Remove Selected Documents")
        remove_topics_button = gr.Button("Remove Selected Topics")

    markdown_output = gr.File(label="Download Markdown", visible=False)

    def reprocess_documents(state, umap_n_neighbors, umap_n_components, umap_min_dist, min_topic_size, nr_topics):
        if not state or 'documents' not in state:
            return "No documents to reprocess. Please upload a file first.", []
        
        new_state = process_documents(state['documents'], umap_n_neighbors, umap_n_components, umap_min_dist, min_topic_size, nr_topics)
        return new_state, create_docs_matrix(new_state)

    file_uploader.upload(
        fn=lambda file: upload_file(file), 
        inputs=[file_uploader],
        outputs=[state]
    ).then(
        fn=process_documents,
        inputs=[state, umap_n_neighbors, umap_n_components, umap_min_dist, min_topic_size, nr_topics],
        outputs=[state]
    ).then(
        fn=create_docs_matrix,
        inputs=[state],
        outputs=[output_matrix]
    )

    reprocess_button.click(
        fn=reprocess_documents,
        inputs=[state, umap_n_neighbors, umap_n_components, umap_min_dist, min_topic_size, nr_topics],
        outputs=[state, output_matrix]
    )

    remove_docs_button.click(
        fn=lambda state, df: remove_documents(state, [int(row[0]) for row in df if row[0]]),
        inputs=[state, output_matrix],
        outputs=[state]
    ).then(
        fn=reprocess_documents,
        inputs=[state, umap_n_neighbors, umap_n_components, umap_min_dist, min_topic_size, nr_topics],
        outputs=[state, output_matrix]
    )

    remove_topics_button.click(
        fn=lambda state, df: remove_topics(state, list(set(row[1] for row in df if row[1]))),
        inputs=[state, output_matrix],
        outputs=[state]
    ).then(
        fn=reprocess_documents,
        inputs=[state, umap_n_neighbors, umap_n_components, umap_min_dist, min_topic_size, nr_topics],
        outputs=[state, output_matrix]
    )

    download_button.click(
        fn=lambda state: gr.File.update(
            value=create_markdown_content(state),
            visible=True,
            filename="arxiv_articles_by_topic.md"
        ),
        inputs=[state],
        outputs=[markdown_output]
    )

demo.launch(share=True, show_error=True, max_threads=10, debug=True)