import gradio as gr
import json
from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_core.documents import Document
from typing import Iterator
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from umap import UMAP


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

def process_documents(documents):
    contents = [doc.page_content for doc in documents]

    representation_model = KeyBERTInspired()

    umap_model = UMAP(
        n_neighbors=5, 
        n_components=5, 
        min_dist=0.0, 
        metric='cosine'
    )

    topic_model = BERTopic(
        "english",
        verbose=True,
        umap_model=umap_model,
        min_topic_size=10,
        n_gram_range=(1, 3),
        representation_model=representation_model
    )

    topics, _ = topic_model.fit_transform(contents)

    topic_labels = topic_model.generate_topic_labels(nr_words=3, topic_prefix=False, separator=' ')

    print(f"Generated {len(topic_labels)} topics from data.")
    print("Topic Labels: ", topic_labels)

    results = []

    for doc, topic in zip(documents, topics):
        label = topic_labels[topic]

        results.append([label, doc.metadata['Title']])

    return results

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

    with gr.Row():
        output_matrix = gr.Matrix(
            label="Processing Result",
            col_count=2,
            headers=["Topic", "Title"]
        )

    file_uploader.upload(
        fn=lambda file: upload_file(file), 
        inputs=[file_uploader],
        outputs=[state]
    ).then(
        fn=process_documents,
        inputs=[state],
        outputs=[output_matrix]
    )

demo.launch(share=True, show_error=True, max_threads=10, debug=True)