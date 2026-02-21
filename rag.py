import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import click
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM

DATASETS_ROOT = Path("./datasets")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1"

def dataset_path(dataset_name: str) -> Path:
    return DATASETS_ROOT / dataset_name


def metadata_path(dataset_name: str) -> Path:
    return dataset_path(dataset_name) / "dataset_meta.json"


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def read_txt_file(file_path: Path) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return ""


def read_pdf_file(file_path: Path) -> str:
    text_parts = []
    with file_path.open("rb") as f:
        pdf = PyPDF2.PdfReader(f)
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def extract_text_from_file(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext == ".txt":
        return read_txt_file(file_path).strip()
    if ext == ".pdf":
        return read_pdf_file(file_path).strip()
    return ""


def ingest_dataset(dataset_name: str, folder_path: Path) -> None:
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    target = dataset_path(dataset_name)

    if target.exists():
        raise click.ClickException(
            f"Dataset '{dataset_name}' already exists. Delete it first or use another name."
        )

    target.mkdir(parents=True, exist_ok=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    vector_store = Chroma(
        persist_directory=str(target),
        embedding_function=get_embeddings(),
    )

    files_processed = 0
    files_skipped = 0

    for file_path in sorted(folder_path.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in {".txt", ".pdf"}:
            files_skipped += 1
            continue

        raw_text = extract_text_from_file(file_path)
        if not raw_text:
            click.echo(f"Skipping empty/unreadable file: {file_path.name}")
            files_skipped += 1
            continue

        chunks = splitter.split_text(raw_text)
        if not chunks:
            click.echo(f"Skipping file with no chunks: {file_path.name}")
            files_skipped += 1
            continue

        metadatas = [
            {
                "source": file_path.name,
                "extension": file_path.suffix.lower(),
                "dataset": dataset_name,
            }
            for _ in chunks
        ]
        vector_store.add_texts(texts=chunks, metadatas=metadatas)
        files_processed += 1
        click.echo(f"Ingested: {file_path.name} ({len(chunks)} chunks)")

    vector_store.persist()
    chunk_count = vector_store._collection.count()

    if chunk_count == 0:
        shutil.rmtree(target, ignore_errors=True)
        raise click.ClickException(
            "No usable TXT/PDF content found. Dataset was not created."
        )

    dataset_meta = {
        "dataset_name": dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_folder": str(folder_path.resolve()),
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "chunk_count": chunk_count,
    }
    metadata_path(dataset_name).write_text(json.dumps(dataset_meta, indent=2), encoding="utf-8")

    click.echo(f"\nDataset created: {dataset_name}")
    click.echo(f"Chunks stored: {chunk_count}")
    click.echo(f"Processed files: {files_processed}")
    click.echo(f"Skipped files: {files_skipped}")


def ask_question(dataset_name: str, question: str) -> None:
    target = dataset_path(dataset_name)
    if not target.exists():
        raise click.ClickException(f"Dataset '{dataset_name}' does not exist.")

    vector_store = Chroma(
        persist_directory=str(target),
        embedding_function=get_embeddings(),
    )
    docs = vector_store.similarity_search(question, k=4)
    if not docs:
        click.echo("No relevant context found in this dataset.")
        return

    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"{doc.page_content}\n(Source: {source})")
    context = "\n\n".join(context_parts)

    model = OllamaLLM(model=LLM_MODEL)
    prompt = (
        "Answer only from the provided context. If context is insufficient, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer with source references."
    )
    answer = model.invoke(prompt)
    click.echo(f"\nAnswer:\n{answer}")


def read_dataset_meta(dataset_name: str) -> dict:
    meta_file = metadata_path(dataset_name)
    if not meta_file.exists():
        return {}
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def list_datasets() -> None:
    if not DATASETS_ROOT.exists():
        click.echo("No datasets found.")
        return

    dataset_dirs = sorted([p for p in DATASETS_ROOT.iterdir() if p.is_dir()])
    if not dataset_dirs:
        click.echo("No datasets found.")
        return

    click.echo("Dataset Name | Chunks | Created At (UTC)")
    click.echo("-" * 70)

    for d in dataset_dirs:
        name = d.name
        created_at = "unknown"
        chunk_count = "unknown"

        meta = read_dataset_meta(name)
        if meta:
            created_at = meta.get("created_at", "unknown")
            chunk_count = str(meta.get("chunk_count", "unknown"))

        if chunk_count == "unknown":
            try:
                vector_store = Chroma(
                    persist_directory=str(d),
                    embedding_function=get_embeddings(),
                )
                chunk_count = str(vector_store._collection.count())
            except Exception:
                chunk_count = "error"

        click.echo(f"{name} | {chunk_count} | {created_at}")


def delete_dataset(dataset_name: str) -> None:
    target = dataset_path(dataset_name)
    if not target.exists():
        raise click.ClickException(f"Dataset '{dataset_name}' does not exist.")
    shutil.rmtree(target)
    click.echo(f"Deleted dataset: {dataset_name}")


@click.command()
@click.option("-d", "--dataset", help="Dataset name.")
@click.option(
    "-path",
    "--path",
    "folder_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Folder path to ingest (TXT/PDF files only).",
)
@click.option("-q", "--question", help="Ask a question against a dataset.")
@click.option("--list-datasets", "list_datasets_flag", is_flag=True, help="List all datasets.")
@click.option("--delete-dataset", "delete_dataset_name", help="Delete dataset by name.")
def main(dataset, folder_path, question, list_datasets_flag, delete_dataset_name):
    if list_datasets_flag:
        if any([dataset, folder_path, question, delete_dataset_name]):
            raise click.ClickException(
                "--list-datasets must be used alone."
            )
        list_datasets()
        return

    if delete_dataset_name:
        if any([dataset, folder_path, question, list_datasets_flag]):
            raise click.ClickException(
                "--delete-dataset must be used alone."
            )
        delete_dataset(delete_dataset_name)
        return

    if dataset and folder_path and not question:
        ingest_dataset(dataset, folder_path)
        return

    if dataset and question and not folder_path:
        ask_question(dataset, question)
        return

    raise click.ClickException(
        "Invalid usage.\n"
        "Create dataset: python assignment3.py -d <datasetname> -path <folder path>\n"
        "Ask question:   python assignment3.py -d <datasetname> -q \"<question>\"\n"
        "List datasets:  python assignment3.py --list-datasets\n"
        "Delete dataset: python assignment3.py --delete-dataset <datasetname>"
    )


if __name__ == "__main__":
    main()
