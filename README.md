# RAG Dataset CLI (TXT/PDF + Ollama + ChromaDB)

This project provides a simple CLI to:
- Create a dataset from `.txt` and `.pdf` files
- Ask questions against a specific dataset
- List datasets with chunk count and creation time
- Delete datasets

The main script is `rag.py`.

## 1) Prerequisites

- Python 3.12+
- Ollama installed and running
- Required Ollama models pulled:
  - `nomic-embed-text`
  - `llama3.1`

Start Ollama and pull models:

```bash
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.1
```

## 2) Install Dependencies

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## 3) CLI Usage

### Create dataset
Ingest all supported files from a folder into a named dataset.

```bash
python rag.py -d <datasetname> -path <folder_path>
```

Example:

```bash
python rag.py -d finance_docs -path ./datanew
```

### Ask question
Query one dataset.

```bash
python rag.py -d <datasetname> -q "<your question>"
```

Example:

```bash
python rag.py -d finance_docs -q "What is the main summary of the report?"
```

### List datasets
Shows dataset name, chunk count, and UTC creation time.

```bash
python rag.py --list-datasets
```

### Delete dataset

```bash
python rag.py --delete-dataset <datasetname>
```

Example:

```bash
python rag.py --delete-dataset finance_docs
```

## 4) Supported Files

- `.txt`
- `.pdf`

Other file types are skipped.

## 5) Storage Layout

Datasets are stored under:

```text
datasets/<datasetname>/
```

Metadata file per dataset:

```text
datasets/<datasetname>/dataset_meta.json
```

## 6) Notes

- Dataset creation fails if the dataset name already exists.
- If no usable TXT/PDF content is found, dataset creation is aborted.
- Retrieval answers are generated from top similar chunks in ChromaDB.
- If you see CLI help text mentioning `assignment3.py`, use `rag.py` in commands for this repository.
