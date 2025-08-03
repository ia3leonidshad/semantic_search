# Retrieval Experiments

A configurable Python framework for experimenting with different retrieval methods including vector search (Faiss), lexical search (BM25), and hybrid approaches.

## Solution

- Multiple base retrievers are used to build the final pipeline: clip, bm25 and openai text embeddings large.
- Queries are extended to diversify recall sets of retrievers.
- Base retrievers pull candidates, which are labelled by LLM-as-a-judge to build ground truth dataset for evaluation.
- Another 100 queries are generated, passes through the same pipeline and used as a training set for the final ranking model.
- Final ranking model is xboost build on features provided from base retrievers.
- Streamlit demo visualizes features and final results.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

## Running evaluation and model creation

Check out scripts folder [README](./scripts/README.md).

Demo app [README](./docs/README_streamlit.md).

More info in [docs folder](./docs/).

## Project Structure

```
src/
├── models/          # Embedding model implementations
├── retrievers/      # Retrieval method implementations
├── data/            # Data loading and preprocessing
└── utils/           # Utilities and configuration
└── scripts/         # Script to build models and evaluation
```

## License

MIT License
