import pickle

from typing import Optional
from transformers import pipeline, Pipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_SEPARATOR = "\n"

DEFAULT_EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
# DEFAULT_MODEL = "gpt2"
# DEFAULT_MODEL = "gpt2-large"
# DEFAULT_MODEL = "EleutherAI/gpt-neo-1.3B"
DEFAULT_MODEL = "EleutherAI/gpt-neo-125M"

DEFAULT_K = 2
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_LENGTH = 1024
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.85
DEFAULT_DO_SAMPLE = True
DEFAULT_NO_REPEAT_NGRAM_SIZE = 2


def _make_splitter() -> CharacterTextSplitter:
    return CharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        separator=DEFAULT_SEPARATOR,
    )


def _make_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBEDDINGS_MODEL,
    )


def _make_pipeline() -> Pipeline:
    return pipeline(
        "text-generation",
        model=DEFAULT_MODEL,
    )


def _make_prompt(sources: list[str], prompt: str) -> str:
    sources = ", ".join(f'"""{source}"""' for source in sources)
    return f"""context: {sources}\n\nprompt: \"\"\"{prompt}\"\"\"\n\nresponse: \"\"\""""


def split_source(
    source: str, splitter: Optional[CharacterTextSplitter] = None
) -> list[str]:
    if splitter is None:
        splitter = _make_splitter()
    return splitter.split_text(source)


def split_sources(
    sources: list[str], splitter: Optional[CharacterTextSplitter] = None
) -> list[str]:
    if splitter is None:
        splitter = _make_splitter()

    splits = []
    for source in sources:
        splits.extend(
            split_source(
                source,
                splitter=splitter,
            )
        )

    return splits


def make_store(
    name: str,
    sources: list[str],
    embeddings: Optional[HuggingFaceEmbeddings] = None,
) -> FAISS:
    if embeddings is None:
        embeddings = _make_embeddings()

    store = FAISS.from_texts(sources, embeddings)
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(store, f)

    return store


def load_store(name: str) -> FAISS:
    with open(f"{name}.pkl", "rb") as f:
        store = pickle.load(f)

    return store


def generate(
    store: FAISS,
    prompt: str,
    k: int = DEFAULT_K,
    temperature=DEFAULT_TEMPERATURE,
    max_length=DEFAULT_MAX_LENGTH,
    top_k=DEFAULT_TOP_K,
    top_p=DEFAULT_TOP_P,
    do_sample=DEFAULT_DO_SAMPLE,
    no_repeat_ngram_size=DEFAULT_NO_REPEAT_NGRAM_SIZE,
    pipeline: Optional[pipeline] = None,
) -> str:
    if pipeline is None:
        pipeline = _make_pipeline()

    sources = [document.page_content for document in store.similarity_search(prompt, k)]
    prompt = _make_prompt(sources, prompt)

    return pipeline(
        prompt,
        return_full_text=False,
        early_stopping=True,
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
    )[0]["generated_text"]
