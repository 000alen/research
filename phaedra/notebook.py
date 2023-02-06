import os
import json
import faiss
import torch

from datasets import (
    Dataset,
    load_from_disk,
    Features,
    Value,
    Sequence,
    concatenate_datasets,
)
from typing import Optional
from names_generator import generate_name
from functools import partial
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
)

from phaedra.sources import Source
from phaedra.sources.pdf import PDF
from phaedra.sources.plain import Plain
from phaedra.sources.text import Text
from phaedra.sources.web import Web


torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

_TITLE_COLUMN = "title"
_TEXT_COLUMN = "text"
_EMBEDDINGS_COLUMN = "embeddings"
_D = 768
_M = 128


_FEATURES = Features(
    {
        _TITLE_COLUMN: Value("string"),
        _TEXT_COLUMN: Value("string"),
        _EMBEDDINGS_COLUMN: Sequence(Value("float32")),
    }
)

_FEATURES_WITHOUT_EMBEDDINGS = Features(
    {
        _TITLE_COLUMN: Value("string"),
        _TEXT_COLUMN: Value("string"),
    }
)


_source_types: dict[str, Source] = {
    "pdf": PDF,
    "plain": Plain,
    "text": Text,
    "web": Web,
}

_context_encoder = DPRContextEncoder.from_pretrained(
    "facebook/dpr-ctx_encoder-multiset-base"
).to(device=device)

_context_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
    "facebook/dpr-ctx_encoder-multiset-base"
)

_tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
_model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")


def _split_text(text: str, n=100, character=" ") -> list[str]:
    """Split the text every ``n``-th occurrence of ``character``"""

    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def _split_documents(documents: dict) -> dict:
    """Split documents into passages"""

    titles, texts = [], []
    for title, text in zip(documents[_TITLE_COLUMN], documents[_TEXT_COLUMN]):
        if text is not None:
            for passage in _split_text(text):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {_TITLE_COLUMN: titles, _TEXT_COLUMN: texts}


def _embed(
    documents: dict,
    context_encoder: DPRContextEncoder,
    context_tokenizer: DPRContextEncoderTokenizerFast,
) -> dict:
    """Compute the DPR embeddings of document passages"""

    input_ids = context_tokenizer(
        documents["title"],
        documents["text"],
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )["input_ids"]
    embeddings = context_encoder(
        input_ids.to(device=device), return_dict=True
    ).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}


class Notebook:
    name: str

    _path: Optional[str]
    _dataset: Optional[Dataset]
    _retriever: Optional[RagRetriever]

    def __init__(self, name: Optional[str] = None):
        if name is None:
            name = generate_name()

        self.name = name

        self._path = None
        self._dataset = None
        self._retriever = None

    @staticmethod
    def _get_paths(path: str):
        notebook_path = os.path.join(path, "notebook.json")
        index_path = os.path.join(path, "index.faiss")
        dataset_path = os.path.join(path, "dataset")
        return notebook_path, index_path, dataset_path

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    @classmethod
    def load(cls, path: str):
        notebook_path, index_path, dataset_path = cls._get_paths(path)

        assert os.path.isdir(path)
        assert os.path.isfile(notebook_path)
        assert os.path.isfile(index_path)
        assert os.path.isdir(dataset_path)

        with open(notebook_path, "r") as file:
            notebook = cls.from_dict(json.load(file))

        notebook._path = path
        notebook._dataset = load_from_disk(dataset_path)
        notebook._dataset.load_faiss_index("embeddings", index_path)
        notebook._generate_retriever()

        return notebook

    def _generate_retriever(self):
        self._retriever = RagRetriever.from_pretrained(
            "facebook/rag-sequence-nq",
            index_name="custom",
            indexed_dataset=self._dataset,
        )

    def _generate_index(self):
        index = faiss.IndexHNSWFlat(_D, _M, faiss.METRIC_INNER_PRODUCT)
        self._dataset.add_faiss_index("embeddings", custom_index=index)
        self._generate_retriever()

    def _generate_dataset(self, dataset: Dataset):
        if self._dataset is None:
            self._dataset = dataset
        else:
            self._dataset = concatenate_datasets([self._dataset, dataset])
        self._generate_index()

    def to_dict(self):
        return {
            "name": self.name,
        }

    def save(self, path: Optional[str] = None):
        if path is None:
            assert self._path is not None
            path = self._path
        else:
            self._path = path

        notebook_path, index_path, dataset_path = self._get_paths(path)

        if not os.path.isdir(path):
            os.makedirs(path)

        if not os.path.isdir(dataset_path):
            os.makedirs(dataset_path)

        self._dataset.get_index("embeddings").save(index_path)
        self._dataset.drop_index("embeddings")
        self._dataset.save_to_disk(dataset_path)
        self._dataset.load_faiss_index("embeddings", index_path)

        with open(notebook_path, "w") as file:
            json.dump(self.to_dict(), file)

    def add_source(self, name: str, type: str, origin: str):
        assert type in _source_types

        source_cls = _source_types[type]
        text = source_cls.extract_text(origin)

        dataset = (
            Dataset.from_dict(
                {
                    _TITLE_COLUMN: [name],
                    _TEXT_COLUMN: [text],
                }
            )
            .map(
                _split_documents,
                batched=True,
                features=_FEATURES_WITHOUT_EMBEDDINGS,
            )
            .map(
                partial(
                    _embed,
                    context_encoder=_context_encoder,
                    context_tokenizer=_context_tokenizer,
                ),
                batched=True,
                features=_FEATURES,
            )
        )

        self._generate_dataset(dataset)

    def question(self, question: str):
        if self._retriever is None:
            self._generate_retriever()

        _model.set_retriever(self._retriever)

        inputs = _tokenizer.question_encoder(question, return_tensors="pt")
        output = _model.generate(inputs["input_ids"], max_length=512)

        return _tokenizer.batch_decode(output, skip_special_tokens=True)
