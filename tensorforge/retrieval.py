"""Local document retrieval and prompt construction for retrieval-augmented generation."""

import re

import numpy as np


class DocumentIndex:
    """A small TF-IDF cosine index with explicit document IDs."""

    def __init__(self, documents):
        if (
            not isinstance(documents, dict)
            or not documents
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in documents.items())
        ):
            raise ValueError("documents must be a nonempty mapping of IDs to text.")
        self.documents = dict(documents)
        self.ids = list(documents)
        words = [self._words(documents[key]) for key in self.ids]
        vocabulary = sorted(set(word for text in words for word in text))
        self.vocabulary = {word: index for index, word in enumerate(vocabulary)}
        counts = np.zeros((len(words), len(vocabulary)))
        for row, text in enumerate(words):
            for word in text:
                counts[row, self.vocabulary[word]] += 1
        self.idf = np.log((1 + len(words)) / (1 + (counts > 0).sum(axis=0))) + 1
        matrix = counts * self.idf
        self.matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    @staticmethod
    def _words(text):
        return re.findall(r"\w+", text.casefold())

    def search(self, query, k=3):
        if not isinstance(query, str) or type(k) is not int or k < 1:
            raise ValueError("Provide a text query and positive k.")
        vector = np.zeros(len(self.vocabulary))
        for word in self._words(query):
            if word in self.vocabulary:
                vector[self.vocabulary[word]] += 1
        vector *= self.idf
        norm = np.linalg.norm(vector)
        if norm == 0:
            return []
        scores = self.matrix @ (vector / norm)
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            {"id": self.ids[i], "text": self.documents[self.ids[i]], "score": float(scores[i])}
            for i in order
            if scores[i] > 0
        ]

    def prompt(self, question, k=3):
        hits = self.search(question, k)
        context = "\n\n".join(f"[{hit['id']}] {hit['text']}" for hit in hits)
        return (
            f"Use the following documents to answer the question.\n\n{context}\n\nQuestion: {question}\nAnswer:",
            hits,
        )

    def generate(self, model, tokenizer, question, k=3, **generation_options):
        prompt, hits = self.prompt(question, k)
        ids = tokenizer.encode(prompt)
        output = model.generate([ids], **generation_options)
        return {"text": tokenizer.decode(output.numpy()[0, len(ids) :]), "documents": hits}
