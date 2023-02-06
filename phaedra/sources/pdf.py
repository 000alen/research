import pdfplumber

from phaedra.sources import Source


class PDF(Source):
    @staticmethod
    def extract_text(path: str) -> str:
        with pdfplumber.open(path) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages])
