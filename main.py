import os
import logging

from phaedra.notebook import Notebook

logging.disable(logging.INFO)

if os.path.isdir("Bitcoin"):
    print("Loading notebook...")
    notebook = Notebook.load("Bitcoin")
else:
    print("Creating notebook...")
    notebook = Notebook(
        name="Bitcoin",
    )

    notebook.add_source(
        name="Bitcoin",
        type="pdf",
        origin="./test/bitcoin.pdf",
    )

    notebook.save(
        path="./Bitcoin/",
    )

print(notebook.question("What is the blockchain?"))
