# ============================================================
# PDF LOADER
# ============================================================

from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# LOAD PDF DOCUMENTS
# ============================================================

def load_documents(data_path="data"):
    """
    Load all PDF files from the data folder
    and convert every page into a LangChain Document.
    """

    documents = []

    # Find all PDF files inside the data folder
    for pdf_file in Path(data_path).glob("*.pdf"):

        # Open PDF
        reader = PdfReader(pdf_file)

        # Read every page
        for page_number, page in enumerate(reader.pages):

            # Extract text from page
            text = page.extract_text() or ""

            # Create LangChain Document
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_file.name,
                        "page": page_number + 1
                    }
                )
            )

    return documents


# ============================================================
# SPLIT DOCUMENTS INTO CHUNKS
# ============================================================

def create_chunks(documents):
    """
    Split documents into smaller overlapping chunks
    for embedding and retrieval.
    """

    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # Split documents
    chunks = text_splitter.split_documents(documents)

    return chunks