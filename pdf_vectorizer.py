# Pdf processor

import os
from typing import List, Dict
from datetime import datetime
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


OPENAI_API_KEY = os.environ['OPENAI_API_KEY']


class PDFProcessingSystem:
    def __init__(self, pdfs_directory: str, storage_directory: str):
        """
        Initialize the PDF processing system

        Args:
            pdfs_directory (str): Directory containing PDF files
            storage_directory (str): Directory to store processed data
        """
        self.pdfs_directory = Path(pdfs_directory)
        self.storage_directory = Path(storage_directory)
        self.metadata_file = self.storage_directory / "processing_metadata.json"
        self.vectorstore_path = self.storage_directory / "vectorstore"

        # Create storage directory if it doesn't exist
        self.storage_directory.mkdir(parents=True, exist_ok=True)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=200,
            length_function=len,
        )

        self.embeddings = OpenAIEmbeddings()

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for change detection"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def load_metadata(self) -> Dict:
        """Load processing metadata from JSON file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def save_metadata(self, metadata: Dict):
        """Save processing metadata to JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def process_pdfs(self):
        """Process all PDFs in the directory and update vectorstore"""
        pdf_files = list(self.pdfs_directory.glob("*.pdf"))
        metadata = self.load_metadata()

        # Check which files need processing
        files_to_process = []
        for pdf_path in pdf_files:
            current_hash = self.calculate_file_hash(pdf_path)
            stored_hash = metadata.get(str(pdf_path), {}).get('hash', '')

            if current_hash != stored_hash:
                files_to_process.append(pdf_path)

        if not files_to_process:
            print("All files are up to date!")
            return

        print(f"Processing {len(files_to_process)} PDF files...")

        # Process PDFs and create documents
        all_documents = []
        for pdf_path in tqdm(files_to_process, desc="Processing PDFs"):
            try:
                # Load and split the PDF
                loader = PyPDFLoader(str(pdf_path))
                documents = loader.load_and_split()
                chunks = self.text_splitter.split_documents(documents)

                # Add source file information to metadata
                for chunk in chunks:
                    chunk.metadata["source_file"] = str(pdf_path)

                all_documents.extend(chunks)

                # Update metadata
                metadata[str(pdf_path)] = {
                    'hash': self.calculate_file_hash(pdf_path),
                    'last_processed': datetime.now().isoformat(),
                    'num_chunks': len(chunks)
                }

            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
                continue

        if self.vectorstore_path.exists():
            print("Loading existing vectorstore...")
            vectorstore = FAISS.load_local(
                str(self.vectorstore_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("Adding new documents to existing vectorstore...")
            vectorstore.add_documents(all_documents)
        else:
            print("Creating new vectorstore...")
            vectorstore = FAISS.from_documents(all_documents, self.embeddings)

            # Save vectorstore and metadata
        print("Saving vectorstore...")
        vectorstore.save_local(str(self.vectorstore_path))
        self.save_metadata(metadata)

        print("Processing complete!")
        return vectorstore

    def load_vectorstore(self) -> FAISS:
        """Load the stored vectorstore"""
        if not self.vectorstore_path.exists():
            raise FileNotFoundError("Vectorstore not found. Please process PDFs first.")
        return FAISS.load_local(
            str(self.vectorstore_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def get_processing_stats(self) -> Dict:
        """Get statistics about processed PDFs"""
        metadata = self.load_metadata()
        return {
            'total_files': len(metadata),
            'total_chunks': sum(file_data['num_chunks'] for file_data in metadata.values()),
            'last_processed': max(file_data['last_processed'] for file_data in metadata.values())
            if metadata else None
        }


# Example usage
if __name__ == "__main__":
    # Initialize the system
    processor = PDFProcessingSystem(
        pdfs_directory="./course_pdfs",
        storage_directory="./storage_directory"
    )

    # Process PDFs
    vectorstore = processor.process_pdfs()

    # Print processing stats
    stats = processor.get_processing_stats()
    print("\nProcessing Statistics:")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Total chunks created: {stats['total_chunks']}")
    print(f"Last processing time: {stats['last_processed']}")