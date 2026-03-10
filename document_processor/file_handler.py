import os
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from langchain_text_splitters import MarkdownHeaderTextSplitter
from config import constants
from config.settings import settings
from utils.logging import logger

class DocumentProcessor:
    def __init__(self):
        self.headers = [("#", "header1"), ("##", "header2")]
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def validate_files(self, files):
        """
        Validate uploaded files based on size and type.
        """
        total_size = 0
        for f in files:
            file_path = f if isinstance(f, str) else f.name
            total_size += os.path.getsize(file_path)

        if total_size > settings.MAX_TOTAL_SIZE:
            raise ValueError(f"Total size of all files exceeds maximum of {settings.MAX_TOTAL_SIZE / 1024 / 1024} MB")
        
        for file in files:
            file_path = file if isinstance(file, str) else file.name
            file_size = os.path.getsize(file_path)
            if file_size > settings.MAX_FILE_SIZE:
                raise ValueError(f"File {file_path} exceeds maximum size of {settings.MAX_FILE_SIZE / 1024 / 1024} MB")
        
        logger.info(f"All {len(files)} files validated successfully")
        return True

    def _generate_hash(self, content: bytes) -> str:
        """Generate SHA256 hash for content"""
        return hashlib.sha256(content).hexdigest()

    def _save_to_cache(self, chunks: List, cache_path: Path):
        with open(cache_path, "wb") as f:
            pickle.dump({
                "timestamp": datetime.now().timestamp(),
                "chunks": chunks    
            }, f)
        logger.info(f"Saved {len(chunks)} chunks to cache: {cache_path}")

    def _load_from_cache(self, cache_path: Path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["chunks"]

    def _is_valid_cache(self, cache_path: Path) -> bool:
        """Check if cache is valid"""
        if not cache_path.exists():
            return False
        
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=settings.CACHE_EXPIRE_DAYS)

    def _process_files(self, file_obj_or_path):
        """Cloud parsing logic with LlamaParse + Local Fallback"""
        file_path = file_obj_or_path if isinstance(file_obj_or_path, str) else file_obj_or_path.name
        if not file_path.endswith((".md", ".txt", ".pdf", ".docx")):
            logger.warning(f"File {file_path} is not a supported file type")
            return []

        markdown_text = ""
        try:
            # Initialize LlamaParse for Cloud OCR
            from llama_parse import LlamaParse
            
            parser = LlamaParse(
                api_key=settings.LLAMA_CLOUD_API_KEY,
                result_type="markdown",  # Parse into Markdown for our header splitter
                verbose=True,
                language="en" # Set to vi if you have mostly Vietnamese docs
            )

            # Sync parse
            logger.info(f"Sending {file_path} to LlamaParse Cloud...")
            documents = parser.load_data(file_path)
            
            # Merge all parsed pages into one markdown text
            markdown_text = "\n\n".join([doc.text for doc in documents])
            
        except Exception as e:
            logger.warning(f"LlamaParse cloud OCR failed: {e}. Falling back to basic local extraction.")
            try:
                if file_path.endswith(".pdf"):
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    markdown_text = "\n\n".join([doc.page_content for doc in docs])
                elif file_path.endswith((".txt", ".md")):
                    from langchain_community.document_loaders import TextLoader
                    loader = TextLoader(file_path, autodetect_encoding=True)
                    docs = loader.load()
                    markdown_text = "\n\n".join([doc.page_content for doc in docs])
                else:
                    logger.error(f"Fallback reading for {file_path} format not currently supported without LlamaParse.")
                    return []
            except Exception as fallback_e:
                logger.error(f"Fallback extraction also failed: {fallback_e}")
                return []
        
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers)
        chunks = splitter.split_text(markdown_text)
        return chunks
        
    def process(self, files):
        """Process files with caching for subsequent queries"""
        self.validate_files(files)
        all_chunks = []
        seen_hashes = set()

        for file in files:
            try:
                file_path = file if isinstance(file, str) else file.name
                with open(file_path, "rb") as f:
                    file_hash = self._generate_hash(f.read())
                    
                    cache_path = self.cache_dir / f"{file_hash}.pkl"
                    
                    if self._is_valid_cache(cache_path):
                        logger.info(f"Loading chunks from cache for {file_path}")
                        chunks = self._load_from_cache(cache_path)
                    else:
                        logger.info(f"Processing new file: {file_path}")
                        chunks = self._process_files(file)
                        self._save_to_cache(chunks, cache_path)
                    
                    for chunk in chunks:
                        chunk_hash = self._generate_hash(chunk.page_content.encode())
                        if chunk_hash not in seen_hashes:
                            all_chunks.append(chunk)
                            seen_hashes.add(chunk_hash)
            except Exception as e:
                logger.error(f"Error processing file {file.name}: {e}")
                continue
        
        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks