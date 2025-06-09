import os
import PyPDF2
from docx import Document
import docx2txt
import spacy
import re
from neo4j import AsyncGraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import concurrent.futures
import csv
import time
import logging
from tqdm import tqdm
import pandas as pd
import psutil
from fuzzywuzzy import fuzz
import json
import asyncio

def estimate_resources(folder_path, safety_factor=1.5):
    total_size = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    
    estimated_extracted_size = total_size * 2
    estimated_memory = estimated_extracted_size * 3
    estimated_storage = estimated_extracted_size + (estimated_extracted_size * 3)
    
    estimated_memory *= safety_factor
    estimated_storage *= safety_factor
    
    estimated_memory_gb = estimated_memory / (1024**3)
    estimated_storage_gb = estimated_storage / (1024**3)
    
    available_memory = psutil.virtual_memory().available / (1024**3)
    available_storage = psutil.disk_usage('/').free / (1024**3)
    
    return {
        "estimated_memory_gb": estimated_memory_gb,
        "estimated_storage_gb": estimated_storage_gb,
        "available_memory_gb": available_memory,
        "available_storage_gb": available_storage
    }

class HybridDeduplicator:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        self.neo4j_driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def deduplicate_csv(self, csv_path, threshold=80):
        df = pd.read_csv(csv_path)
        deduplicated = []
        seen = set()

        for _, row in df.iterrows():
            name = row['names']
            if isinstance(name, str):
                name = eval(name)
            if isinstance(name, list) and name:
                name = name[0].lower()
            else:
                continue

            if name not in seen:
                similar = [s for s in seen if fuzz.ratio(name, s) > threshold]
                if not similar:
                    seen.add(name)
                    deduplicated.append(row)
                else:
                    most_similar = max(similar, key=lambda s: fuzz.ratio(name, s))
                    existing_index = next(i for i, r in enumerate(deduplicated) if r['names'] and eval(r['names'])[0].lower() == most_similar)
                    deduplicated[existing_index] = self.merge_rows(deduplicated[existing_index], row)

        pd.DataFrame(deduplicated).to_csv(csv_path, index=False)

    @staticmethod
    def merge_rows(row1, row2):
        merged = {}
        for key in row1.keys():
            if isinstance(row1[key], str) and row1[key].startswith('[') and row1[key].endswith(']'):
                val1 = eval(row1[key])
                val2 = eval(row2[key])
                if isinstance(val1, list) and isinstance(val2, list):
                    merged[key] = str(list(set(val1 + val2)))
                else:
                    merged[key] = row1[key]
            else:
                merged[key] = row1[key] if row1[key] else row2[key]
        return merged

    async def deduplicate_neo4j(self):
        async with self.neo4j_driver.session() as session:
            await session.run("""
                MATCH (p1:Person)
                MATCH (p2:Person)
                WHERE p1 <> p2 AND p1.name =~ p2.name
                WITH p1, p2, apoc.text.levenshteinSimilarity(p1.name, p2.name) AS similarity
                WHERE similarity > 0.8
                CALL apoc.merge.nodes([p1, p2])
                YIELD node
                RETURN count(node)
            """)

    async def process(self, csv_path):
        print("Deduplicating CSV...")
        self.deduplicate_csv(csv_path)
        print("Deduplicating in Neo4j...")
        await self.deduplicate_neo4j()

class CriminalLawDocumentParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.max_length = 2000000  # 2 million characters
        self.neo4j_driver = AsyncGraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "Monster_12#$"))
        self.tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english')
        self.lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
        self.deduplicator = HybridDeduplicator("bolt://localhost:7687", "neo4j", "Monster_12#$")

        # Set up logging
        logging.basicConfig(filename='document_processing.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()

        # Add custom entities for criminal law domain
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        patterns = [
            {"label": "CRIME", "pattern": [{"LOWER": {"IN": ["murder", "theft", "assault", "fraud", "burglary"]}}]},
            {"label": "LEGAL_TERM", "pattern": [{"LOWER": {"IN": ["defendant", "plaintiff", "verdict", "sentence", "bail", "parole"]}}]},
            {"label": "EVIDENCE", "pattern": [{"LOWER": {"IN": ["dna", "fingerprint", "witness", "testimony", "exhibit"]}}]}
        ]
        ruler.add_patterns(patterns)

    async def clear_database(self):
        async with self.neo4j_driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
        self.logger.info("Database cleared")

    async def create_indexes(self):
        async with self.neo4j_driver.session() as session:
            await session.run("CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.name)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (l:Location) ON (l.name)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (c:Crime) ON (c.name)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (lt:LegalTerm) ON (lt.name)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (e:Evidence) ON (e.name)")
            await session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.source)")
        self.logger.info("Indexes created in Neo4j")

    def process_large_text(self, text):
        max_chunk_size = 1000000  # 1 million characters
        for i in range(0, len(text), max_chunk_size):
            chunk = text[i:i+max_chunk_size]
            doc = self.nlp(chunk)
            yield doc

    def parse_pdf(self, file_path):
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    yield page.extract_text()
        except Exception as e:
            self.logger.error(f"Error parsing PDF file {file_path}: {str(e)}")
            raise

    def parse_docx(self, file_path):
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                yield paragraph.text
        except Exception as e:
            self.logger.error(f"Error parsing DOCX file {file_path}: {str(e)}")
            raise

    def parse_txt(self, file_path):
        try:
            with open(file_path, 'r', errors='ignore') as file:
                for line in file:
                    yield line
        except Exception as e:
            self.logger.error(f"Error parsing TXT file {file_path}: {str(e)}")
            raise

    def parse_doc(self, file_path):
        try:
            text = docx2txt.process(file_path)
            yield text
        except Exception as e:
            self.logger.error(f"Error parsing DOC file {file_path}: {str(e)}")
            raise

    def extract_topics(self, text):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        num_topics = min(10, len(feature_names))
        tfidf_scores = tfidf_matrix.toarray().flatten()
        top_indices = tfidf_scores.argsort()[-num_topics:][::-1]
        return [feature_names[i] for i in top_indices]

    def extract_information(self, text, source_document):
        entities = {"PERSON": [], "GPE": [], "CRIME": [], "LEGAL_TERM": [], "EVIDENCE": []}
        topics_discussed = self.extract_topics(text)
        date_pattern = r'\d{1,2}/\d{1,2}/\d{4}'
        dates = re.findall(date_pattern, text)
        comments = []

        for doc in self.process_large_text(text):
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)
            comments.extend([sent.text for sent in doc.sents if len(sent.text.split()) > 5])

        return {
            "names": entities["PERSON"],
            "topics_discussed": topics_discussed,
            "locations": entities["GPE"],
            "crimes": entities["CRIME"],
            "legal_terms": entities["LEGAL_TERM"],
            "evidence": entities["EVIDENCE"],
            "dates": dates,
            "comments": comments,
            "source_document": source_document
        }

    async def process_document_async(self, file_path, output_csv_path):
        start_time = time.time()
        self.logger.info(f"Started processing: {file_path}")
        
        try:
            _, file_extension = os.path.splitext(file_path)
            parse_method = getattr(self, f"parse_{file_extension[1:].lower()}", None)
            
            if not parse_method:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            text = " ".join(list(parse_method(file_path)))
            data = self.extract_information(text, os.path.basename(file_path))
            
            with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=data.keys())
                if csvfile.tell() == 0:
                    writer.writeheader()
                writer.writerow(data)
            
            end_time = time.time()
            processing_time = end_time - start_time
            self.logger.info(f"Finished processing: {file_path}. Time taken: {processing_time:.2f} seconds")
            
            return f"Successfully processed: {file_path}", processing_time
        
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return f"Error processing file: {file_path}", 0

    async def process_folder_async(self, folder_path, output_csv_path):
        resources = estimate_resources(folder_path)
        print(f"Estimated required memory: {resources['estimated_memory_gb']:.2f} GB")
        print(f"Estimated required storage: {resources['estimated_storage_gb']:.2f} GB")
        print(f"Available memory: {resources['available_memory_gb']:.2f} GB")
        print(f"Available storage: {resources['available_storage_gb']:.2f} GB")

        if resources['estimated_memory_gb'] > resources['available_memory_gb'] or \
           resources['estimated_storage_gb'] > resources['available_storage_gb']:
            proceed = input("WARNING: Estimated resources exceed available resources. Proceed? (y/n): ")
            if proceed.lower() != 'y':
                print("Aborting process.")
                return

        start_time = time.time()
        file_list = []
        for root, _, files in os.walk(folder_path):
            file_list.extend([os.path.join(root, file) for file in files])

        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(executor, self.process_document_async, file_path, output_csv_path)
                for file_path in file_list
            ]
            for future in tqdm(asyncio.as_completed(futures), total=len(file_list), desc="Processing documents"):
                result, processing_time = await future
                results.append((result, processing_time))

        end_time = time.time()
        total_time = end_time - start_time
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")

        for result, processing_time in results:
            print(f"{result} - Time taken: {processing_time:.2f} seconds")

        print(f"Total processing time: {total_time:.2f} seconds")
        
        print("Deduplicating data...")
        await self.deduplicator.process(output_csv_path)
        
        return output_csv_path

    async def load_csv_to_database(self, csv_path):
        await self.create_indexes()
        df = pd.read_csv(csv_path)
        async with self.neo4j_driver.session() as session:
            for _, row in df.iterrows():
                await session.run(
                    """
                    CREATE (d:Document {source: $source})
                    WITH d
                    UNWIND $names as name
                    MERGE (p:Person {name: name})
                    MERGE (p)-[:MENTIONED_IN]->(d)
                    WITH d
                    UNWIND $topics as topic
                    MERGE (t:Topic {name: topic})
                    MERGE (t)-[:DISCUSSED_IN]->(d)
                    WITH d
                    UNWIND $locations as location
                    MERGE (l:Location {name: location})
                    MERGE (l)-[:MENTIONED_IN]->(d)
                    WITH d
                    UNWIND $crimes as crime
                    MERGE (c:Crime {name: crime})
                    MERGE (c)-[:MENTIONED_IN]->(d)
                    WITH d
                    UNWIND $legal_terms as term
                    MERGE (lt:LegalTerm {name: term})
                    MERGE (lt)-[:MENTIONED_IN]->(d)
                    WITH d
                    UNWIND $evidence as item
                    MERGE (e:Evidence {name: item})
                    MERGE (e)-[:MENTIONED_IN]->(d)
                    WITH d
                    UNWIND $dates as date
                    MERGE (dt:Date {value: date})
                    MERGE (dt)-[:MENTIONED_IN]->(d)
                    """,
                    source=row['source_document'],
                    names=eval(row['names']),
                    topics=eval(row['topics_discussed']),
                    locations=eval(row['locations']),
                    crimes=eval(row['crimes']),
                    legal_terms=eval(row['legal_terms']),
                    evidence=eval(row['evidence']),
                    dates=eval(row['dates'])
                )
        self.logger.info(f"Loaded data from {csv_path} to database")

async def main_async():
    parser = CriminalLawDocumentParser()
    
    clear_db = input("Do you want to clear the database before processing? (y/n): ").lower() == 'y'
    if clear_db:
        await parser.clear_database()

    folder_path = input("Enter the folder path to process: ")
    output_csv_path = input("Enter the output CSV file path: ")
    resume = input("Do you want to resume from the last checkpoint? (y/n): ").lower() == 'y'

    processed_files = load_checkpoint() if resume else set()

    csv_path = await parser.process_folder_async(folder_path, output_csv_path)
    
    load_to_db = input("Do you want to load the processed data to the database? (y/n): ").lower() == 'y'
    if load_to_db:
        await parser.load_csv_to_database(csv_path)

    print("Processing completed. Check the log file for details.")

if __name__ == "__main__":
    asyncio.run(main_async())
