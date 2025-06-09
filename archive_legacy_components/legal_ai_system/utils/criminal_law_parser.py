import os
import PyPDF2
from docx import Document
import docx2txt
import spacy
import numpy as np
import re
from spacy.tokens import Doc
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import concurrent.futures
from functools import partial
import csv

class CriminalLawDocumentParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        # Increase the maximum length limit (adjust as needed)
        self.nlp.max_length = 2000000  # 2 million characters
        
        self.neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "Monster_12#$"))
        self.tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english')
        self.lda_model = LatentDirichletAllocation(n_components=10, random_state=42)



        # Add custom entities for criminal law domain
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        patterns = [
            {"label": "CRIME", "pattern": [{"LOWER": {"IN": ["murder", "theft", "assault", "fraud", "burglary"]}}]},
            {"label": "LEGAL_TERM", "pattern": [{"LOWER": {"IN": ["defendant", "plaintiff", "verdict", "sentence", "bail", "parole"]}}]},
            {"label": "EVIDENCE", "pattern": [{"LOWER": {"IN": ["dna", "fingerprint", "witness", "testimony", "exhibit"]}}]}
        ]
        ruler.add_patterns(patterns)

   def process_large_text(self, text):
        # Process text in chunks
        max_chunk_size = 1000000  # 1 million characters
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        docs = []
        for chunk in chunks:
            doc = self.nlp(chunk)
            docs.append(doc)
        
        # Combine the processed chunks
        combined_doc = Doc.from_docs(docs)
        return combined_doc

    def extract_information(self, text, source_document):
        if len(text) > self.nlp.max_length:
            doc = self.process_large_text(text)
        else:
            doc = self.nlp(text)
        
        names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        topics_discussed = self.extract_topics(text)
        locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        crimes = [ent.text for ent in doc.ents if ent.label_ == "CRIME"]
        legal_terms = [ent.text for ent in doc.ents if ent.label_ == "LEGAL_TERM"]
        evidence = [ent.text for ent in doc.ents if ent.label_ == "EVIDENCE"]
        
        date_pattern = r'\d{1,2}/\d{1,2}/\d{4}'
        dates = re.findall(date_pattern, text)
        
        comments = [sent.text for sent in doc.sents if len(sent.text.split()) > 5]
        
        return {
            "names": names,
            "topics_discussed": topics_discussed,
            "locations": locations,
            "crimes": crimes,
            "legal_terms": legal_terms,
            "evidence": evidence,
            "dates": dates,
            "comments": comments,
            "source_document": source_document
        }

    def parse_pdf(self, file_path):
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except PyPDF2.PdfReadError as e:
            raise ValueError(f"Error parsing PDF file: {e}")

    def parse_docx(self, file_path):
        doc = Document(file_path)
        return " ".join([paragraph.text for paragraph in doc.paragraphs])

    def parse_txt(self, file_path):
        with open(file_path, 'r', errors='ignore') as file:
            return file.read()

    def parse_doc(self, file_path):
        return docx2txt.process(file_path)

    def extract_topics(self, text):
        doc = self.nlp(text)
        
        # Named Entity Recognition
        ner_topics = [ent.text for ent in doc.ents if ent.label_ in ["CRIME", "LEGAL_TERM", "EVIDENCE", "ORG", "PERSON", "GPE"]]
        
        # Keyword Extraction using TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Adjust the number of topics based on the document length
        num_topics = min(10, len(feature_names))
        
        # Convert the sparse matrix to a dense numpy array and flatten it
        tfidf_scores = tfidf_matrix.toarray().flatten()
        
        # Get the indices of the top scoring terms
        top_indices = tfidf_scores.argsort()[-num_topics:][::-1]
        
        # Get the top scoring terms
        tfidf_topics = [feature_names[i] for i in top_indices]
        
        # Topic Modeling using LDA (optional for single documents)
        if len(feature_names) > 10:  # Only perform LDA if there are enough terms
            lda_output = self.lda_model.fit_transform(tfidf_matrix)
            lda_topics = []
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_indices = topic.argsort()[:-5 - 1:-1]
                top_words = [feature_names[i] for i in top_indices]
                lda_topics.extend(top_words)
        else:
            lda_topics = []
        
        # Combine all topics and remove duplicates
        all_topics = list(set(ner_topics + tfidf_topics + lda_topics))
        
        return all_topics

    def extract_information(self, text, source_document):
        doc = self.nlp(text)
        
        names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        topics_discussed = self.extract_topics(text)
        locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        crimes = [ent.text for ent in doc.ents if ent.label_ == "CRIME"]
        legal_terms = [ent.text for ent in doc.ents if ent.label_ == "LEGAL_TERM"]
        evidence = [ent.text for ent in doc.ents if ent.label_ == "EVIDENCE"]
        
        date_pattern = r'\d{1,2}/\d{1,2}/\d{4}'
        dates = re.findall(date_pattern, text)
        
        comments = [sent.text for sent in doc.sents if len(sent.text.split()) > 5]
        
        return {
            "names": names,
            "topics_discussed": topics_discussed,
            "locations": locations,
            "crimes": crimes,
            "legal_terms": legal_terms,
            "evidence": evidence,
            "dates": dates,
            "comments": comments,
            "source_document": source_document
        }
    
    def add_to_graph(self, data):
        with self.neo4j_driver.session() as session:
            for name in data["names"]:
                session.run("MERGE (p:Person {name: $name, source: $source})", 
                            name=name, source=data["source_document"])
            
            for topic in data["topics_discussed"]:
                session.run("MERGE (t:TopicDiscussed {name: $topic, source: $source})", 
                            topic=topic, source=data["source_document"])
            
            for location in data["locations"]:
                session.run("MERGE (l:Location {name: $location, source: $source})", 
                            location=location, source=data["source_document"])
            
            for crime in data["crimes"]:
                session.run("MERGE (c:Crime {name: $crime, source: $source})", 
                            crime=crime, source=data["source_document"])
            
            for term in data["legal_terms"]:
                session.run("MERGE (lt:LegalTerm {name: $term, source: $source})", 
                            term=term, source=data["source_document"])
            
            for item in data["evidence"]:
                session.run("MERGE (e:Evidence {name: $item, source: $source})", 
                            item=item, source=data["source_document"])
            
            for date in data["dates"]:
                session.run("MERGE (d:Date {value: $date, source: $source})", 
                            date=date, source=data["source_document"])
            
            for comment in data["comments"]:
                result = session.run(
                    """
                    CREATE (c:Comment {text: $comment, source: $source})
                    WITH c
                    MATCH (p:Person) WHERE p.name IN $names AND p.source = $source
                    MERGE (p)-[:MENTIONED_IN]->(c)
                    WITH c
                    MATCH (t:TopicDiscussed) WHERE t.name IN $topics AND t.source = $source
                    MERGE (t)-[:MENTIONED_IN]->(c)
                    WITH c
                    MATCH (l:Location) WHERE l.name IN $locations AND l.source = $source
                    MERGE (l)-[:MENTIONED_IN]->(c)
                    WITH c
                    MATCH (cr:Crime) WHERE cr.name IN $crimes AND cr.source = $source
                    MERGE (cr)-[:MENTIONED_IN]->(c)
                    WITH c
                    MATCH (lt:LegalTerm) WHERE lt.name IN $legal_terms AND lt.source = $source
                    MERGE (lt)-[:MENTIONED_IN]->(c)
                    WITH c
                    MATCH (e:Evidence) WHERE e.name IN $evidence AND e.source = $source
                    MERGE (e)-[:MENTIONED_IN]->(c)
                    RETURN c
                    """,
                    comment=comment, 
                    source=data["source_document"],
                    names=data["names"],
                    topics=data["topics_discussed"],
                    locations=data["locations"],
                    crimes=data["crimes"],
                    legal_terms=data["legal_terms"],
                    evidence=data["evidence"]
                )

    def write_to_csv(self, data, output_csv_path):
        with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data.keys())
            if not csvfile.tell():  # Write header only on the first file
                writer.writeheader()
            writer.writerow(data)

    def process_document(self, file_path, output_csv_path):
        _, file_extension = os.path.splitext(file_path)
        
        if file_extension.lower() == '.pdf':
            text = self.parse_pdf(file_path)
        elif file_extension.lower() == '.docx':
            text = self.parse_docx(file_path)
        elif file_extension.lower() == '.txt':
            text = self.parse_txt(file_path)
        elif file_extension.lower() == '.doc':
            text = self.parse_doc(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        data = self.extract_information(text, os.path.basename(file_path))
        self.add_to_graph(data)
        self.write_to_csv(data, output_csv_path)
        return f"Successfully processed: {file_path}"

    def process_folder(self, folder_path, output_csv_path):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    futures.append(executor.submit(self.process_document, file_path, output_csv_path))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    print(result)
                except Exception as e:
                    print(f"Error processing file: {str(e)}")

# Usage
parser = CriminalLawDocumentParser()
parser.process_folder("D:/01_Court_Information/Test", "output.csv")
