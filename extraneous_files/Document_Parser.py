import os
import PyPDF2
from docx import Document
import docx2txt
import spacy
import re
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import concurrent.futures
from functools import partial

class CriminalLawDocumentParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "Monster_12#$"))
        self.tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        self.lda_model = LatentDirichletAllocation(n_components=10, random_state=42)

        # Add custom entities for criminal law domain
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        patterns = [
            {"label": "CRIME", "pattern": [{"LOWER": {"IN": ["murder", "theft", "assault", "fraud", "burglary"]}}]},
            {"label": "LEGAL_TERM", "pattern": [{"LOWER": {"IN": ["defendant", "plaintiff", "verdict", "sentence", "bail", "parole"]}}]},
            {"label": "EVIDENCE", "pattern": [{"LOWER": {"IN": ["dna", "fingerprint", "witness", "testimony", "exhibit"]}}]}
        ]
        ruler.add_patterns(patterns)

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
        tfidf_topics = [feature_names[i] for i in tfidf_matrix.sum(axis=0).argsort()[0, -10:][0]]
        
        # Topic Modeling using LDA
        lda_output = self.lda_model.fit_transform(tfidf_matrix)
        lda_topics = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-5 - 1:-1]]
            lda_topics.extend(top_words)
        
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

    def process_document(self, file_path):
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
        return f"Successfully processed: {file_path}"

    def process_folder(self, folder_path):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    futures.append(executor.submit(self.process_document, file_path))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    print(result)
                except Exception as e:
                    print(f"Error processing file: {str(e)}")

# Usage
parser = CriminalLawDocumentParser()
parser.process_folder("D:/01_Court_Information/Test")
