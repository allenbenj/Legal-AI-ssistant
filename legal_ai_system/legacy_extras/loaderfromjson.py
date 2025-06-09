import json
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

class GraphDatabaseBuilder:
    def __init__(self):
        self.neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "Monster_12#$"))
        self.tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english')
        self.lda_model = LatentDirichletAllocation(n_components=10, random_state=42)

    def clear_database(self):
        with self.neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared.")

    def create_graph_from_json(self, json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)

        # Process all documents for TF-IDF and LDA
        documents = [' '.join(doc['ngrams']) for doc in data]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        lda_output = self.lda_model.fit_transform(tfidf_matrix)

        with self.neo4j_driver.session() as session:
            for i, doc in enumerate(data):
                doc_id = doc['id']
                self.create_document_node(session, doc_id, doc['sentiment_score'], lda_output[i])
                
                for entity in doc['named_entities']:
                    self.create_entity_node(session, entity)
                    self.create_relationship(session, doc_id, entity, 'CONTAINS')
                
                for ngram in doc['ngrams']:
                    self.create_ngram_node(session, ngram)
                    self.create_relationship(session, doc_id, ngram, 'CONTAINS')
                    
                    for entity in doc['named_entities']:
                        if entity in ngram:
                            self.create_relationship(session, ngram, entity, 'ASSOCIATED_WITH')
                
                for word, pos in doc['pos_tags']:
                    if pos.startswith('VB'):  # Verb
                        for entity1 in doc['named_entities']:
                            for entity2 in doc['named_entities']:
                                if entity1 != entity2:
                                    self.create_action_relationship(session, entity1, entity2, word, pos)

            self.calculate_and_set_weights(session, data)

    def create_document_node(self, session, doc_id, sentiment_score, topics):
        query = (
            "CREATE (d:Document {id: $id, sentiment: $sentiment}) "
            "SET d.topics = $topics"
        )
        session.run(query, id=doc_id, sentiment=sentiment_score, topics=topics.tolist())

    def create_entity_node(self, session, entity):
        query = "MERGE (e:Entity {name: $name})"
        session.run(query, name=entity)

    def create_ngram_node(self, session, ngram):
        query = "MERGE (n:Ngram {text: $text})"
        session.run(query, text=ngram)

    def create_relationship(self, session, start_node, end_node, rel_type):
        query = f"MATCH (a), (b) WHERE a.id = $start_id AND b.name = $end_name OR b.text = $end_name CREATE (a)-[:{rel_type}]->(b)"
        session.run(query, start_id=start_node, end_name=end_node)

    def create_action_relationship(self, session, entity1, entity2, action, pos):
        query = (
            "MATCH (e1:Entity {name: $entity1}), (e2:Entity {name: $entity2}) "
            "MERGE (e1)-[r:ACTION {word: $action, pos: $pos}]->(e2)"
        )
        session.run(query, entity1=entity1, entity2=entity2, action=action, pos=pos)

    def calculate_and_set_weights(self, session, data):
        # Calculate co-occurrences
        co_occurrences = {}
        for doc in data:
            entities = doc['named_entities']
            for i in range(len(entities)):
                for j in range(i+1, len(entities)):
                    pair = tuple(sorted([entities[i], entities[j]]))
                    co_occurrences[pair] = co_occurrences.get(pair, 0) + 1

        # Set weights based on co-occurrences and sentiment
        for (entity1, entity2), weight in co_occurrences.items():
            query = (
                "MATCH (e1:Entity {name: $entity1})-[r]-(e2:Entity {name: $entity2}) "
                "WITH e1, e2, r, AVG(CASE WHEN (e1)-[:CONTAINS]-(d:Document) OR (e2)-[:CONTAINS]-(d:Document) THEN d.sentiment ELSE 0 END) AS avg_sentiment "
                "SET r.weight = $weight * (1 + avg_sentiment)"
            )
            session.run(query, entity1=entity1, entity2=entity2, weight=weight)

    def close(self):
        self.neo4j_driver.close()

# Usage
builder = GraphDatabaseBuilder()

# Ask user if they want to clear the database
clear_db = input("Do you want to clear the existing database before loading new data? (yes/no): ").lower()
if clear_db == 'yes':
    builder.clear_database()

builder.create_graph_from_json('D:/01_Court_Information/document_extraction_non-clan.json')

# Example query to check the result
with builder.neo4j_driver.session() as session:
    result = session.run("MATCH (n) RETURN labels(n) AS label, count(*) AS count")
    for record in result:
        print(f"{record['label'][0]}s: {record['count']}")

builder.close()
