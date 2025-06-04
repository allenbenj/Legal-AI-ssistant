import spacy
import csv
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

class SexualAssaultCaseAnalyzer:
    def __init__(self, nlp, output_csv_path):
        self.nlp = nlp
        self.output_csv_path = output_csv_path
        self.initialize_pipeline()
        self.initialize_csv()
        self.sentence_classifier = self.train_sentence_classifier()

    def initialize_pipeline(self):
        # Add custom entities for sexual assault cases
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        patterns = [
            {"label": "ASSAULT_TYPE", "pattern": [{"LOWER": {"IN": ["rape", "molestation", "groping", "sexual abuse", "sexual assault"]}}]},
            {"label": "LEGAL_TERM", "pattern": [{"LOWER": {"IN": ["consent", "victim", "perpetrator", "testimony", "evidence", "witness"]}}]},
            {"label": "EVIDENCE", "pattern": [{"LOWER": {"IN": ["dna", "rape kit", "clothing", "text messages", "photos", "video"]}}]},
            {"label": "LOCATION", "pattern": [{"ENT_TYPE": "GPE"}, {"LOWER": {"IN": ["bedroom", "party", "dorm", "car", "office"]}}]},
            {"label": "TIME", "pattern": [{"ENT_TYPE": "TIME"}]},
            {"label": "DRUG", "pattern": [{"LOWER": {"IN": ["alcohol", "rohypnol", "ghb", "ketamine", "cocaine"]}}]}
        ]
        ruler.add_patterns(patterns)

    def initialize_csv(self):
        headers = ['case_id', 'speaker', 'recipient', 'timestamp', 'content', 'sentence_type', 'names', 'assault_types', 'legal_terms', 'evidence', 'locations', 'times', 'drugs', 'comments']
        with open(self.output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()

    def train_sentence_classifier(self):
        # This is a simplified example. In a real-world scenario, you'd use a larger, labeled dataset.
        sentences = [
            "The victim reported the incident the next morning.",
            "The accused denies all allegations of wrongdoing.",
            "DNA evidence was collected from the scene.",
            "Witnesses saw the victim leave the party with the accused.",
            "The victim's statement contains inconsistencies.",
            "The accused has no prior criminal record."
        ]
        labels = ['victim_statement', 'defense_statement', 'evidence', 'witness_account', 'analysis', 'background']

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)
        y = labels

        classifier = MultinomialNB()
        classifier.fit(X, y)

        return (vectorizer, classifier)

    def classify_sentence(self, sentence):
        vectorizer, classifier = self.sentence_classifier
        X = vectorizer.transform([sentence])
        return classifier.predict(X)[0]

    def process_text(self, text, case_id, speaker, recipient, timestamp):
        doc = self.nlp(text)
        sentences = list(doc.sents)

        for sentence in sentences:
            data = {
                'case_id': case_id,
                'speaker': speaker,
                'recipient': recipient,
                'timestamp': timestamp,
                'content': sentence.text,
                'sentence_type': self.classify_sentence(sentence.text),
                'names': [],
                'assault_types': [],
                'legal_terms': [],
                'evidence': [],
                'locations': [],
                'times': [],
                'drugs': [],
                'comments': []
            }

            for ent in sentence.ents:
                if ent.label_ == "ASSAULT_TYPE":
                    data['assault_types'].append(ent.text)
                elif ent.label_ == "LEGAL_TERM":
                    data['legal_terms'].append(ent.text)
                elif ent.label_ == "EVIDENCE":
                    data['evidence'].append(ent.text)
                elif ent.label_ == "PERSON":
                    data['names'].append(ent.text)
                elif ent.label_ == "LOCATION":
                    data['locations'].append(ent.text)
                elif ent.label_ == "TIME":
                    data['times'].append(ent.text)
                elif ent.label_ == "DRUG":
                    data['drugs'].append(ent.text)

            self.write_to_csv(data)

    def write_to_csv(self, data):
        with open(self.output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data.keys())
            writer.writerow(data)

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize analyzer
analyzer = SexualAssaultCaseAnalyzer(nlp, "sexual_assault_case_output.csv")

# Example usage
case_statement1 = "The victim, Jane Doe, reported that on June 15th at approximately 11 PM, she was sexually assaulted by John Smith in his apartment after a party. She stated that she had consumed several alcoholic drinks and felt disoriented."
case_statement2 = "The accused, John Smith, denies all allegations. He claims that any sexual contact was consensual and that both parties had been drinking at the party."

analyzer.process_text(case_statement1, "Case2023-001", "Investigator", "Case File", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
analyzer.process_text(case_statement2, "Case2023-001", "Defense Attorney", "Case File", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
