import spacy
from gliner_spacy.pipeline import GlinerSpacy
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from fastapi import FastAPI
from pydantic import BaseModel
import ray

# Initialize Ray and enable the dashboard
ray.init(include_dashboard=True)

# Labels for entity recognition
labels = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION", "AGE", "DOB", "DATEOFBIRTH"]

# Initialize the Spacy model with GLINER
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(
    "gliner_spacy", config={"gliner_model": "urchade/gliner_multi_pii-v1", "labels": labels}
)

# Custom NLP Engine for Presidio
class LoadedSpacyNlpEngine(SpacyNlpEngine):
    def __init__(self, loaded_spacy_model):
        super().__init__()
        self.nlp = {"en": loaded_spacy_model}

loaded_nlp_engine = LoadedSpacyNlpEngine(loaded_spacy_model=nlp)

# Presidio Analyzer and Anonymizer
analyzer = AnalyzerEngine(nlp_engine=loaded_nlp_engine)
anonymizer = AnonymizerEngine()

# Define a FastAPI app
app = FastAPI()

# Pydantic model for request body
class TextRequest(BaseModel):
    text: str

# Ray remote function for entity analysis
@ray.remote
def analyze_text(text: str):
    return analyzer.analyze(text=text, entities=labels, language="en")

# Define the endpoint for anonymizing text
@app.post("/anonymize")
def anonymize_text(request: TextRequest):
    text = request.text

    # Use Ray to perform entity analysis in parallel
    analyzer_results = ray.get(analyze_text.remote(text))

    # Redact the identified PII data
    pii_sanitized_text = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators={
            "PERSON": OperatorConfig("replace", {"new_value": "[REDACTED NAME]"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[REDACTED PHONE NUMBER]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[REDACTED EMAIL]"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "[REDACTED PLACE]"}),
            "AGE": OperatorConfig("replace", {"new_value": "[REDACTED AGE]"}),
            "DOB": OperatorConfig("replace", {"new_value": "[REDACTED DOB]"}),
            "DATEOFBIRTH": OperatorConfig("replace", {"new_value": "[REDACTED DOB]"})
        },
    )
    return {"anonymized_text": pii_sanitized_text.text}

# Run the app with Uvicorn if this script is the main program
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

