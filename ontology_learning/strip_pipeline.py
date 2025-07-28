from typing import Final

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .config import chat_model
from .utils import debuglog


PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=["ontology"],
    template="""
You are a Turtle syntax specialist. Your task is to extract and clean up a valid Turtle ontology from the provided messy output.

## Instructions:

1. **Extract only valid Turtle syntax** from the input text
2. **Remove all explanatory text, step descriptions, and analysis**
3. **Keep only ontology-level comments** (those that describe classes, properties, or the ontology itself)
4. **Remove procedural comments** (like "# Step 1:", "# Following the analysis", etc.)
5. **Ensure proper Turtle formatting** with correct prefixes, syntax, and structure
6. **Validate that all statements are proper RDF triples**
7. **CRITICAL: Never modify the actual Turtle content** - only extract and
   clean formatting. Only fix obvious syntax errors (missing periods,
   incorrect punctuation, malformed URIs). Do not change class names,
   property names, relationships, or semantic content.

## Output Requirements:

- Start with proper prefix declarations
- Include only valid Turtle statements (@prefix, class definitions, property definitions, instances)
- Maintain proper indentation and formatting
- Ensure all URIs, literals, and syntax are correctly formatted
- Remove any markdown formatting or step indicators
- Keep descriptive rdfs:comment statements that explain the ontology concepts

## Input:
{ontology}

## Output:
Provide only the clean, valid Turtle ontology below:
""",
)


@debuglog
def strip_ontology(model: str, ontology: str, **kwargs) -> str:
    llm = chat_model("gpt-4.1-mini") | StrOutputParser()
    return llm.invoke(
        PROMPT.format(
            ontology=ontology,
        )
    )
