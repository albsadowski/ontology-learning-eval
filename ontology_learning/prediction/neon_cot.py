from pathlib import Path
from typing import Final

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..config import chat_model
from ..fix_pipeline import fix_ontology
from ..strip_pipeline import strip_ontology
from ..get_answer import get_answer
from ..tasks import Task
from ..utils import debuglog


PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=["input"],
    template="""
You are an experienced Knowledge Engineer specializing in legal document
ontologies. Your task is to create a comprehensive ontology from the
following M&A contract fragment using a systematic approach.

Follow these steps:

**Step 1: Domain Analysis**
Analyze the M&A contract fragment below and identify:
- The domain scope (what aspects of M&A transactions are covered)
- Key legal and business concepts present
- The purpose this ontology should serve

**Step 2: Generate Competency Questions**
Create 5-8 competency questions that this ontology should be able to answer. These should cover the main information in the contract fragment.

Example format:
- "What entities are involved in this transaction?"
- "What are the financial terms mentioned?"
- "What conditions or requirements are specified?"

**Step 3: Extract Entities and Relationships**
From your competency questions, extract:
- Key entities (companies, financial terms, legal concepts, dates, etc.)
- Properties and relationships between entities
- Provide output in this format: {{"Entity": ["Company", "Financial_Term"], "Property": ["hasValue", "involves"]}}

**Step 4: Create Conceptual Model**
Generate subject-relation-object triples expressing the relationships:
- Company hasRole Acquirer
- Transaction hasValue Purchase_Price
- Contract contains Condition

**Step 5: Implement Formal Ontology**
Convert your conceptual model into a complete ontology in Turtle syntax, including:
- Class definitions using rdfs:subClassOf
- Object properties with appropriate domains and ranges
- Data properties for literal values
- Individual instances where applicable
- Natural language descriptions using rdfs:comment

Contract Fragment:
{input}

Generate the complete ontology following all steps above:
""",
)


@debuglog
def create_populated_ontology(model: str, input: str, **kwargs):
    llm = chat_model(model) | StrOutputParser()
    return llm.invoke(PROMPT.format(input=input))


def predict(
    model: str,
    task: Task,
    input: str,
    debug: bool,
    fix: bool,
    strip: bool,
    task_dir: Path | None,
) -> str | None:
    populated_ontology = create_populated_ontology(model, input, debug=debug)
    if fix:
        populated_ontology = fix_ontology(model, input, populated_ontology, debug)
    if strip:
        populated_ontology = strip_ontology(model, populated_ontology, debug=debug)
    if task_dir:
        with open(task_dir / "neoncot", "w") as f:
            f.write(populated_ontology)
    return get_answer(model, task, populated_ontology, debug=debug)
