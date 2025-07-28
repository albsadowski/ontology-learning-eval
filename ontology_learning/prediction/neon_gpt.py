"""
NeOn-GPT reproduction.

"""

from functools import partial
from pathlib import Path
from typing import Final, TypedDict

from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import StrOutputParser

from ..config import chat_model
from ..fix_pipeline import fix_ontology
from ..strip_pipeline import strip_ontology
from ..get_answer import get_answer
from ..tasks import Task
from ..utils import debuglog


REQUIREMENTS_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=["input"],
    template="""
You are a knowledge engineer. Your task is to create an ontology to represent the key information
in a specific M&A contract fragment.

Based on the **Contract Fragment** provided below, help me:
1.  Define the purpose of the ontology (i.e., what key knowledge should it capture from this specific text?).
2.  Define the scope of the ontology (i.e., what are the boundaries of the knowledge within this text?).
3.  Identify the key requirements for representing this specific fragment.

**Contract Fragment:**
{input}
""",
)


COMPETENCY_QUESTIONS_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=["requirements", "input"],
    template="""
Based on the **Contract Fragment** and **Ontology Requirements** below, generate a numbered
list of 5 competency questions. These questions must be answerable using only the information
present in the provided fragment.

**Ontology Requirements:**
{requirements}

**Contract Fragment:**
{input}
""",
)


CONCEPTUAL_MODEL_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=["competency_questions", "input"],
    template="""
Given the **Contract Fragment** and **Competency Questions** below, your task is to
create a conceptual model.

First, extract the main entities and their properties that are explicitly mentioned in the fragment.
Second, generate the corresponding subject-relation-object triples for the extracted concepts.

The entire conceptual model must be based *only* on the information available in the provided text.

**Competency Questions:**
{competency_questions}

**Contract Fragment:**
{input}
""",
)


IMPLEMENTATION_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=["conceptual_model"],
    template="""
Based on the conceptual model below (containing entities, properties, and triples), implement a
basic ontology serialized in Turtle syntax. Define the classes and properties.

**Conceptual Model:**
{conceptual_model}
""",
)


ENRICHMENT_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=["ontology_draft", "input"],
    template="""
Enrich the following Turtle ontology draft. Perform these three steps in order:
1.  Add meaningful inverse object properties where appropriate.
2.  Add `rdfs:comment` annotations with natural language descriptions for the main classes and properties.
3.  Populate the ontology with specific instances found **only within the Contract Fragment provided below**.
    For example, if the fragment mentions 'Buyer's consent shall not be unreasonably withheld', you might
    create an instance of a 'ConsentLimitation' class.

Provide only the final, enriched Turtle code.

**Initial Ontology Draft:**
```turtle
{ontology_draft}
```

**Contract Fragment:**
{input}
""",
)


class State(TypedDict):
    model: str
    input: str
    requirements: str | None
    competency_questions: str | None
    conceptual_model: str | None
    ontology_draft: str | None
    populated_ontology: str | None


@debuglog
def prepare_requirements(state: State, **kwargs):
    llm = chat_model(state["model"]) | StrOutputParser()
    res = llm.invoke(
        REQUIREMENTS_PROMPT.format(
            input=state["input"],
        ),
    )
    return {"requirements": res}


@debuglog
def prepare_competency_questions(state: State, **kwargs):
    llm = chat_model(state["model"]) | StrOutputParser()
    res = llm.invoke(
        COMPETENCY_QUESTIONS_PROMPT.format(
            input=state["input"],
            requirements=state["requirements"],
        ),
    )
    return {"competency_questions": res}


@debuglog
def prepare_conceptual_model(state: State, **kwargs):
    llm = chat_model(state["model"]) | StrOutputParser()
    res = llm.invoke(
        CONCEPTUAL_MODEL_PROMPT.format(
            input=state["input"],
            competency_questions=state["competency_questions"],
        ),
    )
    return {"conceptual_model": res}


@debuglog
def prepare_ontology_draft(state: State, **kwargs):
    llm = chat_model(state["model"]) | StrOutputParser()
    res = llm.invoke(
        IMPLEMENTATION_PROMPT.format(
            conceptual_model=state["conceptual_model"],
        ),
    )
    return {"ontology_draft": res}


@debuglog
def enrich_draft(state: State, **kwargs):
    llm = chat_model(state["model"]) | StrOutputParser()
    res = llm.invoke(
        ENRICHMENT_PROMPT.format(
            input=state["input"],
            ontology_draft=state["ontology_draft"],
        ),
    )
    return {"populated_ontology": res}


def graph(debug: bool):
    builder = StateGraph(State)
    ctx = {"debug": debug}

    builder.add_node("prepare_requirements", partial(prepare_requirements, **ctx))
    builder.add_node(
        "prepare_competency_questions", partial(prepare_competency_questions, **ctx)
    )
    builder.add_node(
        "prepare_conceptual_model", partial(prepare_conceptual_model, **ctx)
    )
    builder.add_node("prepare_ontology_draft", partial(prepare_ontology_draft, **ctx))
    builder.add_node("enrich_draft", partial(enrich_draft, **ctx))

    builder.add_edge(START, "prepare_requirements")
    builder.add_edge("prepare_requirements", "prepare_competency_questions")
    builder.add_edge("prepare_competency_questions", "prepare_conceptual_model")
    builder.add_edge("prepare_conceptual_model", "prepare_ontology_draft")
    builder.add_edge("prepare_ontology_draft", "enrich_draft")
    builder.add_edge("enrich_draft", END)

    return builder.compile()


def predict(
    model: str,
    task: Task,
    input: str,
    debug: bool,
    fix: bool,
    strip: bool,
    task_dir: Path | None,
) -> str | None:
    populated_ontology = graph(debug).invoke(
        {
            "model": model,
            "input": input,
        }
    )["populated_ontology"]
    if not populated_ontology:
        print(f"err: failed to construct ontology for {input}")
        return None
    if fix:
        populated_ontology = fix_ontology(model, input, populated_ontology, debug)
    if strip:
        populated_ontology = strip_ontology(model, populated_ontology, debug=debug)
    if task_dir:
        with open(task_dir / "neongpt", "w") as f:
            f.write(populated_ontology)
    return get_answer(model, task, populated_ontology, debug=debug)
