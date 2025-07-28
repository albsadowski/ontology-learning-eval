from functools import partial
from typing import Final, TypedDict

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from .config import chat_model
from .utils import debuglog


CHECK_CONSISTENCY_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=["ontology"],
    template="""
You are a logic reasoner analyzing an ontology for logical inconsistencies.
Examine the ontology below and identify any logical contradictions, invalid relationships, or inconsistent constraints.

Look for common inconsistency patterns:
- Contradictory class restrictions (e.g., something that must be both X and not-X)
- Invalid property domain/range assignments  
- Circular dependencies in class hierarchies
- Inconsistent data type assignments
- Conflicting cardinality constraints
- Entities that violate their own class definitions

Ontology:
{ontology}

If you find inconsistencies, list them clearly with explanations. If the ontology is logically consistent, respond with "CONSISTENT".

Format inconsistencies as:
- Inconsistency 1: [Description of the logical contradiction]
- Inconsistency 2: [Description of another issue]
""",
)


FIX_INCONSISTENCIES_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=["input", "ontology", "inconsistencies"],
    template="""
You are an experienced Knowledge Engineer fixing logical inconsistencies in an ontology.
The ontology below has been identified as having logical inconsistencies.

Original contract fragment for context:
{input}

Current ontology with inconsistencies:
{ontology}

Inconsistencies found:
{inconsistencies}

Your task: Fix all identified inconsistencies while preserving the semantic content from the original
contract fragment. Make minimal changes necessary to resolve the logical contradictions.

Provide the corrected ontology in complete Turtle syntax:
""",
)


DETECT_PITFALLS_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=["ontology"],
    template="""
You are an ontology validation expert analyzing an ontology for common modeling pitfalls. Examine the ontology below and
identify potential issues that could affect its quality and usability.

Look for these common ontology pitfalls:
- **Missing disjointness**: Classes that should be mutually exclusive but aren't declared as disjoint
- **Missing domain/range**: Properties without proper domain and range restrictions
- **Equivalent classes**: Classes that represent the same concept but aren't declared equivalent
- **Missing inverse relationships**: Properties that should have inverse properties
- **Circular definitions**: Classes defined in terms of themselves
- **Overly generic classes**: Classes that are too broad and need specialization
- **Missing annotations**: Classes/properties lacking human-readable descriptions
- **Inconsistent naming**: Inconsistent naming conventions across the ontology

Ontology:
{ontology}

For each pitfall found, report in this format:
- **Pitfall Type**: [Type from list above]
- **Description**: [Specific issue found]
- **Severity**: [Critical/Important/Minor]

If no pitfalls are found, respond with "NO PITFALLS DETECTED".
""",
)


FIX_PITFALLS_PROMPT: Final[PromptTemplate] = PromptTemplate(
    input_variables=["input", "ontology", "pitfalls"],
    template="""
You are an experienced Knowledge Engineer fixing ontology modeling pitfalls. The ontology below has been
analyzed and several pitfalls have been identified.

Original contract fragment for context:
{input}

Current ontology with pitfalls:
{ontology}

Pitfalls identified:
{pitfalls}

Your task: Fix all identified pitfalls while maintaining the semantic integrity of the ontology.
Focus on Critical and Important pitfalls first.

For each pitfall:
- Add missing disjointness axioms where appropriate
- Define proper domain/range for properties  
- Add missing inverse relationships
- Include necessary annotations and descriptions
- Ensure consistent naming conventions

Provide the corrected ontology in complete Turtle syntax.
Return exclusively the ontology, no need to explain the fixes.
""",
)


class FixOntologyState(TypedDict):
    model: str
    input: str
    ontology: str
    inconsistencies: str | None
    pitfalls: str | None


@debuglog
def check_consistency(state: FixOntologyState, **kwargs):
    llm = chat_model(state["model"]) | StrOutputParser()
    res = llm.invoke(
        CHECK_CONSISTENCY_PROMPT.format(
            ontology=state["ontology"],
        ),
    )
    return {"inconsistencies": res}


@debuglog
def fix_inconsistencies(state: FixOntologyState, **kwargs):
    llm = chat_model(state["model"]) | StrOutputParser()
    res = llm.invoke(
        FIX_INCONSISTENCIES_PROMPT.format(
            input=state["input"],
            ontology=state["ontology"],
            inconsistencies=state["inconsistencies"],
        ),
    )
    return {"ontology": res}


@debuglog
def detect_pitfalls(state: FixOntologyState, **kwargs):
    llm = chat_model(state["model"]) | StrOutputParser()
    res = llm.invoke(
        DETECT_PITFALLS_PROMPT.format(
            ontology=state["ontology"],
        ),
    )
    return {"pitfalls": res}


@debuglog
def fix_pitfalls(state: FixOntologyState, **kwargs):
    llm = chat_model(state["model"]) | StrOutputParser()
    res = llm.invoke(
        FIX_PITFALLS_PROMPT.format(
            input=state["input"],
            ontology=state["ontology"],
            pitfalls=state["pitfalls"],
        )
    )
    return {"ontology": res}


def graph(debug: bool):
    builder = StateGraph(FixOntologyState)
    ctx = {"debug": debug}

    builder.add_node("check_consistency", partial(check_consistency, **ctx))
    builder.add_node("fix_inconsistencies", partial(fix_inconsistencies, **ctx))
    builder.add_node("detect_pitfalls", partial(detect_pitfalls, **ctx))
    builder.add_node("fix_pitfalls", partial(fix_pitfalls, **ctx))

    builder.add_edge(START, "check_consistency")
    builder.add_conditional_edges(
        "check_consistency",
        lambda state: "consistent"
        if "CONSISTENT" in state["inconsistencies"]
        else "inconsistent",
        {
            "consistent": "detect_pitfalls",
            "inconsistent": "fix_inconsistencies",
        },
    )
    builder.add_edge("fix_inconsistencies", "detect_pitfalls")
    builder.add_conditional_edges(
        "detect_pitfalls",
        lambda state: "not found"
        if "NO PITFALLS DETECTED" in state["pitfalls"]
        else "found",
        {
            "found": "fix_pitfalls",
            "not found": END,
        },
    )
    builder.add_edge("fix_pitfalls", END)

    return builder.compile()


def fix_ontology(model: str, input: str, ontology: str, debug: bool) -> str:
    res = graph(debug).invoke(
        {
            "model": model,
            "input": input,
            "ontology": ontology,
        }
    )
    return res["ontology"]
