from dataclasses import dataclass
from pathlib import Path
from typing import Final


@dataclass(frozen=True, kw_only=True)
class Task:
    name: str
    path: Path
    instruction: str
    question: str
    answers: list[tuple[str, str]]


def maud(task: str):
    return Path("./legalbench/data") / f"maud_{task}"


DEFAULT_INSTRUCTION: Final[str] = (
    "Read the segment of a merger agreement and answer the multiple-choice question by choosing the option that best characterizes the agreement."
)


TASKS: Final[dict[str, Task]] = {
    "t1": Task(
        name="t1",
        path=maud("ability_to_consummate_concept_is_subject_to_mae_carveouts"),
        instruction=DEFAULT_INSTRUCTION,
        question='Is the "ability to consummate" concept subject to Material Adverse Effect (MAE) carveouts?',
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t2": Task(
        name="t2",
        path=maud("accuracy_of_fundamental_target_rws_bringdown_standard"),
        instruction=DEFAULT_INSTRUCTION,
        question="How accurate must the fundamental representations and warranties be according to the bring down provision?",
        answers=[
            ("A", "Accurate at another materiality standard (e.g., hybrid standard)"),
            ("B", "Accurate in all material respects"),
            ("C", "Accurate in all respects"),
        ],
    ),
    "t3": Task(
        name="t3",
        path=maud(
            "accuracy_of_target_capitalization_rw_(outstanding_shares)_bringdown_standard_answer"
        ),
        instruction=DEFAULT_INSTRUCTION,
        question="How accurate must the capitalization representations and warranties be according to the bring down provision?",
        answers=[
            ("A", "Accurate in all material respects"),
            ("B", "Accurate in all respects"),
            ("C", "Accurate in all respects with below-threshold carveout"),
            ("D", "Accurate in all respects with de minimis exception"),
        ],
    ),
    "t4": Task(
        name="t4",
        path=maud("accuracy_of_target_general_rw_bringdown_timing_answer"),
        instruction=DEFAULT_INSTRUCTION,
        question="When are representations and warranties required to be made according to the bring down provision?",
        answers=[
            ("A", "At Closing Only"),
            ("B", "At Signing & At Closing"),
        ],
    ),
    "t5": Task(
        name="t5",
        path=maud("additional_matching_rights_period_for_modifications_(cor)"),
        instruction=DEFAULT_INSTRUCTION,
        question="How long is the additional matching rights period for modifications in case the board changes its recommendation?",
        answers=[
            ("A", "2 business days or less"),
            ("B", "3 business days"),
            ("C", "3 days"),
            ("D", "4 business days"),
            ("E", "5 business days"),
            ("F", "> 5 business days"),
            ("G", "None"),
        ],
    ),
    "t6": Task(
        name="t6",
        path=maud(
            "application_of_buyer_consent_requirement_(negative_interim_covenant)"
        ),
        instruction=DEFAULT_INSTRUCTION,
        question="What negative covenants does the requirement of Buyer consent apply to?",
        answers=[
            ("A", "Applies only to specified negative covenants"),
            ("B", "Applies to all negative covenants"),
        ],
    ),
    "t7": Task(
        name="t7",
        path=maud("buyer_consent_requirement_(ordinary_course)"),
        instruction=DEFAULT_INSTRUCTION,
        question="In case the Buyer's consent for the acquired company's ordinary business operations is required, are there any limitations on the Buyer's right to condition, withhold, or delay their consent?",
        answers=[
            (
                "A",
                "Yes. Consent may not be unreasonably withheld, conditioned or delayed.",
            ),
            ("B", "No."),
        ],
    ),
    "t8": Task(
        name="t8",
        path=maud("change_in_law__subject_to_disproportionate_impact_modifier"),
        instruction=DEFAULT_INSTRUCTION,
        question="Do changes in law that have disproportionate impact qualify for Material Adverse Effect (MAE)?",
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t9": Task(
        name="t9",
        path=maud(
            "changes_in_gaap_or_other_accounting_principles__subject_to_disproportionate_impact_modifier"
        ),
        instruction=DEFAULT_INSTRUCTION,
        question="Do changes in GAAP or other accounting principles that have disproportionate impact qualify for Material Adverse Effect (MAE)?",
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t10": Task(
        name="t10",
        path=maud("cor_permitted_in_response_to_intervening_event"),
        instruction=DEFAULT_INSTRUCTION,
        question="Is Change of Recommendation permitted in response to an intervening event?",
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t11": Task(
        name="t11",
        path=maud("cor_permitted_with_board_fiduciary_determination_only"),
        instruction=DEFAULT_INSTRUCTION,
        question="Is Change of Recommendation permitted as long as the board determines that such change is required to fulfill its fiduciary obligations?",
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t12": Task(
        name="t12",
        path=maud("cor_standard_(intervening_event)"),
        instruction=DEFAULT_INSTRUCTION,
        question="What standard should the board follow when determining whether to change its recommendation in response to an intervening event?",
        answers=[
            ("A", '"Breach" of fiduciary duties'),
            ("B", '"Inconsistent" with fiduciary duties'),
            ("C", '"Reasonably likely/expected breach" of fiduciary duties'),
            (
                "D",
                '"Reasonably likely/expected to be inconsistent" with fiduciary duties',
            ),
            ("E", '"Reasonably likely/expected violation" of fiduciary duties'),
            ("F", '"Required to comply" with fiduciary duties'),
            ("G", '"Violation" of fiduciary duties'),
            ("H", "More likely than not violate fiduciary duties"),
            ("I", "Other specified standard"),
        ],
    ),
    "t13": Task(
        name="t13",
        path=maud("cor_standard_(superior_offer)"),
        instruction=DEFAULT_INSTRUCTION,
        question="What standard should the board follow when determining whether to change its recommendation in connection with a superior offer?",
        answers=[
            ("A", '"Breach" of fiduciary duties'),
            ("B", '"Inconsistent" with fiduciary duties'),
            ("C", '"Reasonably likely/expected breach" of fiduciary duties'),
            (
                "D",
                '"Reasonably likely/expected to be inconsistent" with fiduciary duties',
            ),
            ("E", '"Reasonably likely/expected violation" of fiduciary duties'),
            ("F", '"Required to comply" with fiduciary duties'),
            ("G", '"Violation" of fiduciary duties'),
            ("H", "More likely than not violate fiduciary duties"),
            ("I", "None"),
            ("J", "Other specified standard"),
        ],
    ),
    "t14": Task(
        name="t14",
        path=maud("definition_contains_knowledge_requirement_-_answer"),
        instruction=DEFAULT_INSTRUCTION,
        question='What is the knowledge requirement in the definition of "Intervening Event"?',
        answers=[
            (
                "A",
                "Known, but consequences unknown or not reasonably foreseeable, at signing",
            ),
            ("B", "Known, but consequences unknown, at signing"),
            ("C", "Not known and not reasonably foreseeable at signing"),
            ("D", "Not known at signing"),
        ],
    ),
    "t15": Task(
        name="t15",
        path=maud("definition_includes_asset_deals"),
        instruction=DEFAULT_INSTRUCTION,
        question="What qualifies as a superior offer in terms of asset deals?",
        answers=[
            ("A", '"All or substantially all"'),
            ("B", "50%"),
            ("C", 'Greater than 50% but not "all or substantially all"'),
            ("D", "Less than 50%"),
        ],
    ),
    "t16": Task(
        name="t16",
        path=maud("definition_includes_stock_deals"),
        instruction=DEFAULT_INSTRUCTION,
        question="What qualifies as a superior offer in terms of stock deals?",
        answers=[
            ("A", '"All or substantially all"'),
            ("B", "50%"),
            ("C", 'Greater than 50% but not "all or substantially all"'),
            ("D", "Less than 50%"),
        ],
    ),
    "t17": Task(
        name="t17",
        path=maud("fiduciary_exception__board_determination_standard"),
        instruction=DEFAULT_INSTRUCTION,
        question="Under what circumstances could the Board take actions on a different acquisition proposal notwithstanding the no-shop provision?",
        answers=[
            (
                "A",
                'If failure to take actions would lead to "breach" of fiduciary duties',
            ),
            (
                "B",
                'If failure to take actions would be "inconsistent" with fiduciary duties',
            ),
            (
                "C",
                'If failure to take actions would lead to "reasonably likely/expected breach" of fiduciary duties',
            ),
            (
                "D",
                'If failure to take actions would lead to "reasonably likely/expected to be inconsistent" with fiduciary duties',
            ),
            (
                "E",
                'If failure to take actions would lead to "reasonably likely/expected violation" of fiduciary duties',
            ),
            (
                "F",
                'If taking such actions is "required to comply" with fiduciary duties',
            ),
            (
                "G",
                'If failure to take actions would lead to "violation" of fiduciary duties',
            ),
            ("H", "Under no circumstances could the Board do so."),
            ("I", "Other circumstances"),
        ],
    ),
    "t18": Task(
        name="t18",
        path=maud("fiduciary_exception_board_determination_trigger_(no_shop)"),
        instruction=DEFAULT_INSTRUCTION,
        question="What type of offer could the Board take actions on notwithstanding the no-shop provision?",
        answers=[
            ("A", "Acquisition Proposal only"),
            (
                "B",
                "Superior Offer, or Acquisition Proposal reasonably likely/expected to result in a Superior Offer",
            ),
        ],
    ),
    "t19": Task(
        name="t19",
        path=maud("financial_point_of_view_is_the_sole_consideration"),
        instruction=DEFAULT_INSTRUCTION,
        question='Is "financial point of view" the sole consideration when determining whether an offer is superior?',
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t20": Task(
        name="t20",
        path=maud("fls_(mae)_standard"),
        instruction=DEFAULT_INSTRUCTION,
        question="What is the Forward Looking Standard (FLS) with respect to Material Adverse Effect (MAE)?",
        answers=[
            ("A", "'Could' (reasonably) be expected to"),
            ("B", "'Would'"),
            ("C", "'Would' (reasonably) be expected to"),
            ("D", "No"),
            ("E", "Other forward-looking standard"),
        ],
    ),
    "t21": Task(
        name="t21",
        path=maud(
            "general_economic_and_financial_conditions_subject_to_disproportionate_impact_modifier"
        ),
        instruction=DEFAULT_INSTRUCTION,
        question="Do changes caused by general economic and financial conditions that have disproportionate impact qualify for Material Adverse Effect (MAE)?",
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t22": Task(
        name="t22",
        path=maud("includes_consistent_with_past_practice"),
        instruction=DEFAULT_INSTRUCTION,
        question='Does the wording of the Efforts Covenant clause include "consistent with past practice"?',
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t23": Task(
        name="t23",
        path=maud("initial_matching_rights_period_(cor)"),
        instruction=DEFAULT_INSTRUCTION,
        question="How long is the initial matching rights period in case the board changes its recommendation?",
        answers=[
            ("A", "2 business days or less"),
            ("B", "3 business days"),
            ("C", "3 calendar days"),
            ("D", "4 business days"),
            ("E", "4 calendar days"),
            ("F", "5 business days"),
            ("G", "Greater than 5 business days"),
        ],
    ),
    "t24": Task(
        name="t24",
        path=maud("initial_matching_rights_period_(ftr)"),
        instruction=DEFAULT_INSTRUCTION,
        question="How long is the initial matching rights period in connection with the Fiduciary Termination Right (FTR)?",
        answers=[
            ("A", "2 business days or less"),
            ("B", "3 business days"),
            ("C", "3 calendar days"),
            ("D", "4 business days"),
            ("E", "4 calendar days"),
            ("F", "5 business days"),
            ("G", "5 calendar days"),
            ("H", "Greater than 5 business days"),
        ],
    ),
    "t25": Task(
        name="t25",
        path=maud("intervening_event_-_required_to_occur_after_signing_-_answer"),
        instruction=DEFAULT_INSTRUCTION,
        question='Is an "Intervening Event" required to occur after signing?',
        answers=[
            ("A", "No. It may occur or arise prior to signing."),
            ("B", "Yes. It must occur or arise after signing."),
        ],
    ),
    "t26": Task(
        name="t26",
        path=maud("knowledge_definition"),
        instruction=DEFAULT_INSTRUCTION,
        question="What counts as Knowledge?",
        answers=[
            ("A", "Actual knowledge"),
            ("B", "Constructive knowledge"),
        ],
    ),
    "t27": Task(
        name="t27",
        path=maud(
            "liability_standard_for_no-shop_breach_by_target_non-do_representatives"
        ),
        instruction=DEFAULT_INSTRUCTION,
        question="What is the liability standard for no-shop breach by target non-DO representatives?",
        answers=[
            ("A", "Strict liability"),
            ("B", "Knowledge-based liability"),
        ],
    ),
    "t28": Task(
        name="t28",
        path=maud("ordinary_course_efforts_standard"),
        instruction=DEFAULT_INSTRUCTION,
        question="What is the efforts standard?",
        answers=[
            ("A", "Commercially reasonable efforts"),
            ("B", "Flat covenant (no efforts standard)"),
            ("C", "Reasonable best efforts"),
        ],
    ),
    "t29": Task(
        name="t29",
        path=maud(
            "pandemic_or_other_public_health_event__subject_to_disproportionate_impact_modifier"
        ),
        instruction=DEFAULT_INSTRUCTION,
        question="Do pandemics or other public health events have to have disproportionate impact to qualify for Material Adverse Effect (MAE)?",
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t30": Task(
        name="t30",
        path=maud(
            "pandemic_or_other_public_health_event_specific_reference_to_pandemic-related_governmental_responses_or_measures"
        ),
        instruction=DEFAULT_INSTRUCTION,
        question="Is there specific reference to pandemic-related governmental responses or measures in the clause that qualifies pandemics or other public health events for Material Adverse Effect (MAE)?",
        answers=[
            ("A", "No"),
            ("B", "Yes"),
        ],
    ),
    "t31": Task(
        name="t31",
        path=maud("relational_language_(mae)_applies_to"),
        instruction=DEFAULT_INSTRUCTION,
        question="What carveouts pertaining to Material Adverse Effect (MAE) does the relational language apply to?",
        answers=[
            ("A", "All MAE carveouts"),
            ("B", "No"),
            ("C", "Some MAE carveouts"),
        ],
    ),
    "t32": Task(
        name="t32",
        path=maud("specific_performance"),
        instruction=DEFAULT_INSTRUCTION,
        question="What is the wording of the Specific Performance clause regarding the parties' entitlement in the event of a contractual breach?",
        answers=[
            ("A", '"entitled to seek" specific performance'),
            ("B", '"entitled to" specific performance'),
        ],
    ),
    "t33": Task(
        name="t33",
        path=maud("tail_period_length"),
        instruction=DEFAULT_INSTRUCTION,
        question="How long is the Tail Period?",
        answers=[
            ("A", "12 months or longer"),
            ("B", "Other"),
            ("C", "within 12 months"),
            ("D", "within 6 months"),
            ("E", "within 9 months"),
        ],
    ),
    "t34": Task(
        name="t34",
        path=maud("type_of_consideration"),
        instruction=DEFAULT_INSTRUCTION,
        question="What type of consideration is specified in this agreement?",
        answers=[
            ("A", "All Cash"),
            ("B", "All Stock"),
            ("C", "Mixed Cash/Stock"),
            ("D", "Mixed Cash/Stock: Election"),
        ],
    ),
}
