"""
Prompt templates for the SciFact claim verification pipeline.
"""

VERIFICATION_PROMPT_TEMPLATE = """You are a scientific fact-checker. Given a biomedical claim and \
retrieved evidence from scientific abstracts, determine whether the evidence \
SUPPORTS, REFUTES, or provides NOT_ENOUGH_INFO for the claim.
{few_shot_section}
## Claim
{claim}

## Retrieved Evidence
{evidence_block}

## Instructions
1. Analyze the claim against the provided evidence.
2. Output a JSON object with exactly these fields:
   - "label": one of "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"
   - "evidence_doc_ids": list of doc_ids that are most relevant
   - "explanation": a clear 2-4 sentence explanation in natural language \
justifying your verdict, referencing specific findings from the abstracts.
3. Choose NOT_ENOUGH_INFO if the retrieved abstracts do not directly address \
the claim, even if they are topically related.

Respond ONLY with valid JSON, no markdown fences, no extra text."""


# ---------------------------------------------------------------------------
# Few-shot examples — one per label
# ---------------------------------------------------------------------------

FEW_SHOT_BLOCK = """
## Examples

### Example 1 — SUPPORTS
**Claim:** Beta-blockers reduce mortality in patients with chronic heart failure.

**Retrieved Evidence:**
[Doc 10021]
A double-blind randomised controlled trial of carvedilol versus placebo in 2289 patients with chronic heart failure (LVEF ≤ 35%) found that carvedilol significantly reduced all-cause mortality (hazard ratio 0.65, 95% CI 0.52–0.81, p < 0.001). Secondary endpoints including cardiovascular death and hospitalisation for heart failure were also significantly reduced in the carvedilol group.

**Output:**
{"label": "SUPPORTS", "evidence_doc_ids": [10021], "explanation": "Doc 10021 reports a large RCT demonstrating that carvedilol (a beta-blocker) significantly reduced all-cause mortality (HR 0.65, p<0.001) compared to placebo in chronic heart failure patients. This directly supports the claim that beta-blockers reduce mortality in this population."}

---

### Example 2 — REFUTES
**Claim:** Vitamin E supplementation reduces the incidence of cardiovascular disease.

**Retrieved Evidence:**
[Doc 10045]
A meta-analysis of 9 randomised trials (n = 118,765) found no significant reduction in major cardiovascular events among participants receiving vitamin E supplementation compared to placebo (relative risk 1.03, 95% CI 0.97–1.09). A non-significant trend toward increased risk of haemorrhagic stroke was observed in the high-dose vitamin E subgroup.

**Output:**
{"label": "REFUTES", "evidence_doc_ids": [10045], "explanation": "Doc 10045 presents a large meta-analysis showing that vitamin E supplementation does not significantly reduce cardiovascular events (RR 1.03, 95% CI 0.97–1.09) and shows a trend toward increased haemorrhagic stroke risk. This evidence directly contradicts the claim that vitamin E reduces cardiovascular disease incidence."}

---

### Example 3 — NOT_ENOUGH_INFO
**Claim:** CRISPR-Cas9 gene editing cures sickle cell disease in adult patients.

**Retrieved Evidence:**
[Doc 10078]
This study characterises the molecular mechanism of CRISPR-Cas9-mediated correction of the HBB point mutation (E6V) in human haematopoietic stem cells (HSCs) in vitro. Edited HSCs maintained long-term self-renewal capacity and demonstrated restored haemoglobin expression in erythroid differentiation assays. No in vivo or clinical outcome data were reported.

**Output:**
{"label": "NOT_ENOUGH_INFO", "evidence_doc_ids": [], "explanation": "Doc 10078 describes in vitro CRISPR-Cas9 editing of the HBB mutation in stem cells, but explicitly reports no clinical outcomes and no data from adult patients. The abstract addresses a mechanistic preclinical step and cannot confirm or refute whether this approach cures sickle cell disease in adults."}

---
"""


def build_evidence_block(retrieved_docs: list[dict]) -> str:
    """
    Format retrieved documents into an evidence block for the prompt.

    Args:
        retrieved_docs: List of dicts with keys doc_id and text.

    Returns:
        Formatted string with each document labelled by its doc_id.
    """
    parts = [f"[Doc {doc['doc_id']}]\n{doc['text']}" for doc in retrieved_docs]
    return "\n\n".join(parts)


def build_verification_prompt(
    claim: str,
    retrieved_docs: list[dict],
    few_shot: bool = False,
) -> str:
    """
    Construct the full verification prompt for a single claim.

    Args:
        claim:          Biomedical claim string.
        retrieved_docs: Top-k retrieved documents from the FAISS index.
        few_shot:       If True, prepend three labelled examples (one per class)
                        to help the model calibrate its output format and
                        improve NOT_ENOUGH_INFO recall.

    Returns:
        Prompt string ready to send to the LLM.
    """
    evidence_block = build_evidence_block(retrieved_docs)
    few_shot_section = FEW_SHOT_BLOCK if few_shot else "\n"
    return VERIFICATION_PROMPT_TEMPLATE.format(
        few_shot_section=few_shot_section,
        claim=claim,
        evidence_block=evidence_block,
    )
