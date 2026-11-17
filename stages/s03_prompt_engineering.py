"""
STAGE 3 - Prompt engineering: same task, three prompts, compare the results.

Task: classify an operator's incident report.
  v1 naive       -> free text, inconsistent, impossible to parse in code
  v2 engineered  -> role + definitions + output format
  v3 production  -> v2 + few-shot examples + structured output validated by Pydantic
Run: python -m stages.s03_prompt_engineering
"""
from app.llm import get_chat_model
from app.prompts import INCIDENT_CLASSIFIER_PROMPT
from app.schemas import IncidentClassification

model = get_chat_model()
REPORTS = [
    "Gas holder pressure 27 mbar and CHP engine stopped, flare not lit",
    "Feeding pump making noise but flow is normal",
    "Operator felt dizzy near the desulfurisation unit",
    "Biogas production 8% lower than yesterday, no alarms active",
]

PROMPT_V1 = "Classify this incident: {report}"

PROMPT_V2 = """You are a safety engineer at a biogas plant.
Classify the incident with:
- severity: low | medium | high | critical (critical = any risk to people)
- area: digester | gas_handling | chp_engine | feeding | instrumentation | other
Answer exactly in the format: severity=<value>; area=<value>
Incident: {report}"""

for report in REPORTS:
    print("=" * 100, f"\nREPORT: {report}")
    print("\n[v1 naive]\n", model.invoke(PROMPT_V1.format(report=report)).content[:400])
    print("\n[v2 engineered]\n", model.invoke(PROMPT_V2.format(report=report)).content)
    # v3: the schema is sent to the model as a tool/JSON schema; the reply is parsed and validated.
    result: IncidentClassification = model.with_structured_output(IncidentClassification).invoke(
        [("system", INCIDENT_CLASSIFIER_PROMPT), ("human", f"Report: {report}")])
    print("\n[v3 structured]\n", result.model_dump())

print("""
What to notice:
 - v1 answers change in length and wording on every run -> you cannot build software on top of it.
 - v2 is consistent, but you still parse strings by hand.
 - v3 returns a typed Python object: ready for an API, a database or an alert rule.
Exercise: add a 4th report that is ambiguous and improve the definitions until v3 classifies it well.
""")
