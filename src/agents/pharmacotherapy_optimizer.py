from typing import Any

# Try to import crewai components
try:
    from crewai import Agent, Task
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False

def create_pharmacotherapy_agent(llm: Any, tools: list) -> Any:
    """Creates the Clinical Pharmacist & Guideline Specialist Agent."""
    if not HAS_CREWAI:
        return None
        
    return Agent(
        role="Clinical Pharmacist & Guideline Specialist",
        goal="Audit patient active medications against ADA Standards of Care guidelines to flag dosage safety conflicts and clinical inertia.",
        backstory="A clinical pharmacist trained in guidelines adherence. You query clinical databases and recommend optimal patient-tailored drug transitions.",
        verbose=True,
        allow_delegation=False,
        tools=tools,
        llm=llm
    )

def create_pharmacotherapy_task(agent: Any) -> Any:
    """Creates the CrewAI task for the Pharmacotherapy Agent."""
    if not HAS_CREWAI:
        return None
        
    return Task(
        description="Review the patient's active drugs and biomarkers (such as renal levels). Use the ADA Guideline Query Search Tool "
                    "to audit the profile against official ADA guidelines. Identify contraindications or clinical inertia. "
                    "CRITICAL: Whenever you recommend a medication change or flag a contraindication, you MUST cite the exact [Page X] or (ID: ...) provided by the Search Tool.",
        expected_output="A pharmacotherapy audit detail identifying ADA guideline clashes, metformin dosage safety levels, and recommended drug classes, with strict data lineage citations (e.g. [Page X]).",
        agent=agent
    )
