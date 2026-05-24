from typing import Any

# Try to import crewai components
try:
    from crewai import Agent, Task
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False

def create_clinical_reasoner_agent(llm: Any) -> Any:
    """Creates the Explainable AI (XAI) Clinical Lead Agent."""
    if not HAS_CREWAI:
        return None
        
    return Agent(
        role="Explainable AI (XAI) Clinical Lead",
        goal="Reconcile inputs from all prior agents, draft the causal evidence chain, and write a unified, risk-mitigated treatment plan.",
        backstory="A medical director dedicated to evidence-based medicine, safety protocols, and transparent clinical explanations.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

def create_clinical_reasoner_task(agent: Any) -> Any:
    """Creates the CrewAI task for the Clinical Reasoner Agent."""
    if not HAS_CREWAI:
        return None
        
    return Task(
        description="Synthesize the findings from all prior analyses. Resolve any clinical contradictions. Generate a beautifully structured "
                    "DiaTrace.AI Clinical Decision Report. Outline demographics, math-based complications, drug safety warnings, a robust "
                    "four-point Causal Evidence Chain, and actionable guideline-aligned treatment suggestions. "
                    "CRITICAL: You MUST carry over any ADA [Page X] citations or (ID: ...) references from the Pharmacotherapy Agent directly into the Causal Evidence Chain.",
        expected_output="A complete, professional, human-auditable clinical decision support report in markdown format, strictly citing source guidelines in the evidence chain.",
        agent=agent
    )
