"""
Research Assistant - Blackboard-Core
=====================================

A real AI-powered research assistant using Google Gemini.
Demonstrates the full power of the Blackboard Pattern with actual LLM calls.

Prerequisites:
    1. Create .env file with: GEMINI_API_KEY=your_key
    2. pip install python-dotenv

Run with:
    python examples/research_assistant.py "Your research topic here"
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    print("   python-dotenv not installed. Install with: pip install python-dotenv")
    print("   Or set GEMINI_API_KEY environment variable directly.")

from blackboard import (
    Orchestrator, Worker, WorkerOutput, WorkerInput,
    Artifact, Feedback, Blackboard
)
from blackboard.llm import LiteLLMClient
from blackboard.tui import watch


# =============================================================================
# Configuration
# =============================================================================

GEMINI_MODEL = "gemini/gemini-3-flash-preview"


# =============================================================================
# Workers (Class-based for reliable execution)
# =============================================================================

class Researcher(Worker):
    """Researches a topic and gathers key information."""
    
    name = "Researcher"
    description = "Researches topics and gathers key facts, statistics, and recent developments"
    
    def __init__(self, llm: "LiteLLMClient"):
        self.llm = llm
    
    async def run(self, state: Blackboard, inputs: Optional[WorkerInput] = None) -> WorkerOutput:
        instructions = inputs.instructions if inputs else "Research the topic"
        
        prompt = f"""You are a research assistant. Research the following topic and provide a comprehensive brief.

Topic: {state.goal}
Instructions: {instructions}

Provide:
1. **Key Facts**: 3-5 important facts with sources if possible
2. **Recent Developments**: What's happened in 2023-2024
3. **Key Players**: Major companies, researchers, or organizations
4. **Trends**: Current direction and momentum

Format as clear markdown with headers."""

        response = await self.llm.agenerate(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        return WorkerOutput(
            artifact=Artifact(
                type="research",
                content=content,
                creator=self.name
            )
        )


class Analyst(Worker):
    """Analyzes research and extracts insights."""
    
    name = "Analyst"
    description = "Analyzes research findings and extracts actionable insights"
    
    def __init__(self, llm: "LiteLLMClient"):
        self.llm = llm
    
    async def run(self, state: Blackboard, inputs: Optional[WorkerInput] = None) -> WorkerOutput:
        research = state.get_last_artifact("research")
        if not research:
            return WorkerOutput(
                artifact=Artifact(
                    type="analysis",
                    content="No research to analyze yet.",
                    creator=self.name
                )
            )
        
        prompt = f"""Analyze this research and provide strategic insights.

Research:
{research.content[:3000]}

Provide:
1. **Key Trends**: What patterns emerge from this data?
2. **Implications**: What does this mean for the future?
3. **Opportunities**: What actions could be taken?
4. **Risks**: What challenges or risks exist?

Be concise and actionable."""

        response = await self.llm.agenerate(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        return WorkerOutput(
            artifact=Artifact(
                type="analysis",
                content=content,
                creator=self.name
            )
        )


class Writer(Worker):
    """Writes polished content from research and analysis."""
    
    name = "Writer"
    description = "Writes polished, engaging summaries based on research and analysis"
    
    def __init__(self, llm: "LiteLLMClient"):
        self.llm = llm
    
    async def run(self, state: Blackboard, inputs: Optional[WorkerInput] = None) -> WorkerOutput:
        # Gather context from previous artifacts
        context_parts = []
        for artifact in state.artifacts[-3:]:
            context_parts.append(f"[{artifact.type.upper()}]\n{artifact.content[:1500]}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""Write a professional executive summary based on this research and analysis.

{context}

Requirements:
- Start with a compelling headline
- Include an executive summary (2-3 sentences)
- Present key findings with evidence
- End with actionable recommendations
- Length: 300-400 words
- Use markdown formatting"""

        response = await self.llm.agenerate(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        return WorkerOutput(
            artifact=Artifact(
                type="article",
                content=content,
                creator=self.name
            )
        )


class Critic(Worker):
    """Reviews content for quality."""
    
    name = "Critic"
    description = "Reviews content for accuracy, clarity, and completeness"
    
    def __init__(self, llm: "LiteLLMClient"):
        self.llm = llm
    
    async def run(self, state: Blackboard, inputs: Optional[WorkerInput] = None) -> WorkerOutput:
        article = state.get_last_artifact()
        if not article:
            return WorkerOutput(
                feedback=Feedback(
                    source=self.name,
                    critique="No content to review",
                    passed=False
                )
            )
        
        prompt = f"""Review this article for quality:

{article.content}

Score on:
1. Clarity (1-10)
2. Accuracy (1-10)  
3. Actionability (1-10)

If average score >= 7, approve it. Otherwise, suggest specific improvements.
Reply with: APPROVED or NEEDS_REVISION followed by your critique."""

        response = await self.llm.agenerate(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        passed = "APPROVED" in content.upper()
        
        return WorkerOutput(
            feedback=Feedback(
                source=self.name,
                artifact_id=article.id,
                critique=content,
                passed=passed
            )
        )


# =============================================================================
# Main Execution
# =============================================================================

async def main():
    # Get topic from command line or use default
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = "The impact of AI on software development in 2024"
    
    print("\n" + "="*70)
    print("  🔬 BLACKBOARD RESEARCH ASSISTANT")
    print("  Powered by Google Gemini")
    print("="*70)
    print(f"\n  Topic: {topic}\n")
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found!")
        print("   Create a .env file with: GEMINI_API_KEY=your_key")
        return
    
    # Create LLM client
    try:
        llm = LiteLLMClient(model=GEMINI_MODEL)
        print(f"  ✅ Connected to {GEMINI_MODEL}\n")
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        return
    
    # Create workers (pass LLM to each)
    workers = [
        Researcher(llm),
        Analyst(llm),
        Writer(llm),
        Critic(llm)
    ]
    
    # Create orchestrator
    orchestrator = Orchestrator(
        llm=llm,
        workers=workers,
        verbose=False
    )
    
    # Run with TUI
    try:
        result = watch(
            orchestrator,
            goal=f"Research and write a comprehensive report on: {topic}",
            max_steps=10
        )
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Display results with rich markdown formatting
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text
    
    console = Console()
    
    console.print()
    console.print(Panel.fit(
        Text.assemble(
            ("📊 RESEARCH COMPLETE\n\n", "bold cyan"),
            ("Status: ", "bold"),
            (result.status.value.upper(), "green" if result.status.value == "done" else "red"),
            (f"\nSteps: {result.step_count}", ""),
            (f"\nArtifacts: {len(result.artifacts)}", "")
        ),
        border_style="cyan"
    ))
    
    # Show final article with markdown rendering
    final = result.get_last_artifact("article") or result.get_last_artifact()
    if final:
        console.print()
        console.print(Panel(
            Markdown(str(final.content)[:3000]),
            title="[bold green]📝 Final Article[/bold green]",
            border_style="green",
            padding=(1, 2)
        ))
    
    # Save session
    result.save_to_json("research_session.json")
    console.print(f"\n  💾 Session saved to research_session.json")
    console.print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
