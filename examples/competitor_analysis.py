"""
Competitor Analysis Agent

This example demonstrates how to build a competitor analysis bot using:
- WebSearchWorker: For finding competitor information
- BrowserWorker: For scraping pricing pages
- A custom Writer worker: For generating the report

Usage:
    # Set environment variables
    export TAVILY_API_KEY="your_key"
    
    # Run the example
    python examples/competitor_analysis.py

Requirements:
    pip install blackboard-core[stdlib,browser]
"""

import asyncio
import os
from typing import List

from blackboard import Orchestrator, Blackboard, Status
from blackboard.protocols import Worker, WorkerInput, WorkerOutput
from blackboard.state import Artifact
from blackboard.llm import OpenAIClient


# =============================================================================
# Custom Workers
# =============================================================================

class CompetitorAnalysisWriter(Worker):
    """
    Writes a competitor analysis report based on gathered research.
    
    Input Schema:
        - competitors: List of competitor names
        - focus_areas: What to analyze (pricing, features, etc.)
    """
    
    name = "CompetitorAnalysisWriter"
    description = "Writes a structured competitor analysis report from research data"
    parallel_safe = True
    
    def __init__(self, llm=None):
        self.llm = llm
    
    async def run(self, state: Blackboard, inputs: WorkerInput = None) -> WorkerOutput:
        # Gather research from artifacts
        research_artifacts = [
            a for a in state.artifacts 
            if a.type in ("search_results", "webpage_content", "text")
        ]
        
        if not research_artifacts:
            return WorkerOutput(
                metadata={"error": "No research data found. Perform web search first."}
            )
        
        # Combine research
        research_text = "\n\n---\n\n".join([
            f"## {a.metadata.get('source', 'Research')}\n{a.content[:2000]}"
            for a in research_artifacts[:5]  # Limit to 5 artifacts
        ])
        
        prompt = f"""Based on the following research, write a comprehensive competitor analysis report.

RESEARCH DATA:
{research_text}

ORIGINAL GOAL: {state.goal}

Write a structured report with:
1. Executive Summary
2. Competitor Overview (table format if possible)
3. Pricing Comparison
4. Feature Analysis
5. Recommendations

Use Markdown formatting."""

        if self.llm:
            response = await self.llm.generate(prompt)
            content = response if isinstance(response, str) else response.content
        else:
            content = f"# Competitor Analysis Report\n\n*Generated from {len(research_artifacts)} research sources*\n\n{research_text}"
        
        return WorkerOutput(
            artifact=Artifact(
                type="report",
                content=content,
                creator=self.name,
                metadata={
                    "format": "markdown",
                    "sources": len(research_artifacts)
                }
            )
        )


# =============================================================================
# Main Application
# =============================================================================

def create_orchestrator():
    """Factory function to create the competitor analysis orchestrator."""
    from blackboard.stdlib import WebSearchWorker, BrowserWorker
    
    # Initialize LLM (uses OPENAI_API_KEY from environment)
    llm = OpenAIClient(model="gpt-4o-mini")
    
    # Initialize workers
    workers = [
        WebSearchWorker(),
        BrowserWorker(
            headless=True,
            close_browser_after_run=True  # Clean up after each page
        ),
        CompetitorAnalysisWriter(llm=llm),
    ]
    
    return Orchestrator(
        llm=llm,
        workers=workers,
        verbose=True,
        auto_save_path="./competitor_analysis_session.json"
    )


async def main():
    """Run the competitor analysis agent."""
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your_key'")
        return
    
    if not os.getenv("TAVILY_API_KEY") and not os.getenv("SERPER_API_KEY"):
        print("Warning: No search API key found (TAVILY_API_KEY or SERPER_API_KEY)")
        print("WebSearchWorker may not work without a search provider")
    
    # Create orchestrator
    orchestrator = create_orchestrator()
    
    # Define the analysis goal
    goal = """
    Analyze the top 3 AI coding assistant tools (GitHub Copilot, Cursor, Codeium).
    
    For each competitor:
    1. Search for their pricing information
    2. Visit their pricing pages to get accurate data
    3. Note key features and differentiators
    
    Then write a comprehensive comparison report with pricing tables and recommendations.
    """
    
    print("=" * 60)
    print("COMPETITOR ANALYSIS AGENT")
    print("=" * 60)
    print(f"Goal: {goal.strip()}")
    print("=" * 60)
    print()
    
    # Run the orchestrator
    result = await orchestrator.run(goal=goal, max_steps=15)
    
    # Display results
    print()
    print("=" * 60)
    print(f"STATUS: {result.status.value}")
    print(f"STEPS: {result.step_count}")
    print(f"ARTIFACTS: {len(result.artifacts)}")
    print("=" * 60)
    
    # Display the report if generated
    report = result.get_last_artifact()
    if report and report.type == "report":
        print("\n📊 FINAL REPORT:\n")
        print(report.content)
    else:
        print("\nNo report generated. Check the session file for details.")
    
    return result


if __name__ == "__main__":
    asyncio.run(main())
