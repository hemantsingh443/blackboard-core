"""
Coding Agent

This example demonstrates how to build a data analysis agent using:
- CodeInterpreterWorker: For executing Python code in a sandbox
- A custom data loading utility

The agent can:
1. Load CSV/JSON data files
2. Analyze data using Pandas
3. Generate visualizations
4. Output insights

Usage:
    # Ensure Docker is running (for secure sandbox)
    docker ps
    
    # Run the example
    python examples/coding_agent.py

Requirements:
    pip install blackboard-core[stdlib]
"""

import asyncio
import os
from pathlib import Path

from blackboard import Orchestrator, Blackboard, Status
from blackboard.protocols import Worker, WorkerInput, WorkerOutput
from blackboard.state import Artifact
from blackboard.llm import OpenAIClient


# =============================================================================
# Sample Data Generator
# =============================================================================

def create_sample_data():
    """Create a sample CSV file for analysis."""
    sample_csv = """date,product,sales,region,customer_count
2024-01-01,Widget A,1500,North,45
2024-01-01,Widget B,2300,South,67
2024-01-01,Widget C,890,East,23
2024-01-02,Widget A,1750,North,52
2024-01-02,Widget B,2100,South,61
2024-01-02,Widget C,920,West,28
2024-01-03,Widget A,1600,North,48
2024-01-03,Widget B,2450,South,72
2024-01-03,Widget C,1050,East,31
2024-01-04,Widget A,1850,West,55
2024-01-04,Widget B,2200,South,65
2024-01-04,Widget C,780,North,21
2024-01-05,Widget A,2100,North,63
2024-01-05,Widget B,2600,East,78
2024-01-05,Widget C,1150,South,34
"""
    
    data_path = Path("./sample_sales_data.csv")
    data_path.write_text(sample_csv)
    return str(data_path.absolute())


# =============================================================================
# Custom Workers
# =============================================================================

class DataAnalysisPlanner(Worker):
    """
    Plans data analysis steps based on the goal.
    
    This worker reads the data file and suggests analysis approaches.
    """
    
    name = "DataAnalysisPlanner"
    description = "Analyzes data structure and suggests analysis approaches"
    parallel_safe = True
    
    async def run(self, state: Blackboard, inputs: WorkerInput = None) -> WorkerOutput:
        # Check for data file in state metadata
        data_path = state.metadata.get("data_path")
        
        if not data_path or not Path(data_path).exists():
            return WorkerOutput(
                metadata={"error": "No data file found. Set state.metadata['data_path']"}
            )
        
        # Read first few lines to understand structure
        with open(data_path, 'r') as f:
            preview = f.read(1000)
        
        plan = f"""# Data Analysis Plan

## Data Preview
```
{preview}
```

## Suggested Analyses
1. **Summary Statistics**: Calculate mean, median, min, max for numeric columns
2. **Time Trend**: Analyze sales trends over time
3. **Regional Comparison**: Compare performance across regions
4. **Product Performance**: Rank products by sales
5. **Visualization**: Create bar charts and line plots

## Recommended Code Steps
1. Load data with pandas
2. Clean and validate data
3. Generate summary statistics
4. Create visualizations
5. Write insights
"""
        
        return WorkerOutput(
            artifact=Artifact(
                type="analysis_plan",
                content=plan,
                creator=self.name,
                metadata={"data_path": data_path}
            )
        )


# =============================================================================
# Main Application
# =============================================================================

def create_orchestrator(data_path: str):
    """Factory function to create the coding agent orchestrator."""
    from blackboard.stdlib import CodeInterpreterWorker
    
    # Initialize LLM
    llm = OpenAIClient(model="gpt-4o-mini")
    
    # Initialize workers
    workers = [
        DataAnalysisPlanner(),
        CodeInterpreterWorker(
            allowed_packages=["pandas", "matplotlib", "numpy"],
            timeout=60
        ),
    ]
    
    orchestrator = Orchestrator(
        llm=llm,
        workers=workers,
        verbose=True,
        auto_save_path="./coding_agent_session.json"
    )
    
    return orchestrator


async def main():
    """Run the coding agent."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Create sample data
    print("Creating sample data...")
    data_path = create_sample_data()
    print(f"Sample data created at: {data_path}")
    
    # Create orchestrator
    orchestrator = create_orchestrator(data_path)
    
    # Define the analysis goal
    goal = f"""
    Analyze the sales data in {data_path}.
    
    Tasks:
    1. Load the CSV file using pandas
    2. Calculate summary statistics (total sales, average per product, etc.)
    3. Find the top-performing product and region
    4. Create a simple bar chart showing sales by product
    5. Write a brief insights summary
    
    Save any charts to files and include the analysis results.
    """
    
    # Create initial state with data path
    state = Blackboard(goal=goal)
    state.metadata["data_path"] = data_path
    
    print("=" * 60)
    print("CODING AGENT - DATA ANALYSIS")
    print("=" * 60)
    print(f"Data: {data_path}")
    print("=" * 60)
    print()
    
    # Run the orchestrator
    result = await orchestrator.run(state=state, max_steps=10)
    
    # Display results
    print()
    print("=" * 60)
    print(f"STATUS: {result.status.value}")
    print(f"STEPS: {result.step_count}")
    print(f"ARTIFACTS: {len(result.artifacts)}")
    print("=" * 60)
    
    # Display code execution results
    for artifact in result.artifacts:
        if artifact.type == "code_result":
            print(f"\n📊 Code Output:\n{artifact.content[:1000]}")
    
    return result


if __name__ == "__main__":
    asyncio.run(main())
