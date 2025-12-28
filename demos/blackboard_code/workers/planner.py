"""
Planner Worker - Analyzes tasks and creates execution plans.
"""

from blackboard import Worker, WorkerOutput, Artifact, Blackboard
from blackboard.protocols import WorkerInput
import openai
from demos.blackboard_code.config import MODEL, OPENROUTER_API_KEY, OPENROUTER_HEADERS, MAX_TOKENS


class PlannerInput(WorkerInput):
    """Input schema for PlannerWorker."""
    task: str = ""


class PlannerWorker(Worker):
    """
    Analyzes user tasks and creates step-by-step plans.
    
    This worker breaks down complex requests into actionable steps
    that other workers can execute.
    """
    
    name = "PlannerWorker"
    description = "Analyzes tasks and creates step-by-step execution plans"
    input_schema = PlannerInput
    
    def __init__(self):
        self.client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )
    
    async def run(self, state: Blackboard, inputs: WorkerInput = None) -> WorkerOutput:
        task = inputs.task if inputs and hasattr(inputs, 'task') and inputs.task else state.goal
        
        # Get existing files from artifacts
        existing_files = []
        for art in state.artifacts:
            if art.type == "file":
                existing_files.append(art.metadata.get("path", art.id))
        
        files_context = ""
        if existing_files:
            files_context = f"\nExisting project files: {', '.join(existing_files)}"
        
        prompt = f"""You are a senior software architect planning a project.

Task: {task}
{files_context}

Create a detailed step-by-step plan. For each step specify:
1. What file to create or edit
2. What the file should contain (brief description)
3. Any commands to run

Format your plan as:
## Step 1: [Action]
- File: [filename]
- Description: [what to do]
- Command: [optional command to run]

## Step 2: ...

Be specific and actionable. Keep the plan to 3-6 steps for a small project.
"""

        response = await self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a software architect. Create clear, actionable plans."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS,
            extra_headers=OPENROUTER_HEADERS
        )
        
        plan = response.choices[0].message.content
        
        return WorkerOutput(
            artifact=Artifact(
                type="plan",
                content=plan,
                creator=self.name,
                metadata={"task": task}
            )
        )
