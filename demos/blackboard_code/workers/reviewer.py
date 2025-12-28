"""
Reviewer Worker - Reviews code for quality and correctness.
"""

from typing import Optional
from blackboard import Worker, WorkerOutput, Feedback, Blackboard
from blackboard.protocols import WorkerInput
import openai
from demos.blackboard_code.config import MODEL, OPENROUTER_API_KEY, OPENROUTER_HEADERS, MAX_TOKENS


class ReviewerInput(WorkerInput):
    """Input schema for ReviewerWorker."""
    artifact_id: str = ""


class ReviewerWorker(Worker):
    """
    Reviews generated code for quality, correctness, and best practices.
    
    Acts as a code reviewer/critic to ensure high quality output.
    """
    
    name = "ReviewerWorker"
    description = "Reviews code for quality, bugs, and best practices"
    input_schema = ReviewerInput
    
    def __init__(self):
        self.client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )
    
    async def run(self, state: Blackboard, inputs: WorkerInput = None) -> WorkerOutput:
        # Find the artifact to review
        artifact_id = inputs.artifact_id if inputs and hasattr(inputs, 'artifact_id') else None
        
        code_artifact = None
        if artifact_id:
            code_artifact = state.get_artifact(artifact_id)
        
        # If no specific artifact, review the most recent code
        if not code_artifact:
            for art in reversed(state.artifacts):
                if art.type == "code":
                    code_artifact = art
                    break
        
        if not code_artifact:
            return WorkerOutput(feedback_text="No code to review")
        
        filename = code_artifact.metadata.get("filename", "code")
        code = code_artifact.content
        
        prompt = f"""Review this code for: {filename}

```
{code[:3000]}
```

Provide a brief code review:
1. Overall Assessment (PASS/NEEDS_WORK)
2. Strengths (1-2 points)
3. Issues (if any, with specific fixes)
4. Security considerations (if applicable)

Be concise but thorough. If the code is good, say PASS and note the strengths.
If it needs work, say NEEDS_WORK and explain the specific issues.
"""

        response = await self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a senior code reviewer. Be constructive and specific."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            extra_headers=OPENROUTER_HEADERS
        )
        
        review = response.choices[0].message.content
        
        # Determine if it passed
        passed = "PASS" in review.upper() and "NEEDS_WORK" not in review.upper()
        
        return WorkerOutput(
            feedback=Feedback(
                source=self.name,
                content=review,
                rating=5 if passed else 3,
                metadata={
                    "artifact_id": code_artifact.id,
                    "filename": filename,
                    "passed": passed
                }
            )
        )
