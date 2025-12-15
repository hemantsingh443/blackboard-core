"""
Blackboard-Core Example: Simple Writer/Reviewer Loop

This example demonstrates the self-healing loop pattern:
1. Writer generates content
2. Reviewer checks it
3. If rejected, Writer tries again with feedback
4. Loop continues until approved or max_steps reached
"""

import asyncio
import logging

from blackboard import (
    Orchestrator, Worker, WorkerOutput,
    Artifact, Feedback, Blackboard, Status
)


# =============================================================================
# Mock LLM Client
# =============================================================================

class MockLLMClient:
    """
    A mock LLM that simulates supervisor decisions.
    
    In production, replace with a real LLM client:
        class OpenAIClient:
            def __init__(self, api_key: str):
                self.client = openai.OpenAI(api_key=api_key)
            
            def generate(self, prompt: str) -> str:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
    """
    
    def __init__(self):
        self.step = 0
    
    def generate(self, prompt: str) -> str:
        """Simulate supervisor decisions based on state."""
        self.step += 1
        
        # Decision logic based on what we see in the prompt
        if "Latest Artifact" not in prompt:
            # No artifact yet - call the Writer
            return '''{"reasoning": "No content yet, need to generate", "action": "call", "worker": "Writer", "instructions": "Generate a haiku about programming"}'''
        
        if "Latest Feedback" not in prompt:
            # Have artifact but no feedback - call Reviewer
            return '''{"reasoning": "Content exists, need review", "action": "call", "worker": "Reviewer", "instructions": "Check if the haiku follows 5-7-5 syllable structure"}'''
        
        if "passed: True" in prompt or "Passed: True" in prompt:
            # Passed review - done!
            return '''{"reasoning": "Content passed review", "action": "done"}'''
        
        if self.step > 4:
            # Too many attempts - done anyway
            return '''{"reasoning": "Multiple attempts made, accepting current version", "action": "done"}'''
        
        # Failed review - retry with feedback
        return '''{"reasoning": "Previous attempt failed, need to revise", "action": "call", "worker": "Writer", "instructions": "Revise the haiku based on feedback - ensure 5-7-5 syllable count"}'''


# =============================================================================
# Worker Definitions
# =============================================================================

class HaikuWriter(Worker):
    """Generates haiku poems."""
    
    name = "Writer"
    description = "Generates haiku poems based on instructions"
    
    def __init__(self):
        self.attempt = 0
    
    async def run(self, state: Blackboard) -> WorkerOutput:
        self.attempt += 1
        instructions = state.metadata.get("current_instructions", "")
        
        # Check if there's previous feedback to incorporate
        last_feedback = state.get_latest_feedback()
        
        if self.attempt == 1:
            # First attempt
            content = """Code flows like water
            Bugs emerge from the shadows
            Debug into night"""
        else:
            # Revised attempt (pretend we fixed based on feedback)
            content = """Fingers on the keys
            Algorithms come alive
            Silicon dreams wake"""
        
        print(f"   Writer (attempt {self.attempt}): Generated haiku")
        
        return WorkerOutput(
            artifact=Artifact(
                type="haiku",
                content=content.strip(),
                creator=self.name,
                metadata={"attempt": self.attempt}
            )
        )


class HaikuReviewer(Worker):
    """Reviews haiku poems for correctness."""
    
    name = "Reviewer"
    description = "Reviews haiku for syllable count and quality"
    
    def __init__(self, pass_on_attempt: int = 2):
        self.reviews_done = 0
        self.pass_on_attempt = pass_on_attempt
    
    async def run(self, state: Blackboard) -> WorkerOutput:
        self.reviews_done += 1
        last_artifact = state.get_last_artifact()
        
        if last_artifact is None:
            return WorkerOutput(
                feedback=Feedback(
                    source=self.name,
                    critique="No content to review",
                    passed=False
                )
            )
        
        # Simulate review logic
        if self.reviews_done >= self.pass_on_attempt:
            print(f"   Reviewer: Approved!")
            return WorkerOutput(
                feedback=Feedback(
                    source=self.name,
                    artifact_id=last_artifact.id,
                    critique="Well-structured haiku with proper syllable count (5-7-5). Evocative imagery.",
                    passed=True
                )
            )
        else:
            print(f"   Reviewer: Needs revision")
            return WorkerOutput(
                feedback=Feedback(
                    source=self.name,
                    artifact_id=last_artifact.id,
                    critique="Syllable count seems off in line 2. Please revise for 5-7-5 structure.",
                    passed=False
                )
            )


# =============================================================================
# Main Execution
# =============================================================================

async def main():
    print("=" * 60)
    print("Blackboard-Core Example: Haiku Writer/Reviewer")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
    
    # Initialize components
    llm = MockLLMClient()
    workers = [
        HaikuWriter(),
        HaikuReviewer(pass_on_attempt=2)  # Will pass on second review
    ]
    
    # Create orchestrator
    orchestrator = Orchestrator(
        llm=llm,
        workers=workers,
        verbose=True
    )
    
    # Run the loop (async)
    result = await orchestrator.run(
        goal="Write a haiku about programming",
        max_steps=10
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Status: {result.status.value}")
    print(f"Steps taken: {result.step_count}")
    print(f"Artifacts created: {len(result.artifacts)}")
    print(f"Feedback entries: {len(result.feedback)}")
    
    if result.artifacts:
        print("\n Final Haiku:")
        print("-" * 40)
        print(result.artifacts[-1].content)
        print("-" * 40)
    
    if result.feedback:
        final_feedback = result.feedback[-1]
        print(f"\n Final Review: {' Passed' if final_feedback.passed else ' Failed'}")
        print(f"   {final_feedback.critique}")


if __name__ == "__main__":
    asyncio.run(main())
