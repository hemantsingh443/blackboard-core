"""
Coder Worker - Generates code based on requirements.
"""

from blackboard import Worker, WorkerOutput, Artifact, Blackboard
from blackboard.protocols import WorkerInput
import openai
from ..config import MODEL, OPENROUTER_API_KEY, OPENROUTER_HEADERS, MAX_TOKENS


class CoderInput(WorkerInput):
    """Input schema for CoderWorker."""
    filename: str = ""
    requirements: str = ""


class CoderWorker(Worker):
    """
    Generates code for specific files based on requirements.
    
    Takes a filename and requirements, produces complete code.
    """
    
    name = "CoderWorker"
    description = "Generates code files based on requirements"
    input_schema = CoderInput
    
    def __init__(self):
        self.client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )
    
    async def run(self, state: Blackboard, inputs: WorkerInput = None) -> WorkerOutput:
        filename = inputs.filename if inputs and hasattr(inputs, 'filename') else "code.txt"
        requirements = inputs.requirements if inputs and hasattr(inputs, 'requirements') else ""
        
        if not filename:
            filename = "code.txt"
        
        # Get context from existing files
        context = ""
        for art in state.artifacts:
            if art.type == "file":
                path = art.metadata.get("path", "")
                context += f"\n### {path}\n```\n{art.content[:500]}...\n```\n"
        
        # Determine language from extension
        ext = filename.split(".")[-1] if "." in filename else "txt"
        lang_map = {
            "py": "Python",
            "js": "JavaScript", 
            "ts": "TypeScript",
            "html": "HTML",
            "css": "CSS",
            "json": "JSON",
            "md": "Markdown"
        }
        language = lang_map.get(ext, "code")
        
        prompt = f"""Generate complete {language} code for: {filename}

Requirements:
{requirements}

{f"Existing project files for context:{context}" if context else ""}

IMPORTANT:
- Output ONLY the code, no explanations
- Make it complete and functional
- Include all necessary imports
- Add helpful comments

Output the code:"""

        response = await self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": f"You are an expert {language} developer. Generate complete, working code."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS,
            extra_headers=OPENROUTER_HEADERS
        )
        
        code = response.choices[0].message.content
        
        # Clean up code (remove markdown fences if present)
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        return WorkerOutput(
            artifact=Artifact(
                type="code",
                content=code,
                creator=self.name,
                metadata={
                    "filename": filename,
                    "language": language,
                    "requirements": requirements
                }
            )
        )
