"""
Code Interpreter Worker

High-level wrapper around DockerSandbox for code execution.
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from pydantic import Field

from blackboard.protocols import Worker, WorkerInput, WorkerOutput
from blackboard.state import Blackboard, Artifact
from blackboard.sandbox import DockerSandbox, InsecureLocalExecutor, SandboxResult

logger = logging.getLogger("blackboard.stdlib.code")


class CodeInterpreterInput(WorkerInput):
    """Input schema for CodeInterpreterWorker."""
    code: str = Field(..., description="Python code to execute")
    packages: List[str] = Field(default_factory=list, description="Packages to install before execution")
    files: Dict[str, str] = Field(default_factory=dict, description="Files to create: {filename: content}")
    timeout: float = Field(default=60.0, description="Timeout in seconds")
    language: str = Field(default="python", description="Programming language (currently only python supported)")


class CodeInterpreterWorker(Worker):
    """
    Code execution worker with automatic package installation.
    
    Wraps DockerSandbox (preferred, secure) or InsecureLocalExecutor
    to execute code with dependency management.
    
    Args:
        name: Worker name (default: "CodeInterpreter")
        description: Worker description
        use_docker: Whether to use Docker sandbox (default: True if Docker available)
        docker_image: Docker image to use
        memory_limit: Memory limit for Docker container
        cpu_limit: CPU limit for Docker container
        network_enabled: Whether to allow network access
        
    Example:
        interpreter = CodeInterpreterWorker()
        orchestrator = Orchestrator(llm=my_llm, workers=[interpreter])
        
    Security:
        By default, uses DockerSandbox for full isolation.
        Falls back to InsecureLocalExecutor only if Docker is unavailable
        AND BLACKBOARD_ALLOW_UNSAFE_EXECUTION=1 is set.
    """
    
    name = "CodeInterpreter"
    description = "Executes Python code with package installation support"
    input_schema = CodeInterpreterInput
    parallel_safe = False  # Code execution has side effects
    
    def __init__(
        self,
        name: str = "CodeInterpreter",
        description: str = "Executes Python code with package installation support",
        use_docker: Optional[bool] = None,
        docker_image: str = "python:3.11-slim",
        memory_limit: str = "512m",
        cpu_limit: str = "1.0",
        network_enabled: bool = False,
        default_timeout: float = 60.0
    ):
        self.name = name
        self.description = description
        self.docker_image = docker_image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network_enabled = network_enabled
        self.default_timeout = default_timeout
        
        # Auto-detect Docker availability
        if use_docker is None:
            self._use_docker = self._check_docker_available()
        else:
            self._use_docker = use_docker
        
        self._sandbox = None
    
    def _check_docker_available(self) -> bool:
        """Check if Docker is available."""
        import subprocess
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _get_sandbox(self):
        """Get or create the sandbox."""
        if self._sandbox is not None:
            return self._sandbox
        
        if self._use_docker:
            self._sandbox = DockerSandbox(
                image=self.docker_image,
                timeout=self.default_timeout,
                memory_limit=self.memory_limit,
                cpu_limit=self.cpu_limit,
                network_disabled=not self.network_enabled
            )
            logger.info(f"[{self.name}] Using DockerSandbox ({self.docker_image})")
        else:
            # Check if unsafe execution is allowed
            allow_unsafe = os.getenv("BLACKBOARD_ALLOW_UNSAFE_EXECUTION", "").lower() in ("true", "1", "yes")
            if not allow_unsafe:
                raise RuntimeError(
                    "Docker is not available and BLACKBOARD_ALLOW_UNSAFE_EXECUTION is not set. "
                    "Either install Docker or set BLACKBOARD_ALLOW_UNSAFE_EXECUTION=1 to use local execution."
                )
            
            self._sandbox = InsecureLocalExecutor(
                timeout=self.default_timeout,
                _unsafe_acknowledged=True
            )
            logger.warning(f"[{self.name}] Using InsecureLocalExecutor (Docker not available)")
        
        return self._sandbox
    
    def _build_execution_code(
        self,
        code: str,
        packages: List[str],
        files: Dict[str, str]
    ) -> str:
        """Build the full execution code with setup."""
        parts = []
        
        # Package installation (for Docker)
        if packages and self._use_docker:
            install_code = f'''
import subprocess
import sys

packages = {packages!r}
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
'''
            parts.append(install_code)
        
        # File creation
        if files:
            file_code = "import os\n"
            for filename, content in files.items():
                # Escape content for Python string
                escaped = content.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
                file_code += f"with open('{filename}', 'w') as f:\n    f.write('{escaped}')\n"
            parts.append(file_code)
        
        # User code
        parts.append(code)
        
        return "\n\n".join(parts)
    
    async def run(
        self,
        state: Blackboard,
        inputs: Optional[CodeInterpreterInput] = None
    ) -> WorkerOutput:
        """Execute code and return results."""
        if not inputs or not inputs.code:
            return WorkerOutput(
                metadata={"error": "code is required"}
            )
        
        if inputs.language != "python":
            return WorkerOutput(
                metadata={"error": f"Language '{inputs.language}' not supported. Only 'python' is available."}
            )
        
        logger.info(f"[{self.name}] Executing code ({len(inputs.code)} chars)")
        if inputs.packages:
            logger.info(f"[{self.name}] Installing packages: {inputs.packages}")
        
        try:
            sandbox = self._get_sandbox()
            
            # Build full execution code
            full_code = self._build_execution_code(
                code=inputs.code,
                packages=inputs.packages,
                files=inputs.files
            )
            
            # Execute
            result: SandboxResult = await sandbox.execute(
                code=full_code,
                timeout=inputs.timeout
            )
            
            # Format output
            output_parts = []
            if result.stdout:
                output_parts.append(f"=== STDOUT ===\n{result.stdout}")
            if result.stderr:
                output_parts.append(f"=== STDERR ===\n{result.stderr}")
            if result.error:
                output_parts.append(f"=== ERROR ===\n{result.error}")
            
            content = "\n\n".join(output_parts) if output_parts else "(no output)"
            
            logger.info(f"[{self.name}] Execution {'succeeded' if result.success else 'failed'} in {result.execution_time:.2f}s")
            
            return WorkerOutput(
                artifact=Artifact(
                    type="code_output",
                    content=content,
                    creator=self.name,
                    metadata={
                        "success": result.success,
                        "execution_time": result.execution_time,
                        "stdout_length": len(result.stdout),
                        "stderr_length": len(result.stderr),
                        "error": result.error,
                        "packages": inputs.packages,
                        "sandbox_type": "docker" if self._use_docker else "local"
                    }
                )
            )
            
        except Exception as e:
            logger.error(f"[{self.name}] Execution error: {e}")
            return WorkerOutput(
                metadata={
                    "error": str(e),
                    "code_preview": inputs.code[:200] + "..." if len(inputs.code) > 200 else inputs.code
                }
            )
