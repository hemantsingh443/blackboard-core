"""
Runtime Environments for Code Execution

Provides a clean abstraction for code execution environments.
This replaces the older sandbox.py module with a simplified interface.

.. versionadded:: 1.5.2
.. versionchanged:: 1.6.1
   LocalRuntime renamed to InsecureLocalRuntime for explicit risk naming.

Example:
    from blackboard.runtime import InsecureLocalRuntime, DockerRuntime
    
    # For development (UNSAFE - requires explicit acknowledgment)
    runtime = InsecureLocalRuntime(dangerously_allow_execution=True)
    
    # For production (recommended)
    runtime = DockerRuntime(image="python:3.11-slim")
    
    result = await runtime.execute("print('hello')")
"""

import asyncio
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, runtime_checkable

logger = logging.getLogger("blackboard.runtime")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExecutionResult:
    """Result from code execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0


class RuntimeError(Exception):
    """Base exception for runtime execution errors."""
    pass


class RuntimeTimeoutError(RuntimeError):
    """Raised when code execution times out."""
    pass


class RuntimeSecurityError(RuntimeError):
    """Raised when attempting unsafe execution without acknowledgment."""
    pass


# =============================================================================
# Runtime Protocol
# =============================================================================

@runtime_checkable
class Runtime(Protocol):
    """
    Protocol for code execution environments.
    
    Implementations provide different isolation levels:
    - LocalRuntime: No isolation, runs with host privileges (development only)
    - DockerRuntime: Full container isolation (production)
    
    Example:
        runtime = DockerRuntime()
        result = await runtime.execute("print('hello')")
        print(result.stdout)  # "hello"
    """
    
    async def execute(
        self,
        code: str,
        timeout: float = 30.0,
        env: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """
        Execute code in the runtime environment.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            env: Environment variables to pass
            
        Returns:
            ExecutionResult with stdout, stderr, and status
        """
        ...
    
    async def close(self) -> None:
        """Clean up runtime resources."""
        ...


# =============================================================================
# Local Runtime (Development Only)
# =============================================================================

class InsecureLocalRuntime:
    """
    Local subprocess runtime for code execution.
    
    ⚠️ **SECURITY WARNING**: This runtime executes code with HOST PRIVILEGES.
    Only use for development with trusted code. For production, use DockerRuntime.
    
    The explicit `dangerously_allow_execution=True` parameter is required to
    acknowledge the security implications.
    
    Args:
        dangerously_allow_execution: Must be True to allow execution
        timeout: Default timeout in seconds
        python_path: Path to Python interpreter
        allowed_imports: Optional list of allowed module imports
        
    Example:
        # Development use (explicit acknowledgment required)
        runtime = LocalRuntime(dangerously_allow_execution=True, timeout=30)
        result = await runtime.execute("print('hello')")
        
    Raises:
        RuntimeSecurityError: If dangerously_allow_execution is not True
    """
    
    def __init__(
        self,
        dangerously_allow_execution: bool = False,
        timeout: float = 30.0,
        python_path: Optional[str] = None,
        allowed_imports: Optional[list] = None
    ):
        # Check environment variable as fallback
        env_unsafe = os.getenv(
            "BLACKBOARD_ALLOW_UNSAFE_EXECUTION", "false"
        ).lower() in ("true", "1", "yes")
        
        if not (dangerously_allow_execution or env_unsafe):
            raise RuntimeSecurityError(
                "LocalRuntime executes code with HOST PRIVILEGES. "
                "For production, use DockerRuntime. "
                "To acknowledge this risk and proceed, set:\n"
                "  - dangerously_allow_execution=True in constructor, OR\n"
                "  - BLACKBOARD_ALLOW_UNSAFE_EXECUTION=1 environment variable"
            )
        
        self.timeout = timeout
        self.python_path = python_path or sys.executable
        self.allowed_imports = allowed_imports
        
        logger.debug("InsecureLocalRuntime initialized (unsafe execution acknowledged)")
    
    async def execute(
        self,
        code: str,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """Execute code in a subprocess."""
        import time
        start_time = time.time()
        
        timeout = timeout or self.timeout
        wrapped_code = self._wrap_code(code)
        
        # Create temp file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapped_code)
            temp_path = f.name
        
        try:
            # Build environment
            run_env = os.environ.copy()
            if env:
                run_env.update(env)
            
            # Remove sensitive env vars
            for key in ['AWS_SECRET_ACCESS_KEY', 'OPENAI_API_KEY', 'DATABASE_URL', 
                        'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']:
                run_env.pop(key, None)
            
            # Run in subprocess
            process = await asyncio.create_subprocess_exec(
                self.python_path, temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=run_env
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise RuntimeTimeoutError(f"Execution timed out after {timeout}s")
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=process.returncode == 0,
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                error=None if process.returncode == 0 else f"Exit code: {process.returncode}",
                execution_time=execution_time
            )
            
        except RuntimeTimeoutError:
            raise
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    async def close(self) -> None:
        """No resources to clean up for local runtime."""
        pass
    
    def _wrap_code(self, code: str) -> str:
        """Wrap user code with import restrictions if configured."""
        if self.allowed_imports:
            import_check = f"""
import sys
_allowed = {self.allowed_imports!r}
_original_import = __builtins__.__import__

def _restricted_import(name, *args, **kwargs):
    if name.split('.')[0] not in _allowed:
        raise ImportError(f"Import of '{{name}}' is not allowed")
    return _original_import(name, *args, **kwargs)

__builtins__.__import__ = _restricted_import
"""
            return import_check + code
        return code


# =============================================================================
# Docker Runtime (Production)
# =============================================================================

class DockerRuntime:
    """
    Docker-based runtime for production code execution.
    
    Provides full isolation by running code in ephemeral containers.
    
    Args:
        image: Docker image to use
        timeout: Default timeout in seconds
        memory_limit: Memory limit (e.g., "256m")
        cpu_limit: CPU limit (e.g., "0.5")
        network_disabled: Whether to disable networking
        
    Example:
        runtime = DockerRuntime(timeout=30, memory_limit="128m")
        result = await runtime.execute("print('hello')")
    """
    
    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: float = 30.0,
        memory_limit: str = "256m",
        cpu_limit: str = "0.5",
        network_disabled: bool = True
    ):
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network_disabled = network_disabled
    
    async def execute(
        self,
        code: str,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """Execute code in a Docker container."""
        import time
        import subprocess
        
        start_time = time.time()
        timeout = timeout or self.timeout
        
        # Build docker command
        cmd = [
            "docker", "run", "--rm",
            f"--memory={self.memory_limit}",
            f"--cpus={self.cpu_limit}",
        ]
        
        if self.network_disabled:
            cmd.append("--network=none")
        
        # Add environment variables
        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])
        
        # Add image and Python command
        cmd.extend([self.image, "python", "-c", code])
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                subprocess.run(["docker", "kill", str(process.pid)], capture_output=True)
                raise RuntimeTimeoutError(f"Docker execution timed out after {timeout}s")
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=process.returncode == 0,
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                error=None if process.returncode == 0 else f"Exit code: {process.returncode}",
                execution_time=execution_time
            )
            
        except RuntimeTimeoutError:
            raise
        except FileNotFoundError:
            raise RuntimeError("Docker not found. Please install Docker.")
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    async def close(self) -> None:
        """No persistent resources to clean up."""
        pass


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Deprecated aliases for migration
LocalRuntime = InsecureLocalRuntime  # Deprecated: Use InsecureLocalRuntime
InsecureLocalExecutor = InsecureLocalRuntime
DockerSandbox = DockerRuntime
SandboxResult = ExecutionResult
SandboxError = RuntimeError
SandboxTimeoutError = RuntimeTimeoutError
