"""
Sandbox for Safe Code Execution

Provides isolation for executing untrusted code from LLM-generated workers.
"""

import asyncio
import logging
import subprocess
import sys
import tempfile
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, runtime_checkable

logger = logging.getLogger("blackboard.sandbox")


@dataclass
class SandboxResult:
    """Result from sandbox code execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0


class SandboxError(Exception):
    """Raised when sandbox execution fails."""
    pass


class SandboxTimeoutError(SandboxError):
    """Raised when code execution times out."""
    pass


@runtime_checkable
class Sandbox(Protocol):
    """
    Protocol for code execution sandboxes.
    
    Implementations should provide isolation to prevent:
    - File system access outside allowed paths
    - Network access (unless explicitly allowed)
    - Resource exhaustion (CPU, memory)
    - Access to environment variables/secrets
    
    Example:
        executor = InsecureLocalExecutor(timeout=30)
        result = await executor.execute("print('hello')")
        print(result.stdout)  # "hello"
    """
    
    async def execute(
        self, 
        code: str, 
        timeout: float = 30.0,
        env: Optional[Dict[str, str]] = None
    ) -> SandboxResult:
        """
        Execute code in an isolated environment.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            env: Optional environment variables to pass
            
        Returns:
            SandboxResult with stdout, stderr, and status
            
        Raises:
            SandboxTimeoutError: If execution exceeds timeout
            SandboxError: For other execution errors
        """
        ...


class InsecureLocalExecutor:
    """
    Local process executor for code execution.
    
    ⚠️ NOT SECURE - runs code with same privileges as the host process.
    Use DockerSandbox for untrusted code.
    
    .. warning::
        This executor runs with HOST PRIVILEGES. For production use with
        untrusted code, use DockerSandbox instead.
        
    .. note::
        **Warning Suppression**: Since executors are typically instantiated 
        before the Orchestrator, you must pass ``_unsafe_acknowledged=True`` 
        directly when creating this executor. The ``allow_unsafe_execution`` 
        config option in BlackboardConfig only applies to workers that create
        executors internally.
        
        Example::
        
            # Direct instantiation - use _unsafe_acknowledged
            executor = InsecureLocalExecutor(_unsafe_acknowledged=True)
            
            # Via BlackboardConfig - for internal worker use
            config = BlackboardConfig(allow_unsafe_execution=True)
    
    Args:
        timeout: Default timeout in seconds
        python_path: Path to Python interpreter (defaults to current)
        allowed_imports: List of allowed module imports (None = all allowed)
        _unsafe_acknowledged: Set to True to suppress security warning
        
    Example:
        executor = InsecureLocalExecutor(timeout=10, _unsafe_acknowledged=True)
        result = await executor.execute("x = 1 + 1; print(x)")
    """
    
    def __init__(
        self,
        timeout: float = 30.0,
        python_path: Optional[str] = None,
        allowed_imports: Optional[list] = None,
        _unsafe_acknowledged: bool = False
    ):
        import os
        
        # Check environment variable for unsafe execution acknowledgment
        env_unsafe = os.getenv("BLACKBOARD_ALLOW_UNSAFE_EXECUTION", "false").lower() in ("true", "1", "yes")
        
        # Only warn if NEITHER the explicit flag NOR the env var is set
        if not (_unsafe_acknowledged or env_unsafe):
            logger.warning(
                "⚠️ InsecureLocalExecutor runs code with HOST PRIVILEGES. "
                "Use DockerSandbox for untrusted code. "
                "Set BLACKBOARD_ALLOW_UNSAFE_EXECUTION=1 env var or _unsafe_acknowledged=True to suppress."
            )
        
        self.timeout = timeout
        self.python_path = python_path or sys.executable
        self.allowed_imports = allowed_imports
    
    async def execute(
        self,
        code: str,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None
    ) -> SandboxResult:
        import time
        start_time = time.time()
        
        timeout = timeout or self.timeout
        
        # Wrap code to capture return value
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
            for key in ['AWS_SECRET_ACCESS_KEY', 'OPENAI_API_KEY', 'DATABASE_URL']:
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
                raise SandboxTimeoutError(f"Execution timed out after {timeout}s")
            
            execution_time = time.time() - start_time
            
            return SandboxResult(
                success=process.returncode == 0,
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                error=None if process.returncode == 0 else f"Exit code: {process.returncode}",
                execution_time=execution_time
            )
            
        except SandboxTimeoutError:
            raise
        except Exception as e:
            return SandboxResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
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


class DockerSandbox:
    """
    Docker-based sandbox for full isolation.
    
    Provides strong isolation by running code in a Docker container.
    Requires Docker to be installed and running.
    
    Args:
        image: Docker image to use (default: python:3.11-slim)
        timeout: Default timeout in seconds
        memory_limit: Memory limit (e.g., "256m")
        cpu_limit: CPU limit (e.g., "0.5")
        network_disabled: Whether to disable networking
        
    Example:
        sandbox = DockerSandbox(timeout=30, memory_limit="128m")
        result = await sandbox.execute("print('safe!')")
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
    ) -> SandboxResult:
        import time
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
                # Kill the container
                subprocess.run(["docker", "kill", str(process.pid)], capture_output=True)
                raise SandboxTimeoutError(f"Docker execution timed out after {timeout}s")
            
            execution_time = time.time() - start_time
            
            return SandboxResult(
                success=process.returncode == 0,
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                error=None if process.returncode == 0 else f"Exit code: {process.returncode}",
                execution_time=execution_time
            )
            
        except SandboxTimeoutError:
            raise
        except FileNotFoundError:
            raise SandboxError("Docker not found. Please install Docker.")
        except Exception as e:
            return SandboxResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )


class NoOpSandbox:
    """
    No-op sandbox that executes code directly (UNSAFE).
    
    Only use for trusted code or testing. Provides no isolation.
    """
    
    async def execute(
        self,
        code: str,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None
    ) -> SandboxResult:
        import time
        start_time = time.time()
        
        # Capture stdout
        from io import StringIO
        import contextlib
        
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        try:
            with contextlib.redirect_stdout(stdout_capture):
                with contextlib.redirect_stderr(stderr_capture):
                    exec(code, {"__builtins__": __builtins__}, {})
            
            return SandboxResult(
                success=True,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                error=str(e),
                execution_time=time.time() - start_time
            )
