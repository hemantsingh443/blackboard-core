"""Tests for the runtime module and security features."""

import pytest
import os


class TestLocalRuntime:
    """Tests for LocalRuntime security model."""
    
    def test_raises_error_without_acknowledgment(self):
        """Test that LocalRuntime raises error without explicit flag."""
        from blackboard.runtime import LocalRuntime, RuntimeSecurityError
        
        # Clear env var if set
        os.environ.pop("BLACKBOARD_ALLOW_UNSAFE_EXECUTION", None)
        
        with pytest.raises(RuntimeSecurityError) as exc_info:
            LocalRuntime()
        
        assert "HOST PRIVILEGES" in str(exc_info.value)
        assert "dangerously_allow_execution" in str(exc_info.value)
    
    def test_works_with_explicit_flag(self):
        """Test that LocalRuntime works when explicitly acknowledged."""
        from blackboard.runtime import LocalRuntime
        
        runtime = LocalRuntime(dangerously_allow_execution=True)
        assert runtime is not None
        assert runtime.timeout == 30.0
    
    def test_works_with_env_var(self):
        """Test that LocalRuntime works with environment variable."""
        from blackboard.runtime import LocalRuntime
        
        os.environ["BLACKBOARD_ALLOW_UNSAFE_EXECUTION"] = "true"
        try:
            runtime = LocalRuntime()
            assert runtime is not None
        finally:
            os.environ.pop("BLACKBOARD_ALLOW_UNSAFE_EXECUTION", None)
    
    @pytest.mark.asyncio
    async def test_execute_simple_code(self):
        """Test executing simple code."""
        from blackboard.runtime import LocalRuntime
        
        runtime = LocalRuntime(dangerously_allow_execution=True, timeout=5)
        result = await runtime.execute("print('hello world')")
        
        assert result.success
        assert "hello world" in result.stdout
        assert result.error is None
        
        await runtime.close()
    
    @pytest.mark.asyncio
    async def test_execute_with_error(self):
        """Test that errors are captured properly."""
        from blackboard.runtime import LocalRuntime
        
        runtime = LocalRuntime(dangerously_allow_execution=True, timeout=5)
        result = await runtime.execute("raise ValueError('test error')")
        
        assert not result.success
        assert "ValueError" in result.stderr or "test error" in result.stderr
        
        await runtime.close()
    
    @pytest.mark.asyncio
    async def test_timeout(self):
        """Test that timeouts are enforced."""
        from blackboard.runtime import LocalRuntime, RuntimeTimeoutError
        
        runtime = LocalRuntime(dangerously_allow_execution=True, timeout=1)
        
        with pytest.raises(RuntimeTimeoutError):
            await runtime.execute("import time; time.sleep(10)", timeout=1)
        
        await runtime.close()
    
    @pytest.mark.asyncio
    async def test_sensitive_env_vars_removed(self):
        """Test that sensitive env vars are not passed to subprocess."""
        from blackboard.runtime import LocalRuntime
        
        os.environ["OPENAI_API_KEY"] = "test-secret-key"
        
        runtime = LocalRuntime(dangerously_allow_execution=True)
        result = await runtime.execute("import os; print(os.environ.get('OPENAI_API_KEY', 'NOT_FOUND'))")
        
        assert result.success
        assert "NOT_FOUND" in result.stdout
        assert "test-secret-key" not in result.stdout
        
        await runtime.close()


class TestBlackboardConfigPropagation:
    """Tests for BlackboardConfig serialization and child config creation."""
    
    def test_to_dict_and_from_dict(self):
        """Test config can be serialized and deserialized."""
        from blackboard.config import BlackboardConfig
        
        original = BlackboardConfig(
            max_steps=50,
            allow_unsafe_execution=True,
            max_recursion_depth=5,
            verbose=True
        )
        
        data = original.to_dict()
        restored = BlackboardConfig.from_dict(data)
        
        assert restored.max_steps == 50
        assert restored.allow_unsafe_execution is True
        assert restored.max_recursion_depth == 5
        assert restored.verbose is True
    
    def test_for_child_agent(self):
        """Test creating child config with decremented recursion depth."""
        from blackboard.config import BlackboardConfig
        
        parent = BlackboardConfig(
            max_recursion_depth=3,
            allow_unsafe_execution=True,
            max_steps=100
        )
        
        child = parent.for_child_agent()
        
        # Recursion depth should be decremented
        assert child.max_recursion_depth == 2
        
        # Other settings should be inherited
        assert child.allow_unsafe_execution is True
        assert child.max_steps == 100
    
    def test_recursion_depth_never_negative(self):
        """Test that recursion depth doesn't go below 0."""
        from blackboard.config import BlackboardConfig
        
        parent = BlackboardConfig(max_recursion_depth=0)
        child = parent.for_child_agent()
        
        assert child.max_recursion_depth == 0
    
    def test_from_env_includes_recursion_depth(self):
        """Test that from_env loads max_recursion_depth."""
        from blackboard.config import BlackboardConfig
        
        os.environ["BLACKBOARD_MAX_RECURSION_DEPTH"] = "5"
        try:
            config = BlackboardConfig.from_env()
            assert config.max_recursion_depth == 5
        finally:
            os.environ.pop("BLACKBOARD_MAX_RECURSION_DEPTH", None)


class TestDockerRuntime:
    """Tests for DockerRuntime (requires Docker to actually run)."""
    
    def test_initialization(self):
        """Test that DockerRuntime can be initialized."""
        from blackboard.runtime import DockerRuntime
        
        runtime = DockerRuntime(
            image="python:3.11-slim",
            timeout=60,
            memory_limit="512m"
        )
        
        assert runtime.image == "python:3.11-slim"
        assert runtime.timeout == 60
        assert runtime.memory_limit == "512m"
        assert runtime.network_disabled is True
