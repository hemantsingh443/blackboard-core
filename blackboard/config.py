"""
Configuration System for Blackboard SDK

Provides centralized configuration with environment variable support.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import os


@dataclass
class BlackboardConfig:
    """
    Central configuration for the Blackboard SDK.
    
    Supports configuration via:
    1. Direct instantiation with kwargs
    2. Environment variables with BLACKBOARD_ prefix
    3. Default values
    
    Environment Variables:
        BLACKBOARD_MAX_STEPS: Maximum orchestration steps (default: 20)
        BLACKBOARD_ALLOW_UNSAFE_EXECUTION: Allow InsecureLocalExecutor (default: false)
        BLACKBOARD_SIMPLE_PROMPTS: Use text-based prompts for weaker LLMs (default: false)
        BLACKBOARD_ENABLE_PARALLEL: Enable parallel worker execution (default: true)
        BLACKBOARD_MAX_HISTORY: Maximum history entries to keep (default: 1000)
        BLACKBOARD_TOKEN_BUDGET: Maximum tokens per session (default: 100000)
        BLACKBOARD_AUTO_SAVE_PATH: Path for auto-saving state (default: None)
        BLACKBOARD_VERBOSE: Enable verbose logging (default: false)
    
    Example:
        # Direct configuration
        config = BlackboardConfig(max_steps=50, allow_unsafe_execution=True)
        orchestrator = Orchestrator(llm=llm, workers=workers, config=config)
        
        # Environment variable configuration
        # Set BLACKBOARD_MAX_STEPS=50 in environment
        config = BlackboardConfig.from_env()
        
        # Mixed: env vars as defaults, kwargs override
        config = BlackboardConfig.from_env(allow_unsafe_execution=True)
    """
    
    # Execution limits
    max_steps: int = 20
    max_history: int = 1000
    token_budget: int = 100000
    max_recursion_depth: int = 3  # For fractal agent nesting
    
    # Safety flags
    allow_unsafe_execution: bool = False
    
    # LLM configuration
    simple_prompts: bool = False
    use_tool_calling: bool = True
    allow_json_fallback: bool = False  # v2.0: Secure default - fail clearly if tool calling fails
    strict_tools: bool = False
    reasoning_strategy: str = "oneshot"  # "oneshot" or "cot" (chain-of-thought)
    
    # Execution modes
    enable_parallel: bool = True
    auto_summarize: bool = False
    
    # Paths
    auto_save_path: Optional[str] = None
    
    # Observability
    verbose: bool = False
    
    # Summarization thresholds
    summarize_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "artifacts": 10,
        "feedback": 20,
        "steps": 50
    })
    
    @classmethod
    def from_env(cls, **overrides) -> "BlackboardConfig":
        """
        Load configuration from environment variables.
        
        Environment variables are read with BLACKBOARD_ prefix.
        Explicit overrides take precedence over env vars.
        
        Args:
            **overrides: Explicit config values that override env vars
            
        Returns:
            BlackboardConfig instance
            
        Example:
            # Uses BLACKBOARD_MAX_STEPS if set, otherwise default
            config = BlackboardConfig.from_env()
            
            # Override specific values
            config = BlackboardConfig.from_env(max_steps=100)
        """
        def get_bool(key: str, default: bool) -> bool:
            val = os.getenv(f"BLACKBOARD_{key.upper()}")
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes", "on")
        
        def get_int(key: str, default: int) -> int:
            val = os.getenv(f"BLACKBOARD_{key.upper()}")
            if val is None:
                return default
            try:
                return int(val)
            except ValueError:
                return default
        
        def get_str(key: str, default: Optional[str]) -> Optional[str]:
            return os.getenv(f"BLACKBOARD_{key.upper()}", default)
        
        # Build config from env vars
        env_config = {
            "max_steps": get_int("max_steps", 20),
            "max_history": get_int("max_history", 1000),
            "token_budget": get_int("token_budget", 100000),
            "max_recursion_depth": get_int("max_recursion_depth", 3),
            "allow_unsafe_execution": get_bool("allow_unsafe_execution", False),
            "simple_prompts": get_bool("simple_prompts", False),
            "use_tool_calling": get_bool("use_tool_calling", True),
            "allow_json_fallback": get_bool("allow_json_fallback", False),  
            "strict_tools": get_bool("strict_tools", False),
            "reasoning_strategy": get_str("reasoning_strategy", "oneshot"),
            "enable_parallel": get_bool("enable_parallel", True),
            "auto_summarize": get_bool("auto_summarize", False),
            "auto_save_path": get_str("auto_save_path", None),
            "verbose": get_bool("verbose", False),
        }
        
        # Apply overrides
        env_config.update(overrides)
        
        return cls(**env_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "max_steps": self.max_steps,
            "max_history": self.max_history,
            "token_budget": self.token_budget,
            "max_recursion_depth": self.max_recursion_depth,
            "allow_unsafe_execution": self.allow_unsafe_execution,
            "simple_prompts": self.simple_prompts,
            "use_tool_calling": self.use_tool_calling,
            "allow_json_fallback": self.allow_json_fallback,
            "strict_tools": self.strict_tools,
            "reasoning_strategy": self.reasoning_strategy,
            "enable_parallel": self.enable_parallel,
            "auto_summarize": self.auto_summarize,
            "auto_save_path": self.auto_save_path,
            "verbose": self.verbose,
            "summarize_thresholds": self.summarize_thresholds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlackboardConfig":
        """
        Create config from dictionary.
        
        Used for propagating config to sub-agents in fractal architectures.
        
        Args:
            data: Dictionary with config values
            
        Returns:
            BlackboardConfig instance
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def for_child_agent(self, depth_decrement: int = 1) -> "BlackboardConfig":
        """
        Create a child config with decremented recursion depth.
        
        Used when spawning sub-agents in fractal architectures.
        Inherits all settings including allow_unsafe_execution.
        
        Args:
            depth_decrement: How much to decrement recursion depth
            
        Returns:
            New config for child agent
        """
        child_data = self.to_dict()
        child_data["max_recursion_depth"] = max(0, self.max_recursion_depth - depth_decrement)
        return self.from_dict(child_data)


# Default configuration instance
DEFAULT_CONFIG = BlackboardConfig()
