"""
Prompt Registry

Centralized management for externalized prompts with Jinja2 templating support.

This enables:
- Prompt overrides without code changes
- Jinja2 templates for dynamic prompts
- Runtime patching for optimization experiments

Example:
    registry = PromptRegistry("prompts/", "blackboard.prompts.json")
    
    # Get rendered prompt for a worker
    prompt = registry.get("Writer", {"instructions": "Write a haiku"})
    
    # Set override at runtime (for testing patches)
    registry.set("Writer", "{{ instructions | upper }}")
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

logger = logging.getLogger("blackboard.prompts")

# Try to import Jinja2 (optional dependency)
try:
    from jinja2 import Environment, FileSystemLoader, BaseLoader, Template, TemplateNotFound
    _HAS_JINJA2 = True
except ImportError:
    _HAS_JINJA2 = False
    Template = None  # type: ignore
    logger.debug("Jinja2 not installed, template features disabled")


class PromptRegistry:
    """
    Registry for externalized prompts with Jinja2 template support.
    
    Prompts are loaded in priority order:
    1. Runtime overrides (set via `set()`)
    2. JSON config file (blackboard.prompts.json)
    3. Jinja2 templates from directory (prompts/*.jinja2)
    4. Default prompts from worker classes (fallback)
    
    Args:
        prompts_dir: Directory containing .jinja2 template files
        config_path: Path to JSON config file with prompt overrides
        
    Example:
        registry = PromptRegistry()
        
        # With context variables
        prompt = registry.get("supervisor", {
            "workers": ["Writer", "Critic"],
            "goal": "Write a poem"
        })
    """
    
    def __init__(
        self,
        prompts_dir: str = "prompts/",
        config_path: str = "blackboard.prompts.json"
    ):
        self.prompts_dir = Path(prompts_dir)
        self.config_path = Path(config_path)
        
        # Runtime overrides (highest priority)
        self._overrides: Dict[str, str] = {}
        
        # JSON config overrides
        self._config: Dict[str, str] = {}
        
        # Jinja2 environment for templates
        self._env: Optional["Environment"] = None
        
        # Load config file if exists
        self._load_config()
        
        # Initialize Jinja2 environment if available and dir exists
        self._init_jinja_env()
    
    def _load_config(self) -> None:
        """Load prompts from JSON config file."""
        if not self.config_path.exists():
            logger.debug(f"Config file not found: {self.config_path}")
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            logger.info(f"Loaded {len(self._config)} prompts from {self.config_path}")
        except Exception as e:
            logger.warning(f"Failed to load prompt config: {e}")
    
    def _init_jinja_env(self) -> None:
        """Initialize Jinja2 environment if available."""
        if not _HAS_JINJA2:
            return
        
        if self.prompts_dir.exists():
            self._env = Environment(
                loader=FileSystemLoader(str(self.prompts_dir)),
                autoescape=False,  # Prompts don't need HTML escaping
                trim_blocks=True,
                lstrip_blocks=True
            )
            logger.debug(f"Jinja2 environment initialized: {self.prompts_dir}")
        else:
            # Create base loader for string templates
            self._env = Environment(autoescape=False)
            logger.debug("Jinja2 environment initialized (no templates directory)")
    
    def get(
        self,
        key: str,
        context: Optional[Dict[str, Any]] = None,
        default: Optional[str] = None
    ) -> Optional[str]:
        """
        Get a rendered prompt by key.
        
        Args:
            key: Prompt identifier (e.g., worker name, "supervisor")
            context: Variables to inject into the template
            default: Fallback if prompt not found
            
        Returns:
            Rendered prompt string, or default if not found
        """
        context = context or {}
        
        # Priority 1: Runtime overrides
        if key in self._overrides:
            return self._render_string(self._overrides[key], context)
        
        # Priority 2: JSON config
        if key in self._config:
            return self._render_string(self._config[key], context)
        
        # Priority 3: Jinja2 template file
        template = self._load_template(key)
        if template:
            try:
                return template.render(**context)
            except Exception as e:
                logger.warning(f"Template render error for '{key}': {e}")
        
        # Priority 4: Default fallback
        return default
    
    def get_template(self, key: str) -> Optional["Template"]:
        """
        Get the raw Jinja2 template for a key (for patching/inspection).
        
        Args:
            key: Prompt identifier
            
        Returns:
            Jinja2 Template object or None if not found
        """
        if not _HAS_JINJA2:
            return None
        
        # Check overrides first
        if key in self._overrides:
            return self._env.from_string(self._overrides[key])
        
        # Check config
        if key in self._config:
            return self._env.from_string(self._config[key])
        
        # Try template file
        return self._load_template(key)
    
    def set(self, key: str, template_str: str) -> None:
        """
        Set a runtime override for a prompt.
        
        This is highest priority and will be used instead of
        config file or template files.
        
        Args:
            key: Prompt identifier
            template_str: Jinja2 template string
        """
        self._overrides[key] = template_str
        logger.debug(f"Set override for prompt '{key}'")
    
    def clear_override(self, key: str) -> bool:
        """
        Clear a runtime override.
        
        Returns:
            True if an override was removed, False if none existed
        """
        if key in self._overrides:
            del self._overrides[key]
            return True
        return False
    
    def save_config(self, path: Optional[str] = None) -> None:
        """
        Save current config + overrides to JSON file.
        
        Args:
            path: Optional path to save to (defaults to config_path)
        """
        path = Path(path) if path else self.config_path
        
        # Merge config with overrides
        merged = {**self._config, **self._overrides}
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2)
        
        logger.info(f"Saved {len(merged)} prompts to {path}")
    
    def list_keys(self) -> Dict[str, str]:
        """
        List all available prompt keys with their sources.
        
        Returns:
            Dict mapping key to source ("override", "config", "template")
        """
        keys = {}
        
        # From templates
        if self._env and self.prompts_dir.exists():
            for template_path in self.prompts_dir.glob("*.jinja2"):
                key = template_path.stem  # filename without extension
                keys[key] = "template"
        
        # From config (overrides template)
        for key in self._config:
            keys[key] = "config"
        
        # From overrides (highest priority)
        for key in self._overrides:
            keys[key] = "override"
        
        return keys
    
    def _load_template(self, key: str) -> Optional["Template"]:
        """Load a Jinja2 template from file."""
        if not _HAS_JINJA2 or not self._env:
            return None
        
        try:
            # Try both with and without .jinja2 extension
            try:
                return self._env.get_template(f"{key}.jinja2")
            except TemplateNotFound:
                return self._env.get_template(key)
        except TemplateNotFound:
            return None
        except Exception as e:
            logger.warning(f"Error loading template '{key}': {e}")
            return None
    
    def _render_string(self, template_str: str, context: Dict[str, Any]) -> str:
        """Render a template string with context."""
        if not _HAS_JINJA2 or not self._env:
            # Simple variable substitution fallback
            result = template_str
            for key, value in context.items():
                result = result.replace(f"{{{{ {key} }}}}", str(value))
            return result
        
        try:
            template = self._env.from_string(template_str)
            return template.render(**context)
        except Exception as e:
            logger.warning(f"Template render error: {e}")
            return template_str
    
    def reload(self) -> None:
        """Reload config file and re-initialize templates."""
        self._load_config()
        self._init_jinja_env()


def create_default_prompts_dir(prompts_dir: str = "prompts/") -> None:
    """
    Create a default prompts directory structure.
    
    Called by `blackboard init` CLI command.
    """
    path = Path(prompts_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # Create example template
    example_template = path / "example.jinja2"
    if not example_template.exists():
        example_template.write_text("""
{# Example Jinja2 template for a worker prompt #}
{# Available variables depend on the worker #}

You are a helpful assistant.

{% if instructions %}
## Task
{{ instructions }}
{% endif %}

{% if context %}
## Context
{{ context }}
{% endif %}
""".strip())
    
    logger.info(f"Created prompts directory: {path}")


def create_default_config(config_path: str = "blackboard.prompts.json") -> None:
    """
    Create a default prompts config file.
    
    Called by `blackboard init` CLI command.
    """
    path = Path(config_path)
    if not path.exists():
        path.write_text(json.dumps({
            "_comment": "Prompt overrides - keys can be worker names or 'supervisor'",
        }, indent=2))
        logger.info(f"Created prompts config: {path}")
