"""Tests for PromptRegistry with Jinja2 templating."""

import pytest
import json
from pathlib import Path


class TestPromptRegistry:
    """Tests for PromptRegistry functionality."""
    
    @pytest.fixture
    def prompts_dir(self, tmp_path):
        """Create a temporary prompts directory."""
        return tmp_path / "prompts"
    
    @pytest.fixture
    def config_path(self, tmp_path):
        """Create a temporary config path."""
        return tmp_path / "blackboard.prompts.json"
    
    def test_basic_initialization(self, prompts_dir, config_path):
        """Test registry initializes without errors."""
        from blackboard.prompts import PromptRegistry
        
        registry = PromptRegistry(str(prompts_dir), str(config_path))
        assert registry is not None
    
    def test_get_returns_default_when_not_found(self, prompts_dir, config_path):
        """Test that get() returns default when key not found."""
        from blackboard.prompts import PromptRegistry
        
        registry = PromptRegistry(str(prompts_dir), str(config_path))
        
        result = registry.get("nonexistent", default="fallback value")
        assert result == "fallback value"
    
    def test_set_and_get_override(self, prompts_dir, config_path):
        """Test runtime override via set()."""
        from blackboard.prompts import PromptRegistry
        
        registry = PromptRegistry(str(prompts_dir), str(config_path))
        
        registry.set("Writer", "Custom prompt for {{ task }}")
        result = registry.get("Writer", {"task": "testing"})
        
        assert "Custom prompt for testing" in result
    
    def test_load_from_json_config(self, prompts_dir, config_path):
        """Test loading prompts from JSON config."""
        from blackboard.prompts import PromptRegistry
        
        # Create config file
        config_path.write_text(json.dumps({
            "Critic": "Review this: {{ content }}"
        }))
        
        registry = PromptRegistry(str(prompts_dir), str(config_path))
        
        result = registry.get("Critic", {"content": "test artifact"})
        assert "Review this: test artifact" in result
    
    def test_override_priority_over_config(self, prompts_dir, config_path):
        """Test that runtime overrides take priority over config."""
        from blackboard.prompts import PromptRegistry
        
        config_path.write_text(json.dumps({
            "Writer": "Config prompt"
        }))
        
        registry = PromptRegistry(str(prompts_dir), str(config_path))
        
        # Config should work first
        assert registry.get("Writer") == "Config prompt"
        
        # Override should take precedence
        registry.set("Writer", "Override prompt")
        assert registry.get("Writer") == "Override prompt"
    
    def test_clear_override(self, prompts_dir, config_path):
        """Test clearing a runtime override."""
        from blackboard.prompts import PromptRegistry
        
        config_path.write_text(json.dumps({
            "Writer": "Config prompt"
        }))
        
        registry = PromptRegistry(str(prompts_dir), str(config_path))
        registry.set("Writer", "Override")
        
        assert registry.get("Writer") == "Override"
        
        registry.clear_override("Writer")
        assert registry.get("Writer") == "Config prompt"
    
    def test_list_keys(self, prompts_dir, config_path):
        """Test listing all available prompt keys."""
        from blackboard.prompts import PromptRegistry
        
        # Create config
        config_path.write_text(json.dumps({
            "Writer": "...",
            "Critic": "..."
        }))
        
        # Create template directory and file
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "Validator.jinja2").write_text("template content")
        
        registry = PromptRegistry(str(prompts_dir), str(config_path))
        registry.set("Custom", "runtime override")
        
        keys = registry.list_keys()
        
        assert keys.get("Writer") == "config"
        assert keys.get("Critic") == "config"
        assert keys.get("Validator") == "template"
        assert keys.get("Custom") == "override"
    
    def test_save_config(self, prompts_dir, config_path):
        """Test saving config with overrides merged."""
        from blackboard.prompts import PromptRegistry
        
        config_path.write_text(json.dumps({
            "Writer": "original"
        }))
        
        registry = PromptRegistry(str(prompts_dir), str(config_path))
        registry.set("NewPrompt", "added")
        
        registry.save_config()
        
        # Reload and verify
        saved = json.loads(config_path.read_text())
        assert saved["Writer"] == "original"
        assert saved["NewPrompt"] == "added"
    
    def test_reload(self, prompts_dir, config_path):
        """Test reloading config."""
        from blackboard.prompts import PromptRegistry
        
        config_path.write_text(json.dumps({"key1": "value1"}))
        
        registry = PromptRegistry(str(prompts_dir), str(config_path))
        assert registry.get("key1") == "value1"
        
        # Modify config externally
        config_path.write_text(json.dumps({"key1": "updated"}))
        
        registry.reload()
        assert registry.get("key1") == "updated"


# Test Jinja2 templates if available
jinja2 = pytest.importorskip("jinja2", reason="jinja2 required for template tests")


class TestJinja2Templates:
    """Tests for Jinja2 template functionality."""
    
    @pytest.fixture
    def prompts_dir(self, tmp_path):
        return tmp_path / "prompts"
    
    @pytest.fixture
    def config_path(self, tmp_path):
        return tmp_path / "blackboard.prompts.json"
    
    def test_load_jinja2_template(self, prompts_dir, config_path):
        """Test loading and rendering Jinja2 template."""
        from blackboard.prompts import PromptRegistry
        
        # Create template
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "Writer.jinja2").write_text("""
You are a Writer.
{% if instructions %}
Task: {{ instructions }}
{% endif %}
""".strip())
        
        registry = PromptRegistry(str(prompts_dir), str(config_path))
        
        result = registry.get("Writer", {"instructions": "Write a haiku"})
        
        assert "You are a Writer" in result
        assert "Task: Write a haiku" in result
    
    def test_template_with_loops(self, prompts_dir, config_path):
        """Test Jinja2 template with loops."""
        from blackboard.prompts import PromptRegistry
        
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "supervisor.jinja2").write_text("""
Workers available:
{% for worker in workers %}
- {{ worker }}
{% endfor %}
""".strip())
        
        registry = PromptRegistry(str(prompts_dir), str(config_path))
        
        result = registry.get("supervisor", {"workers": ["Writer", "Critic", "Validator"]})
        
        assert "Writer" in result
        assert "Critic" in result
        assert "Validator" in result
    
    def test_get_template_for_inspection(self, prompts_dir, config_path):
        """Test getting raw template for patching/inspection."""
        from blackboard.prompts import PromptRegistry
        
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "Test.jinja2").write_text("Template: {{ value }}")
        
        registry = PromptRegistry(str(prompts_dir), str(config_path))
        
        template = registry.get_template("Test")
        assert template is not None
        
        # Render directly
        result = template.render(value="direct")
        assert result == "Template: direct"


class TestHelperFunctions:
    """Tests for scaffolding helper functions."""
    
    def test_create_default_prompts_dir(self, tmp_path):
        """Test creating default prompts directory."""
        from blackboard.prompts import create_default_prompts_dir
        
        prompts_path = tmp_path / "prompts"
        create_default_prompts_dir(str(prompts_path))
        
        assert prompts_path.exists()
        assert (prompts_path / "example.jinja2").exists()
    
    def test_create_default_config(self, tmp_path):
        """Test creating default config file."""
        from blackboard.prompts import create_default_config
        
        config_path = tmp_path / "blackboard.prompts.json"
        create_default_config(str(config_path))
        
        assert config_path.exists()
        
        content = json.loads(config_path.read_text())
        assert "_comment" in content
