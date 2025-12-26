"""Tests for CLI commands."""

import pytest
import json
import subprocess
import sys
from pathlib import Path


class TestCLIHelp:
    """Tests for CLI help commands."""
    
    def test_main_help(self):
        """Test main help output."""
        result = subprocess.run(
            [sys.executable, "-m", "blackboard", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "blackboard" in result.stdout.lower() or "usage" in result.stdout.lower()
    
    def test_init_help(self):
        """Test init command help."""
        result = subprocess.run(
            [sys.executable, "-m", "blackboard", "init", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "--prompts-dir" in result.stdout
        assert "--config-path" in result.stdout
    
    def test_optimize_help(self):
        """Test optimize command help."""
        result = subprocess.run(
            [sys.executable, "-m", "blackboard", "optimize", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "run" in result.stdout
        assert "review" in result.stdout
    
    def test_optimize_run_help(self):
        """Test optimize run command help."""
        result = subprocess.run(
            [sys.executable, "-m", "blackboard", "optimize", "run", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "--session-id" in result.stdout
        assert "--db-path" in result.stdout
    
    def test_version_command(self):
        """Test version command."""
        result = subprocess.run(
            [sys.executable, "-m", "blackboard", "version"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "1." in result.stdout  # Flexible version check


class TestCLIInit:
    """Tests for blackboard init command."""
    
    def test_init_creates_prompts_dir(self, tmp_path):
        """Test init creates prompts directory."""
        result = subprocess.run(
            [sys.executable, "-m", "blackboard", "init",
             "--prompts-dir", str(tmp_path / "prompts"),
             "--config-path", str(tmp_path / "blackboard.prompts.json")],
            capture_output=True,
            text=True,
            cwd=str(tmp_path)
        )
        assert result.returncode == 0
        assert (tmp_path / "prompts").exists()
        assert (tmp_path / "prompts" / "example.jinja2").exists()
    
    def test_init_creates_config(self, tmp_path):
        """Test init creates config file."""
        result = subprocess.run(
            [sys.executable, "-m", "blackboard", "init",
             "--prompts-dir", str(tmp_path / "prompts"),
             "--config-path", str(tmp_path / "blackboard.prompts.json")],
            capture_output=True,
            text=True,
            cwd=str(tmp_path)
        )
        assert result.returncode == 0
        config_path = tmp_path / "blackboard.prompts.json"
        assert config_path.exists()
        
        # Verify it's valid JSON
        content = json.loads(config_path.read_text())
        assert "_comment" in content


class TestCLIOptimize:
    """Tests for blackboard optimize commands."""
    
    def test_optimize_run_stub(self, tmp_path):
        """Test optimize run prints stub message."""
        result = subprocess.run(
            [sys.executable, "-m", "blackboard", "optimize", "run",
             "--session-id", "test-session",
             "--db-path", str(tmp_path / "test.db")],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Running optimizer" in result.stdout
        assert "test-session" in result.stdout
    
    def test_optimize_review_no_patches(self, tmp_path):
        """Test optimize review when no patches file exists."""
        result = subprocess.run(
            [sys.executable, "-m", "blackboard", "optimize", "review",
             "--patches-file", str(tmp_path / "nonexistent.json")],
            capture_output=True,
            text=True
        )
        assert result.returncode == 1
        assert "No patches file found" in result.stdout
    
    def test_optimize_review_empty_patches(self, tmp_path):
        """Test optimize review with empty patches file."""
        patches_file = tmp_path / "patches.json"
        patches_file.write_text("[]")
        
        result = subprocess.run(
            [sys.executable, "-m", "blackboard", "optimize", "review",
             "--patches-file", str(patches_file)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "No pending patches" in result.stdout
    
    def test_optimize_review_with_patches(self, tmp_path):
        """Test optimize review displays patches."""
        patches_file = tmp_path / "patches.json"
        patches_file.write_text(json.dumps([
            {
                "worker_name": "Writer",
                "prompt_key": "Writer",
                "original": "old prompt",
                "original_hash": "abc123",
                "proposed": "new prompt",
                "reasoning": "Improved clarity for better output generation"
            },
            {
                "worker_name": "Critic",
                "prompt_key": "Critic",
                "original": "old critic",
                "original_hash": "def456",
                "proposed": "new critic",
                "reasoning": "Added quality constraints"
            }
        ]))
        
        result = subprocess.run(
            [sys.executable, "-m", "blackboard", "optimize", "review",
             "--patches-file", str(patches_file)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "2 pending patch" in result.stdout
        assert "Writer" in result.stdout
        assert "Critic" in result.stdout


class TestCLIServe:
    """Tests for blackboard serve command help (not actually starting server)."""
    
    def test_serve_help(self):
        """Test serve command help."""
        result = subprocess.run(
            [sys.executable, "-m", "blackboard", "serve", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout


class TestCLIUI:
    """Tests for blackboard ui command help."""
    
    def test_ui_help(self):
        """Test ui command help."""
        result = subprocess.run(
            [sys.executable, "-m", "blackboard", "ui", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "--api-url" in result.stdout
        assert "--headless" in result.stdout
