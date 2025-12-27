"""
Integration tests for artifact mutations in the Orchestrator.

Tests the complete workflow: Worker returns mutations -> Orchestrator applies them.
"""

import pytest
from blackboard import (
    Blackboard, 
    Artifact, 
    Status, 
    WorkerOutput,
    SearchReplacePatch,
    ArtifactMutation,
    PatchOperation,
)
from blackboard.protocols import Worker, WorkerInput


class MockEditorWorker(Worker):
    """Worker that returns mutations instead of full artifact replacements."""
    name = "Editor"
    description = "Applies targeted edits to existing artifacts"
    
    def __init__(self, target_artifact_id: str, patches: list):
        self._target_id = target_artifact_id
        self._patches = patches
    
    async def run(self, state: Blackboard, inputs=None) -> WorkerOutput:
        return WorkerOutput(
            mutations=[
                ArtifactMutation(
                    artifact_id=self._target_id,
                    patches=self._patches,
                    description="Applied edits"
                )
            ]
        )


class TestMutationIntegration:
    """Integration tests for the mutation workflow."""
    
    def test_worker_output_has_mutations(self):
        """Test WorkerOutput correctly reports mutations."""
        output = WorkerOutput(
            mutations=[
                ArtifactMutation(
                    artifact_id="test-id",
                    patches=[SearchReplacePatch(search="a", replace="b")]
                )
            ]
        )
        assert output.has_mutations()
        assert len(output.mutations) == 1
    
    def test_worker_output_empty_mutations(self):
        """Test WorkerOutput with no mutations."""
        output = WorkerOutput()
        assert not output.has_mutations()
    
    @pytest.mark.asyncio
    async def test_apply_mutation_to_artifact(self):
        """Test that mutations are applied to artifacts correctly."""
        from blackboard.core import Orchestrator
        from blackboard.patching import apply_patches
        
        # Create state with an artifact
        state = Blackboard(goal="Test mutations")
        original_content = """def hello():
    print("world")
    return True"""
        
        artifact = Artifact(
            type="code",
            content=original_content,
            creator="Writer"
        )
        state.add_artifact(artifact)
        artifact_id = state.artifacts[0].id
        
        # Create mutation
        patches = [
            SearchReplacePatch(
                search='print("world")',
                replace='print("universe")',
                description="Changed greeting"
            )
        ]
        
        # Apply directly (simulating what Orchestrator does)
        mutation = ArtifactMutation(
            artifact_id=artifact_id,
            patches=patches
        )
        
        result = apply_patches(state.artifacts[0].content, mutation.patches)
        assert result.success
        assert 'print("universe")' in result.content
        assert 'print("world")' not in result.content
    
    def test_mutation_with_append(self):
        """Test APPEND operation via mutation."""
        original = "def foo():\n    pass"
        patches = [
            SearchReplacePatch(
                search="",
                replace="\n\ndef bar():\n    pass",
                operation=PatchOperation.APPEND
            )
        ]
        
        from blackboard.patching import apply_patches
        result = apply_patches(original, patches)
        
        assert result.success
        assert "def foo()" in result.content
        assert "def bar()" in result.content
    
    def test_mutation_sequence(self):
        """Test applying multiple mutations to same artifact."""
        original = "Hello world, hello universe, hello everyone"
        
        # Multiple targeted edits
        patches = [
            SearchReplacePatch(search="Hello world", replace="Hi world"),
            SearchReplacePatch(search="hello universe", replace="hi universe"),
            SearchReplacePatch(search="hello everyone", replace="hi everyone"),
        ]
        
        from blackboard.patching import apply_patches
        result = apply_patches(original, patches)
        
        assert result.success
        assert result.content == "Hi world, hi universe, hi everyone"
        assert result.applied_count == 3
    
    def test_mutation_preserves_artifact_metadata(self):
        """Test that mutations preserve artifact metadata."""
        state = Blackboard(goal="Test")
        artifact = Artifact(
            type="code",
            content="old content",
            creator="Writer",
            metadata={"language": "python", "filename": "test.py"}
        )
        state.add_artifact(artifact)
        
        # Verify metadata preserved
        assert state.artifacts[0].metadata["language"] == "python"
        assert state.artifacts[0].metadata["filename"] == "test.py"


class TestCodeEditingScenario:
    """Real-world code editing scenarios."""
    
    def test_function_rename(self):
        """Test renaming a function via patches."""
        original = """class UserService:
    def get_user(self, id):
        return self.db.query(User).filter(id=id).first()
    
    def create_user(self, data):
        user = User(**data)
        return user"""
        
        patches = [
            SearchReplacePatch(
                search="def get_user(self, id):",
                replace="def get_user_by_id(self, user_id: int):",
                description="Rename and add type hint"
            ),
            SearchReplacePatch(
                search="filter(id=id)",
                replace="filter(id=user_id)",
                description="Update filter parameter"
            ),
        ]
        
        from blackboard.patching import apply_patches
        result = apply_patches(original, patches)
        
        assert result.success
        assert "get_user_by_id" in result.content
        assert "user_id: int" in result.content
        assert "filter(id=user_id)" in result.content
    
    def test_add_import(self):
        """Test adding an import statement."""
        original = """import os

def main():
    pass"""
        
        patches = [
            SearchReplacePatch(
                search="import os",
                replace="import os\nimport asyncio",
                description="Add asyncio import"
            ),
        ]
        
        from blackboard.patching import apply_patches
        result = apply_patches(original, patches)
        
        assert result.success
        assert "import asyncio" in result.content
    
    def test_remove_code_block(self):
        """Test removing a code block."""
        original = """def main():
    # TODO: Remove this debug code
    print("DEBUG: Starting main")
    # End debug code
    
    process()"""
        
        patches = [
            SearchReplacePatch(
                search="    # TODO: Remove this debug code\n    print(\"DEBUG: Starting main\")\n    # End debug code\n",
                replace="",
                operation=PatchOperation.DELETE
            ),
        ]
        
        from blackboard.patching import apply_patches
        result = apply_patches(original, patches)
        
        assert result.success
        assert "DEBUG" not in result.content
        assert "process()" in result.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
