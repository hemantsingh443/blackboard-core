"""
Unit tests for the patching module (Milestone 5.1).

Tests cover all patch operations, edge cases, and error handling.
"""

import pytest
from blackboard.patching import (
    SearchReplacePatch,
    PatchOperation,
    PatchResult,
    ArtifactMutation,
    apply_patches,
)


class TestSearchReplacePatch:
    """Tests for the SearchReplacePatch dataclass."""
    
    def test_basic_creation(self):
        """Test basic patch creation."""
        patch = SearchReplacePatch(search="foo", replace="bar")
        assert patch.search == "foo"
        assert patch.replace == "bar"
        assert patch.operation == PatchOperation.SEARCH_REPLACE
        assert patch.occurrence == 0
    
    def test_validation_empty_search(self):
        """Test validation fails for empty search in SEARCH_REPLACE."""
        patch = SearchReplacePatch(search="", replace="bar")
        error = patch.validate()
        assert error is not None
        assert "non-empty" in error
    
    def test_validation_identical_search_replace(self):
        """Test validation fails when search equals replace."""
        patch = SearchReplacePatch(search="foo", replace="foo")
        error = patch.validate()
        assert error is not None
        assert "identical" in error
    
    def test_validation_append_no_search_required(self):
        """Test APPEND operation doesn't require search."""
        patch = SearchReplacePatch(
            search="", 
            replace="new content",
            operation=PatchOperation.APPEND
        )
        error = patch.validate()
        assert error is None
    
    def test_validation_delete_requires_search(self):
        """Test DELETE operation requires search."""
        patch = SearchReplacePatch(
            search="",
            replace="",
            operation=PatchOperation.DELETE
        )
        error = patch.validate()
        assert error is not None


class TestApplyPatches:
    """Tests for the apply_patches function."""
    
    def test_empty_patches_list(self):
        """Test applying empty patches list returns original content."""
        result = apply_patches("hello world", [])
        assert result.success
        assert result.content == "hello world"
        assert result.applied_count == 0
    
    def test_simple_replacement(self):
        """Test basic search-replace."""
        patches = [SearchReplacePatch(search="world", replace="universe")]
        result = apply_patches("hello world", patches)
        assert result.success
        assert result.content == "hello universe"
        assert result.applied_count == 1
    
    def test_multiple_patches(self):
        """Test applying multiple patches in sequence."""
        patches = [
            SearchReplacePatch(search="hello", replace="hi"),
            SearchReplacePatch(search="world", replace="earth"),
        ]
        result = apply_patches("hello world", patches)
        assert result.success
        assert result.content == "hi earth"
        assert result.applied_count == 2
    
    def test_chained_patches(self):
        """Test patches that depend on each other."""
        patches = [
            SearchReplacePatch(search="foo", replace="bar"),
            SearchReplacePatch(search="bar", replace="baz"),  # Finds the result of previous patch
        ]
        result = apply_patches("foo", patches)
        assert result.success
        assert result.content == "baz"
    
    def test_search_not_found(self):
        """Test failure when search string not found."""
        patches = [SearchReplacePatch(search="missing", replace="replacement")]
        result = apply_patches("hello world", patches)
        assert not result.success
        assert result.applied_count == 0
        assert len(result.failed_patches) == 1
        assert "not found" in result.failed_patches[0][1]
    
    def test_non_unique_search_fails(self):
        """Test failure when search string appears multiple times."""
        patches = [SearchReplacePatch(search="a", replace="b")]
        result = apply_patches("banana", patches)
        assert not result.success
        assert "3 times" in result.failed_patches[0][1]
    
    def test_replace_all_occurrences(self):
        """Test replacing all occurrences with occurrence=-1."""
        patches = [SearchReplacePatch(search="a", replace="o", occurrence=-1)]
        result = apply_patches("banana", patches)
        assert result.success
        assert result.content == "bonono"
    
    def test_append_operation(self):
        """Test APPEND operation."""
        patches = [SearchReplacePatch(
            search="",
            replace="\n# appended",
            operation=PatchOperation.APPEND
        )]
        result = apply_patches("hello", patches)
        assert result.success
        assert result.content == "hello\n# appended"
    
    def test_prepend_operation(self):
        """Test PREPEND operation."""
        patches = [SearchReplacePatch(
            search="",
            replace="# prepended\n",
            operation=PatchOperation.PREPEND
        )]
        result = apply_patches("hello", patches)
        assert result.success
        assert result.content == "# prepended\nhello"
    
    def test_insert_after_operation(self):
        """Test INSERT_AFTER operation."""
        patches = [SearchReplacePatch(
            search="def foo():",
            replace="\n    # New comment",
            operation=PatchOperation.INSERT_AFTER
        )]
        result = apply_patches("def foo():\n    pass", patches)
        assert result.success
        assert result.content == "def foo():\n    # New comment\n    pass"
    
    def test_insert_before_operation(self):
        """Test INSERT_BEFORE operation."""
        patches = [SearchReplacePatch(
            search="def foo():",
            replace="# Decorator\n",
            operation=PatchOperation.INSERT_BEFORE
        )]
        result = apply_patches("def foo():\n    pass", patches)
        assert result.success
        assert result.content == "# Decorator\ndef foo():\n    pass"
    
    def test_delete_operation(self):
        """Test DELETE operation."""
        patches = [SearchReplacePatch(
            search="# TODO: remove this\n",
            replace="",
            operation=PatchOperation.DELETE
        )]
        result = apply_patches("# TODO: remove this\ndef foo():\n    pass", patches)
        assert result.success
        assert result.content == "def foo():\n    pass"
    
    def test_fail_on_first_error(self):
        """Test fail_on_first_error stops processing."""
        patches = [
            SearchReplacePatch(search="missing", replace="x"),  # Will fail
            SearchReplacePatch(search="hello", replace="hi"),   # Won't run
        ]
        result = apply_patches("hello world", patches, fail_on_first_error=True)
        assert not result.success
        assert result.applied_count == 0
        assert len(result.failed_patches) == 1
    
    def test_continue_on_error(self):
        """Test continuing after error when fail_on_first_error=False."""
        patches = [
            SearchReplacePatch(search="missing", replace="x"),  # Will fail
            SearchReplacePatch(search="hello", replace="hi"),   # Will succeed
        ]
        result = apply_patches("hello world", patches, fail_on_first_error=False)
        assert not result.success  # Overall failure due to first patch
        assert result.applied_count == 1  # But second patch applied
        assert result.content == "hi world"
    
    def test_multiline_content(self):
        """Test patching multiline content."""
        content = """def hello():
    print("world")
    return True"""
        
        patches = [SearchReplacePatch(
            search='print("world")',
            replace='print("universe")'
        )]
        
        result = apply_patches(content, patches)
        assert result.success
        assert 'print("universe")' in result.content
        assert 'print("world")' not in result.content
    
    def test_code_refactoring_scenario(self):
        """Test realistic code refactoring scenario."""
        original = """class UserService:
    def get_user(self, id):
        return self.db.query(User).filter(id=id).first()
    
    def create_user(self, data):
        user = User(**data)
        self.db.add(user)
        return user"""
        
        patches = [
            SearchReplacePatch(
                search="def get_user(self, id):",
                replace="def get_user(self, user_id: int):",
                description="Rename 'id' to 'user_id' and add type hint"
            ),
            SearchReplacePatch(
                search="filter(id=id)",
                replace="filter(id=user_id)",
                description="Update filter to use new parameter name"
            ),
        ]
        
        result = apply_patches(original, patches)
        assert result.success
        assert "user_id: int" in result.content
        assert "filter(id=user_id)" in result.content
        assert len(result.changes_made) == 2


class TestArtifactMutation:
    """Tests for the ArtifactMutation class."""
    
    def test_mutation_creation(self):
        """Test creating an ArtifactMutation."""
        mutation = ArtifactMutation(
            artifact_id="abc-123",
            patches=[SearchReplacePatch(search="old", replace="new")],
            description="Updated content"
        )
        assert mutation.artifact_id == "abc-123"
        assert len(mutation.patches) == 1
    
    def test_apply_to_method(self):
        """Test applying mutation to content."""
        mutation = ArtifactMutation(
            artifact_id="abc-123",
            patches=[
                SearchReplacePatch(search="hello", replace="hi"),
                SearchReplacePatch(search="world", replace="earth"),
            ]
        )
        
        result = mutation.apply_to("hello world")
        assert result.success
        assert result.content == "hi earth"
    
    def test_empty_mutation(self):
        """Test mutation with no patches."""
        mutation = ArtifactMutation(
            artifact_id="abc-123",
            patches=[]
        )
        
        result = mutation.apply_to("hello world")
        assert result.success
        assert result.content == "hello world"


class TestEdgeCases:
    """Edge case and stress tests."""
    
    def test_empty_content(self):
        """Test patching empty content."""
        patches = [SearchReplacePatch(
            search="",
            replace="new content",
            operation=PatchOperation.APPEND
        )]
        result = apply_patches("", patches)
        assert result.success
        assert result.content == "new content"
    
    def test_special_characters_in_search(self):
        """Test search with regex-special characters."""
        content = "price = $100.00"
        patches = [SearchReplacePatch(search="$100.00", replace="$200.00")]
        result = apply_patches(content, patches)
        assert result.success
        assert result.content == "price = $200.00"
    
    def test_unicode_content(self):
        """Test patching unicode content."""
        content = "Hello, 世界! 🌍"
        patches = [SearchReplacePatch(search="世界", replace="地球")]
        result = apply_patches(content, patches)
        assert result.success
        assert result.content == "Hello, 地球! 🌍"
    
    def test_large_content(self):
        """Test patching large content (simulated file)."""
        lines = [f"line {i}: content here" for i in range(1000)]
        content = "\n".join(lines)
        
        patches = [SearchReplacePatch(
            search="line 500: content here",
            replace="line 500: MODIFIED"
        )]
        
        result = apply_patches(content, patches)
        assert result.success
        assert "line 500: MODIFIED" in result.content
        assert result.content.count("\n") == 999  # 1000 lines = 999 newlines
    
    def test_whitespace_preservation(self):
        """Test that whitespace is preserved exactly."""
        content = "def foo():\n    x = 1\n    y = 2"
        patches = [SearchReplacePatch(
            search="    x = 1",
            replace="    x = 10"
        )]
        result = apply_patches(content, patches)
        assert result.success
        assert "    x = 10" in result.content
        assert result.content.startswith("def foo():\n    x = 10")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
