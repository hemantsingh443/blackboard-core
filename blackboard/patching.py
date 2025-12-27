"""
Patching Module for Delta Protocol

Provides utilities for incremental artifact updates using Search-Replace patches.
This approach is more robust than line-numbered diffs because LLMs can reliably
match exact text strings even when context window limits truncate surrounding lines.

Example:
    from blackboard.patching import SearchReplacePatch, apply_patches
    
    original = "def hello():\n    print('world')"
    patches = [
        SearchReplacePatch(
            search="print('world')",
            replace="print('universe')"
        )
    ]
    result = apply_patches(original, patches)
    print(result.content)  # def hello():\n    print('universe')
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any, Dict
import re


class PatchOperation(str, Enum):
    """Type of patch operation."""
    SEARCH_REPLACE = "search_replace"  # Find and replace exact text
    APPEND = "append"                   # Add content at end
    PREPEND = "prepend"                 # Add content at start
    INSERT_AFTER = "insert_after"       # Insert after a marker
    INSERT_BEFORE = "insert_before"     # Insert before a marker
    DELETE = "delete"                   # Remove matching text


@dataclass
class SearchReplacePatch:
    """
    A single search-replace patch operation.
    
    This is the primary patching primitive. Rather than relying on line numbers
    (which LLMs get wrong), we use exact string matching.
    
    Attributes:
        search: The exact text to find (must be unique in the content)
        replace: The text to substitute
        operation: The type of operation (default: SEARCH_REPLACE)
        occurrence: Which occurrence to replace (0=first, -1=all, n=nth)
        description: Optional human-readable description of the change
        
    Example:
        # Simple replacement
        patch = SearchReplacePatch(
            search="def old_name(",
            replace="def new_name("
        )
        
        # Append to file
        patch = SearchReplacePatch(
            search="",
            replace="\\n# New content",
            operation=PatchOperation.APPEND
        )
    """
    search: str
    replace: str
    operation: PatchOperation = PatchOperation.SEARCH_REPLACE
    occurrence: int = 0  # 0=first, -1=all, n=nth occurrence
    description: str = ""
    
    def validate(self) -> Optional[str]:
        """
        Validate the patch configuration.
        
        Returns:
            None if valid, error message string if invalid
        """
        if self.operation == PatchOperation.SEARCH_REPLACE:
            if not self.search:
                return "SEARCH_REPLACE requires non-empty 'search' field"
            if self.search == self.replace:
                return "search and replace are identical (no-op)"
        elif self.operation in (PatchOperation.INSERT_AFTER, PatchOperation.INSERT_BEFORE):
            if not self.search:
                return f"{self.operation.value} requires a 'search' marker"
        elif self.operation == PatchOperation.DELETE:
            if not self.search:
                return "DELETE requires non-empty 'search' field"
        return None


@dataclass
class PatchResult:
    """
    Result of applying patches to content.
    
    Attributes:
        success: Whether all patches applied successfully
        content: The modified content (or original if failed)
        applied_count: Number of patches that were applied
        failed_patches: List of (patch_index, error_message) for failures
        changes_made: List of descriptions of changes that were applied
    """
    success: bool
    content: str
    applied_count: int = 0
    failed_patches: List[tuple] = field(default_factory=list)
    changes_made: List[str] = field(default_factory=list)
    
    def __bool__(self) -> bool:
        return self.success


def apply_patches(
    content: str,
    patches: List[SearchReplacePatch],
    fail_on_first_error: bool = False,
    validate_unique: bool = True
) -> PatchResult:
    """
    Apply a list of patches to content.
    
    Patches are applied sequentially in order. Each patch operates on the
    result of the previous patch, enabling chained modifications.
    
    Args:
        content: The original content to patch
        patches: List of patches to apply in order
        fail_on_first_error: If True, stop on first failure
        validate_unique: If True, require search strings to be unique
        
    Returns:
        PatchResult with the modified content and status
        
    Example:
        result = apply_patches(
            "Hello world",
            [SearchReplacePatch(search="world", replace="universe")]
        )
        assert result.success
        assert result.content == "Hello universe"
    """
    if not patches:
        return PatchResult(success=True, content=content, applied_count=0)
    
    current_content = content
    applied_count = 0
    failed_patches = []
    changes_made = []
    
    for i, patch in enumerate(patches):
        # Validate patch
        validation_error = patch.validate()
        if validation_error:
            failed_patches.append((i, validation_error))
            if fail_on_first_error:
                return PatchResult(
                    success=False,
                    content=content,  # Return original on failure
                    applied_count=applied_count,
                    failed_patches=failed_patches
                )
            continue
        
        # Apply the patch
        result, error = _apply_single_patch(
            current_content, patch, validate_unique
        )
        
        if error:
            failed_patches.append((i, error))
            if fail_on_first_error:
                return PatchResult(
                    success=False,
                    content=content,
                    applied_count=applied_count,
                    failed_patches=failed_patches
                )
        else:
            current_content = result
            applied_count += 1
            desc = patch.description or f"{patch.operation.value}: '{patch.search[:30]}...'" if len(patch.search) > 30 else f"{patch.operation.value}: '{patch.search}'"
            changes_made.append(desc)
    
    return PatchResult(
        success=len(failed_patches) == 0,
        content=current_content,
        applied_count=applied_count,
        failed_patches=failed_patches,
        changes_made=changes_made
    )


def _apply_single_patch(
    content: str,
    patch: SearchReplacePatch,
    validate_unique: bool
) -> tuple:
    """
    Apply a single patch to content.
    
    Returns:
        (new_content, error_message) - error_message is None on success
    """
    op = patch.operation
    
    if op == PatchOperation.APPEND:
        return content + patch.replace, None
    
    if op == PatchOperation.PREPEND:
        return patch.replace + content, None
    
    if op == PatchOperation.DELETE:
        if patch.search not in content:
            return None, f"DELETE target not found: '{patch.search[:50]}...'"
        return content.replace(patch.search, "", 1 if patch.occurrence == 0 else -1), None
    
    # For operations that need to find the search string
    if patch.search not in content:
        return None, f"Search string not found: '{patch.search[:50]}...'" if len(patch.search) > 50 else f"Search string not found: '{patch.search}'"
    
    # Check uniqueness if required
    if validate_unique and patch.occurrence == 0:
        count = content.count(patch.search)
        if count > 1:
            return None, f"Search string found {count} times (must be unique). Use occurrence=-1 to replace all, or make search more specific."
    
    if op == PatchOperation.SEARCH_REPLACE:
        if patch.occurrence == -1:
            # Replace all occurrences
            return content.replace(patch.search, patch.replace), None
        elif patch.occurrence == 0:
            # Replace first occurrence
            return content.replace(patch.search, patch.replace, 1), None
        else:
            # Replace nth occurrence
            return _replace_nth(content, patch.search, patch.replace, patch.occurrence), None
    
    if op == PatchOperation.INSERT_AFTER:
        idx = content.find(patch.search)
        insert_pos = idx + len(patch.search)
        return content[:insert_pos] + patch.replace + content[insert_pos:], None
    
    if op == PatchOperation.INSERT_BEFORE:
        idx = content.find(patch.search)
        return content[:idx] + patch.replace + content[idx:], None
    
    return None, f"Unknown operation: {op}"


def _replace_nth(content: str, search: str, replace: str, n: int) -> str:
    """Replace the nth occurrence of search with replace."""
    parts = content.split(search)
    if len(parts) <= n:
        return content  # Not enough occurrences
    return search.join(parts[:n]) + replace + search.join(parts[n:])


# =============================================================================
# Artifact Mutation Types (for WorkerOutput integration)
# =============================================================================

@dataclass
class ArtifactMutation:
    """
    A mutation to apply to an existing artifact.
    
    This is returned by Workers instead of a full artifact replacement
    when only a small change is needed.
    
    Attributes:
        artifact_id: ID of the artifact to mutate
        patches: List of patches to apply
        description: Human-readable summary of the mutation
        
    Example:
        return WorkerOutput(
            mutations=[
                ArtifactMutation(
                    artifact_id="abc-123",
                    patches=[
                        SearchReplacePatch(
                            search="old_function",
                            replace="new_function"
                        )
                    ],
                    description="Renamed function"
                )
            ]
        )
    """
    artifact_id: str
    patches: List[SearchReplacePatch]
    description: str = ""
    
    def apply_to(self, content: str, fail_on_first_error: bool = True) -> PatchResult:
        """Apply this mutation's patches to content."""
        return apply_patches(content, self.patches, fail_on_first_error)
