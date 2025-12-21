"""
Blackboard Reference UI

A Streamlit-based dashboard for interacting with the Blackboard API.

Features:
- Chat interface for setting goals
- Live event feed via SSE
- Artifact viewer (Markdown, Code, Images)
- Human-in-the-loop modal for paused runs

Usage:
    # Start the API server
    blackboard serve my_module:create_orchestrator
    
    # In another terminal, start the UI
    blackboard ui

Requirements:
    pip install streamlit httpx sseclient-py
"""

import streamlit as st
import httpx
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any, List


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_API_URL = "http://localhost:8000"


# =============================================================================
# API Client
# =============================================================================

class BlackboardClient:
    """Client for Blackboard API."""
    
    def __init__(self, base_url: str = DEFAULT_API_URL):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)
    
    def health_check(self) -> bool:
        """Check if API is healthy."""
        try:
            response = self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    def start_run(self, goal: str, max_steps: int = 20) -> Dict[str, Any]:
        """Start a new run."""
        response = self.client.post(
            f"{self.base_url}/runs",
            json={"goal": goal, "max_steps": max_steps}
        )
        response.raise_for_status()
        return response.json()
    
    def get_run(self, run_id: str) -> Dict[str, Any]:
        """Get run status."""
        response = self.client.get(f"{self.base_url}/runs/{run_id}")
        response.raise_for_status()
        return response.json()
    
    def resume_run(self, run_id: str, user_input: Optional[Dict] = None) -> Dict[str, Any]:
        """Resume a paused run."""
        response = self.client.post(
            f"{self.base_url}/runs/{run_id}/resume",
            json={"user_input": user_input or {}}
        )
        response.raise_for_status()
        return response.json()
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """List all runs."""
        response = self.client.get(f"{self.base_url}/runs")
        response.raise_for_status()
        data = response.json()
        # API returns {"runs": [...], "total": N}
        return data.get("runs", []) if isinstance(data, dict) else data
    
    def close(self):
        """Close the client."""
        self.client.close()


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render the sidebar with configuration."""
    st.sidebar.title("⚡ Blackboard UI")
    st.sidebar.markdown("---")
    
    # API Configuration
    st.sidebar.subheader("🔧 Configuration")
    api_url = st.sidebar.text_input(
        "API URL",
        value=st.session_state.get("api_url", DEFAULT_API_URL),
        help="URL of the Blackboard API server"
    )
    st.session_state.api_url = api_url
    
    # Connection status
    client = BlackboardClient(api_url)
    if client.health_check():
        st.sidebar.success("✅ Connected")
    else:
        st.sidebar.error("❌ Not connected")
    client.close()
    
    st.sidebar.markdown("---")
    
    # Run history
    st.sidebar.subheader("📋 Run History")
    try:
        client = BlackboardClient(api_url)
        runs = client.list_runs()
        client.close()
        
        # Show newest 5 runs (reverse order)
        recent_runs = runs[-5:][::-1] if runs else []
        
        for run in recent_runs:
            status = run.get("status", "")
            status_emoji = {
                "completed": "✅",
                "done": "✅",
                "failed": "❌",
                "paused": "⏸️",
                "generating": "🔄",
                "pending": "⏳",
                "running": "🔄"
            }.get(status, "❓")
            
            # Show truncated goal instead of ID
            goal_preview = run.get('goal', '')[:20] + "..." if len(run.get('goal', '')) > 20 else run.get('goal', '')
            
            if st.sidebar.button(
                f"{status_emoji} {goal_preview}",
                key=f"run_{run.get('id')}"
            ):
                st.session_state.current_run_id = run.get("id")
                st.rerun()
    except Exception as e:
        st.sidebar.caption(f"Could not load runs: {e}")


def render_chat_interface():
    """Render the main chat interface."""
    st.header("💬 Goal Setting")
    
    # Goal input
    with st.form("goal_form", clear_on_submit=True):
        goal = st.text_area(
            "What would you like to accomplish?",
            placeholder="Write a blog post about AI agents...",
            height=100
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            max_steps = st.slider("Max Steps", 5, 50, 20)
        with col2:
            submitted = st.form_submit_button("🚀 Start", use_container_width=True)
        
        if submitted and goal:
            try:
                client = BlackboardClient(st.session_state.get("api_url", DEFAULT_API_URL))
                result = client.start_run(goal, max_steps)
                client.close()
                
                st.session_state.current_run_id = result.get("run_id")
                st.success(f"Run started: {result.get('run_id')}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start run: {e}")


def render_run_status():
    """Render the current run status."""
    run_id = st.session_state.get("current_run_id")
    if not run_id:
        st.info("No active run. Start a new goal above.")
        return
    
    st.header("📊 Run Status")
    
    try:
        client = BlackboardClient(st.session_state.get("api_url", DEFAULT_API_URL))
        run_data = client.get_run(run_id)
        client.close()
        
        # Status display
        status = run_data.get("status", "unknown")
        status_colors = {
            "completed": "green",
            "done": "green",
            "failed": "red",
            "paused": "orange",
            "generating": "blue",
            "pending": "gray",
            "running": "blue"
        }
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", status.upper())
        with col2:
            st.metric("Steps", run_data.get("step_count", 0) or 0)
        with col3:
            artifacts_count = run_data.get("artifacts_count", 0) or len(run_data.get("artifacts", []))
            st.metric("Artifacts", artifacts_count)
        
        # Show the goal
        st.markdown(f"**Goal:** {run_data.get('goal', 'No goal')}")
        
        # Progress indicator
        if status in ("generating", "running", "pending"):
            st.progress(100)
            st.info("🔄 Run in progress... Refresh to see updates.")
        
        # Human-in-the-loop handling
        if status == "paused":
            render_human_loop(run_id, run_data)
        
        # Show last artifact preview if available
        last_artifact = run_data.get("last_artifact")
        if last_artifact:
            st.subheader("📦 Latest Output")
            preview = last_artifact.get("preview", "")
            creator = last_artifact.get("creator", "Unknown")
            st.markdown(f"**From {creator}:**")
            st.markdown(preview)
        
        # Full artifacts if available
        artifacts = run_data.get("artifacts", [])
        if artifacts:
            render_artifacts(artifacts)
        
    except Exception as e:
        st.error(f"Failed to fetch run: {e}")


def render_human_loop(run_id: str, run_data: Dict):
    """Render human-in-the-loop input form."""
    st.warning("⏸️ Run paused - Human input required")
    
    pending = run_data.get("pending_input", {})
    question = pending.get("question", "Please provide input:")
    
    st.markdown(f"**Question:** {question}")
    
    with st.form("human_input_form"):
        user_response = st.text_area(
            "Your response",
            placeholder="Type your response here...",
            height=100
        )
        
        submitted = st.form_submit_button("📤 Submit & Resume", use_container_width=True)
        
        if submitted and user_response:
            try:
                client = BlackboardClient(st.session_state.get("api_url", DEFAULT_API_URL))
                client.resume_run(run_id, {"response": user_response})
                client.close()
                st.success("Response submitted! Resuming run...")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to resume: {e}")


def render_artifacts(artifacts: List[Dict]):
    """Render artifacts in tabs."""
    if not artifacts:
        return
    
    st.subheader("📦 Artifacts")
    
    # Create tabs for different artifact types
    text_artifacts = []
    code_artifacts = []
    image_artifacts = []
    
    for artifact in artifacts:
        artifact_type = artifact.get("type", "text")
        if artifact_type in ("code", "code_result"):
            code_artifacts.append(artifact)
        elif artifact_type in ("screenshot", "image"):
            image_artifacts.append(artifact)
        else:
            text_artifacts.append(artifact)
    
    tabs = []
    tab_names = []
    
    if text_artifacts:
        tab_names.append(f"📝 Text ({len(text_artifacts)})")
        tabs.append(text_artifacts)
    if code_artifacts:
        tab_names.append(f"💻 Code ({len(code_artifacts)})")
        tabs.append(code_artifacts)
    if image_artifacts:
        tab_names.append(f"🖼️ Images ({len(image_artifacts)})")
        tabs.append(image_artifacts)
    
    if not tab_names:
        return
    
    selected_tabs = st.tabs(tab_names)
    
    for tab, artifact_list in zip(selected_tabs, tabs):
        with tab:
            for artifact in artifact_list:
                with st.expander(
                    f"{artifact.get('type', 'artifact')} - {artifact.get('creator', 'unknown')}",
                    expanded=len(artifact_list) == 1
                ):
                    content = artifact.get("content", "")
                    artifact_type = artifact.get("type", "text")
                    
                    if artifact_type in ("code", "code_result"):
                        st.code(content, language="python")
                    elif artifact_type in ("screenshot", "image"):
                        # For images, content might be base64 or a path
                        st.image(content)
                    else:
                        st.markdown(content)
                    
                    # Metadata
                    metadata = artifact.get("metadata", {})
                    if metadata:
                        st.caption(f"Metadata: {json.dumps(metadata, default=str)[:200]}")


def render_live_feed():
    """Render real-time event feed."""
    run_id = st.session_state.get("current_run_id")
    if not run_id:
        return
    
    st.subheader("📡 Live Feed")
    st.caption("Events from the current run")
    
    # Placeholder for live events
    feed_container = st.empty()
    
    # Note: True SSE streaming requires more complex handling
    # This is a simplified polling approach
    with feed_container.container():
        st.info("Live feed coming soon. Refresh to see updates.")


# =============================================================================
# Main App
# =============================================================================

def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Blackboard UI",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL
    if "current_run_id" not in st.session_state:
        st.session_state.current_run_id = None
    
    # Render components
    render_sidebar()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_chat_interface()
        render_run_status()
    
    with col2:
        render_live_feed()
    
    # Footer
    st.markdown("---")
    st.caption("Blackboard Reference UI • Built with Streamlit")


if __name__ == "__main__":
    main()
