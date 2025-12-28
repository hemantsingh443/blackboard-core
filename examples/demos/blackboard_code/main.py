#!/usr/bin/env python
"""
🔧 Blackboard Code - Orchestrator-Centric Multi-Agent IDE

Fixes:
- Proper markdown rendering with Rich
- Syntax highlighting for code
- Simple approval bar (no preview)
- True diff-based fixes
- Context persistence across follow-up messages
"""

import asyncio
import signal
import sys
import os
import tempfile
import webbrowser
import difflib
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

load_dotenv()


# ============================================================
# Role definitions
# ============================================================
class Role(Enum):
    ORCHESTRATOR = ("🎯", "#9966ff", "Orchestrator")
    PLANNER = ("📋", "#00d4ff", "Planner")
    CODER = ("💻", "#00ff88", "Coder")
    REVIEWER = ("🔍", "#ffd700", "Reviewer")
    FIXER = ("🔧", "#ff6b6b", "Fixer")
    RUNNER = ("⚡", "#00ffff", "Runner")
    USER = ("👤", "#ffffff", "You")


# ============================================================
# Session Context
# ============================================================
@dataclass
class SessionContext:
    goal: str = ""
    conversation: List[Dict] = field(default_factory=list)
    plan: Optional[str] = None
    plan_steps: List[Dict] = field(default_factory=list)
    plan_approved: bool = False
    current_step: int = 0  # Track progress
    files: Dict[str, str] = field(default_factory=dict)
    file_versions: Dict[str, List[str]] = field(default_factory=dict)
    phase: str = "idle"
    
    def add_message(self, role: str, content: str):
        self.conversation.append({"role": role, "content": content, "time": datetime.now().isoformat()})
    
    def get_progress(self) -> str:
        if not self.plan_steps:
            return "No plan yet"
        completed = len([s for s in self.plan_steps[:self.current_step] if s.get("done")])
        total = len(self.plan_steps)
        return f"{completed}/{total} steps done"
    
    def save_file(self, path: str, content: str, sandbox: Path):
        # Track versions
        if path in self.files:
            if path not in self.file_versions:
                self.file_versions[path] = []
            self.file_versions[path].append(self.files[path])
        
        self.files[path] = content
        full = sandbox / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
    
    def get_context_for_llm(self) -> str:
        """Full context for LLM continuity."""
        parts = [f"GOAL: {self.goal}"]
        
        if self.plan:
            parts.append(f"\nAPPROVED PLAN:\n{self.plan}")
        
        if self.plan_steps:
            parts.append(f"\nPROGRESS: {self.get_progress()}")
            for i, s in enumerate(self.plan_steps):
                status = "✓" if s.get("done") else "○"
                parts.append(f"  {status} Step {i+1}: {s.get('name', s.get('cmd', '?'))}")
        
        if self.files:
            parts.append(f"\nCREATED FILES:")
            for f in self.files:
                parts.append(f"  - {f}")
        
        return "\n".join(parts)


# ============================================================
# Terminal Manager - VISIBLE Terminal Windows
# ============================================================
class TerminalManager:
    """
    Manages VISIBLE terminal windows that both AI and user can access.
    Spawns actual cmd.exe/Windows Terminal windows.
    """
    
    def __init__(self, sandbox: Path, on_output=None):
        self.sandbox = sandbox
        self.on_output = on_output
        self.terminals: Dict[str, dict] = {}  # name -> {bat_file, log_file}
        self.command_history: List[dict] = []
        
        # Create terminal scripts directory
        self.scripts_dir = sandbox / ".terminals"
        self.scripts_dir.mkdir(exist_ok=True)
    
    def open_terminal(self, name: str = "main", cwd: str = None) -> str:
        """
        Open a new visible terminal window.
        Returns the path to the terminal's command file.
        """
        work_dir = Path(cwd) if cwd else self.sandbox
        
        # Create batch file for this terminal
        bat_file = self.scripts_dir / f"{name}.bat"
        log_file = self.scripts_dir / f"{name}.log"
        cmd_file = self.scripts_dir / f"{name}_commands.txt"
        
        # Create the terminal batch file
        terminal_script = f'''@echo off
title Blackboard Terminal [{name}]
cd /d "{work_dir}"
color 0A
echo ===============================================
echo   BLACKBOARD TERMINAL: {name.upper()}
echo   Working Dir: {work_dir}
echo ===============================================
echo.
echo Type commands or wait for AI to send them...
echo Commands will be read from: {cmd_file}
echo Output logged to: {log_file}
echo.

:loop
if exist "{cmd_file}" (
    echo.
    echo --- Executing commands from AI ---
    for /f "delims=" %%a in ({cmd_file}) do (
        echo ^> %%a
        call %%a
        echo.
    )
    del "{cmd_file}" 2>nul
    echo --- Done ---
    echo.
)
timeout /t 2 /nobreak >nul
goto loop
'''
        
        bat_file.write_text(terminal_script, encoding="utf-8")
        
        # Store terminal info
        self.terminals[name] = {
            "bat_file": str(bat_file),
            "log_file": str(log_file),
            "cmd_file": str(cmd_file),
            "cwd": str(work_dir)
        }
        
        # Launch the terminal
        if sys.platform == "win32":
            # Try Windows Terminal first, fallback to cmd
            try:
                os.system(f'start wt -d "{work_dir}" cmd /k "{bat_file}"')
            except:
                os.system(f'start cmd /k "{bat_file}"')
        else:
            os.system(f'gnome-terminal -- bash -c "cd {work_dir} && bash"')
        
        self._emit(f"[green]✓ Opened terminal window [{name}][/]")
        self._emit(f"[dim]  Working dir: {work_dir}[/]")
        
        return str(bat_file)
    
    def send_command(self, cmd: str, name: str = "main") -> bool:
        """
        Send a command to a terminal window (will be executed automatically).
        """
        if name not in self.terminals:
            self.open_terminal(name)
        
        cmd_file = Path(self.terminals[name]["cmd_file"])
        
        # Append command to the command file
        with open(cmd_file, "a", encoding="utf-8") as f:
            f.write(cmd + "\n")
        
        self._emit(f"[cyan]→ [{name}] $ {cmd}[/]")
        
        self.command_history.append({
            "cmd": cmd, 
            "name": name, 
            "time": datetime.now().isoformat()
        })
        
        return True
    
    def send_commands(self, cmds: List[str], name: str = "main") -> bool:
        """Send multiple commands to a terminal."""
        for cmd in cmds:
            self.send_command(cmd, name)
        return True
    
    async def run_and_wait(self, cmd: str, name: str = "main", timeout: int = 60) -> Tuple[bool, str]:
        """
        Run a command in hidden mode and wait for result.
        Use this for quick commands where we need the output.
        """
        if name not in self.terminals:
            cwd = self.sandbox
        else:
            cwd = Path(self.terminals[name]["cwd"])
        
        self._emit(f"[#00ffff]━━━ ⚡ RUN ({name}) ━━━[/]")
        self._emit(f"  $ {cmd}")
        
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
            output = stdout.decode() + stderr.decode()
            ok = proc.returncode == 0
            
            if output.strip():
                for line in output.strip().split("\n")[:15]:
                    self._emit(f"  {line}")
            
            self._emit(f"[{'green' if ok else 'red'}]  {'✓' if ok else '✗'} Exit: {proc.returncode}[/]")
            
            return ok, output
            
        except asyncio.TimeoutError:
            self._emit(f"[yellow]  ⏱ Timeout ({timeout}s)[/]")
            return False, "Timeout"
        except Exception as e:
            self._emit(f"[red]  Error: {e}[/]")
            return False, str(e)
    
    def list_terminals(self) -> List[str]:
        """List open terminals."""
        return list(self.terminals.keys())
    
    def get_cwd(self, name: str = "main") -> str:
        """Get working directory of a terminal."""
        if name in self.terminals:
            return self.terminals[name]["cwd"]
        return str(self.sandbox)
    
    def _emit(self, text: str):
        if self.on_output:
            self.on_output(text)



def main():
    import openai
    from textual.app import App, ComposeResult
    from textual.widgets import Header, Footer, Static, Button, RichLog, Input
    from textual.containers import Horizontal, Vertical, Container
    from textual.binding import Binding
    from textual.message import Message
    from textual import work
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.markup import escape
    
    from examples.demos.blackboard_code.config import (
        OPENROUTER_API_KEY, MODEL, MAX_TOKENS, OPENROUTER_HEADERS
    )
    
    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY not set!")
        sys.exit(1)
    
    sandbox = Path(tempfile.mkdtemp(prefix="blackboard_code_"))
    print(f"📁 Sandbox: {sandbox}")
    
    ctx = SessionContext()
    tui: Optional[App] = None
    
    approval_event = asyncio.Event()
    approval_result = False
    
    # ========================================
    # Messages
    # ========================================
    class WriteOutput(Message):
        def __init__(self, content: Any, role: Optional[Role] = None):
            super().__init__()
            self.content = content
            self.role = role
    
    class WriteMarkdown(Message):
        def __init__(self, md: str, role: Optional[Role] = None):
            super().__init__()
            self.md = md
            self.role = role
    
    class WriteCode(Message):
        def __init__(self, code: str, lang: str = "python", filename: str = ""):
            super().__init__()
            self.code = code
            self.lang = lang
            self.filename = filename
    
    class WriteDiff(Message):
        def __init__(self, filename: str, old: str, new: str):
            super().__init__()
            self.filename = filename
            self.old = old
            self.new = new
    
    class ShowApproval(Message):
        def __init__(self, text: str):
            super().__init__()
            self.text = text
    
    class HideApproval(Message):
        pass
    
    class UpdateAgents(Message):
        def __init__(self, status: Dict[str, str]):
            super().__init__()
            self.status = status
    
    class UpdateFiles(Message):
        pass
    
    # ========================================
    # Output helpers
    # ========================================
    def out(content: Any, role: Role = None):
        if tui:
            tui.post_message(WriteOutput(content, role))
    
    def out_md(md: str, role: Role = None):
        if tui:
            tui.post_message(WriteMarkdown(md, role))
    
    def out_code(code: str, lang: str = "python", filename: str = ""):
        if tui:
            tui.post_message(WriteCode(code, lang, filename))
    
    def out_diff(filename: str, old: str, new: str):
        if tui:
            tui.post_message(WriteDiff(filename, old, new))
    
    def agents(status: Dict[str, str]):
        if tui:
            tui.post_message(UpdateAgents(status))
    
    # ========================================
    # LLM
    # ========================================
    client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )
    
    async def llm_stream(prompt: str, system: str, role: Role) -> str:
        out(f"\n[{role.value[1]}]━━━ {role.value[0]} {role.value[2].upper()} ━━━[/]\n", role)
        
        try:
            stream = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS,
                stream=True,
                extra_headers=OPENROUTER_HEADERS
            )
            
            result = ""
            buffer = ""
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    result += token
                    buffer += token
                    
                    if len(buffer) > 80 or "\n" in buffer:
                        out(escape(buffer), role)
                        buffer = ""
            
            if buffer:
                out(escape(buffer), role)
            
            out("\n", role)
            return result
            
        except Exception as e:
            out(f"[red]Error: {e}[/]")
            return ""
    
    async def llm_call(prompt: str, system: str) -> str:
        try:
            r = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS,
                extra_headers=OPENROUTER_HEADERS
            )
            return r.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"
    
    # ========================================
    # Approval
    # ========================================
    async def ask_approval(text: str) -> bool:
        nonlocal approval_result
        if tui:
            tui.post_message(ShowApproval(text))
        approval_event.clear()
        await approval_event.wait()
        if tui:
            tui.post_message(HideApproval())
        return approval_result
    
    # ========================================
    # Agents
    # ========================================
    async def planner() -> str:
        agents({"planner": "working"})
        
        out(f"\n[#00d4ff]━━━ 📋 PLANNER ━━━[/]")
        out("[dim]Creating implementation plan...[/]\n")
        
        prompt = f"""Create a plan for: {ctx.goal}

Output as markdown:
## Files to Create
1. **filename.ext** - description
2. ...

## Commands (if needed)
- `command` - purpose

Be specific but concise."""
        
        # Non-streaming for clean markdown rendering
        plan = await llm_call(prompt, "You are a software architect.")
        
        # Render as markdown
        out_md(plan)
        
        ctx.plan = plan
        ctx.plan_steps = []
        
        # Parse
        for line in plan.split("\n"):
            if "**" in line and "." in line:
                m = re.search(r'\*\*([^*]+\.[a-z]+)\*\*', line, re.I)
                if m:
                    ctx.plan_steps.append({"type": "file", "name": m.group(1), "desc": line.split("-")[-1].strip() if "-" in line else "", "done": False})
            elif "`" in line and any(x in line.lower() for x in ["pip", "npm", "python"]):
                m = re.search(r'`([^`]+)`', line)
                if m:
                    ctx.plan_steps.append({"type": "cmd", "cmd": m.group(1), "done": False})
        
        if not ctx.plan_steps:
            ctx.plan_steps = [{"type": "file", "name": "main.py", "desc": "Main app", "done": False}]
        
        agents({"planner": "done"})
        return plan
    
    async def coder(filename: str, desc: str) -> str:
        agents({"coder": "working"})
        
        out(f"\n[#00ff88]━━━ 💻 CODER: {filename} ━━━[/]")
        out("[dim]Generating code...[/]\n")
        
        ext = filename.split(".")[-1] if "." in filename else "txt"
        lang_map = {"py": "python", "js": "javascript", "html": "html", "css": "css", "json": "json"}
        lang = lang_map.get(ext, "text")
        
        # Full context
        context = ctx.get_context_for_llm()
        
        existing = ""
        if ctx.files:
            existing = "\nEXISTING FILES:\n"
            for f, c in list(ctx.files.items())[:5]:
                existing += f"\n--- {f} ---\n{c[:600]}\n"
        
        prompt = f"""{context}
{existing}

Generate complete code for: {filename}
Description: {desc}

Output ONLY the code. Make it work with other project files."""
        
        # Non-streaming for clean syntax highlighting
        code = await llm_call(prompt, f"You write {lang} code.")
        
        # Clean
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:])
        if code.rstrip().endswith("```"):
            code = code.rstrip()[:-3]
        
        # Show with syntax highlighting
        out_code(code.strip(), lang, filename)
        
        agents({"coder": "done"})
        return code.strip()
    
    async def reviewer(filename: str, code: str) -> Tuple[bool, str]:
        agents({"reviewer": "working"})
        
        out(f"\n[#ffd700]━━━ 🔍 REVIEWER: {filename} ━━━[/]")
        
        context = ctx.get_context_for_llm()
        
        prompt = f"""{context}

FILE: {filename}
```
{code[:2000]}
```

Quick review. Reply with ONLY:
- PASS if acceptable
- FIX: one issue (only for critical bugs)

Be lenient. Multi-file projects may have cross-references."""
        
        response = await llm_call(prompt, "Lenient code reviewer.")
        
        out(f"  {escape(response[:100])}")
        
        passed = "PASS" in response.upper() or "FIX" not in response.upper()
        
        if passed:
            out("[green]  ✓ Approved[/]")
        else:
            out("[yellow]  ⚠ Needs fix[/]")
        
        agents({"reviewer": "done" if passed else "waiting"})
        return passed, response
    
    async def fixer(filename: str, code: str, issue: str) -> str:
        agents({"fixer": "working"})
        
        out(f"\n[#ff6b6b]━━━ 🔧 FIXER: {filename} ━━━[/]")
        out(f"  Issue: {escape(issue[:80])}")
        
        prompt = f"""Fix this issue: {issue}

Current code:
```
{code[:1500]}
```

Output ONLY the corrected code."""
        
        fixed = await llm_stream(prompt, "You fix code bugs.", Role.FIXER)
        
        # Clean
        if fixed.startswith("```"):
            lines = fixed.split("\n")
            fixed = "\n".join(lines[1:])
        if fixed.rstrip().endswith("```"):
            fixed = fixed.rstrip()[:-3]
        
        # Show diff
        out_diff(filename, code, fixed.strip())
        
        agents({"fixer": "done"})
        return fixed.strip()
    
    # Initialize terminal manager
    term_mgr: Optional[TerminalManager] = None
    
    def init_terminal_manager():
        nonlocal term_mgr
        term_mgr = TerminalManager(sandbox, on_output=lambda t: out(t))
    
    async def open_terminal_window(name: str = "main") -> bool:
        """Open a visible terminal window for user interaction."""
        if not term_mgr:
            init_terminal_manager()
        
        agents({"runner": "working"})
        term_mgr.open_terminal(name)
        agents({"runner": "done"})
        return True
    
    async def run_in_terminal(cmd: str, terminal: str = "main") -> bool:
        """
        Send a command to a visible terminal window.
        The user can see the command execute in real-time.
        """
        if not term_mgr:
            init_terminal_manager()
        
        if terminal not in term_mgr.list_terminals():
            term_mgr.open_terminal(terminal)
            await asyncio.sleep(1)  # Wait for terminal to open
        
        agents({"runner": "working"})
        
        # Adapt command for file paths
        files_list = list(ctx.files.keys())
        adapted_cmd = cmd
        
        # Fix requirements.txt path
        if "requirements" in cmd.lower() and "-r" in cmd:
            for f in files_list:
                if "requirements" in f.lower() and f.endswith(".txt"):
                    adapted_cmd = cmd.replace("requirements.txt", f)
                    out(f"[dim]  → Using: {f}[/]")
                    break
        
        term_mgr.send_command(adapted_cmd, terminal)
        agents({"runner": "done"})
        return True
    
    async def run_quick(cmd: str, timeout: int = 60) -> Tuple[bool, str]:
        """
        Run a command silently and return the output.
        Use for commands where AI needs to see the result.
        """
        if not term_mgr:
            init_terminal_manager()
        
        agents({"runner": "working"})
        ok, output = await term_mgr.run_and_wait(cmd, timeout=timeout)
        
        # Auto-fix on failure
        if not ok:
            out(f"[yellow]  🔧 Command failed, attempting fix...[/]")
            
            fix_prompt = f"""Command failed:
$ {cmd}

Error: {output[:500]}

Files: {', '.join(list(ctx.files.keys())[:10])}
OS: {sys.platform}

What is the corrected command? Reply with ONLY the command."""
            
            new_cmd = await llm_call(fix_prompt, "You fix commands.")
            new_cmd = new_cmd.strip().strip("`").split("\n")[0]
            
            if new_cmd and new_cmd != cmd and len(new_cmd) < 200:
                out(f"[cyan]  → Retrying: {new_cmd}[/]")
                ok, output = await term_mgr.run_and_wait(new_cmd, timeout=timeout)
        
        agents({"runner": "done"})
        return ok, output
    
    async def runner(cmd: str, terminal: str = "main", use_visible: bool = True) -> Tuple[bool, str]:
        """
        Main runner - chooses between visible terminal or quiet execution.
        
        Args:
            cmd: Command to run
            terminal: Terminal name for visible mode
            use_visible: If True, run in visible window; if False, run quietly
        """
        # Detect if this needs visible terminal (servers, interactive)
        needs_visible = any(x in cmd.lower() for x in [
            "flask run", "npm start", "npm run", "python -m http",
            "uvicorn", "gunicorn", "node server", "python app",
            "python main", "python manage.py runserver"
        ])
        
        if use_visible or needs_visible:
            await run_in_terminal(cmd, terminal)
            return True, f"Command sent to terminal [{terminal}]"
        else:
            return await run_quick(cmd)
    
    async def setup_project_terminals():
        """
        Open terminals for a typical project structure.
        """
        if not term_mgr:
            init_terminal_manager()
        
        # Check project structure
        has_backend = any("backend" in f or "server" in f or "api" in f for f in ctx.files)
        has_frontend = any("frontend" in f or "client" in f for f in ctx.files)
        
        if has_backend and has_frontend:
            out("[dim]Opening terminals for frontend + backend...[/]")
            term_mgr.open_terminal("backend", str(sandbox / "backend") if (sandbox / "backend").exists() else str(sandbox))
            await asyncio.sleep(0.5)
            term_mgr.open_terminal("frontend", str(sandbox / "frontend") if (sandbox / "frontend").exists() else str(sandbox))
        else:
            term_mgr.open_terminal("main")

    
    # ========================================
    # Orchestrator
    # ========================================
    async def classify_intent(user_input: str) -> Tuple[str, str]:
        """
        Use LLM to classify what the user wants to do.
        Returns (intent, details) where intent is one of:
        - PLAN: Need to create a new project/feature
        - RUN: Execute the application
        - FIX: Fix an issue with existing code
        - EDIT: Modify existing file
        - EXPLAIN: Answer a question
        - COMMAND: Run a specific shell command
        - CONTINUE: Continue previous work
        """
        context_info = ""
        if ctx.files:
            context_info = f"\nExisting project files: {list(ctx.files.keys())}"
            if ctx.plan:
                context_info += f"\nPrevious plan exists: {len(ctx.plan_steps)} steps, {ctx.get_progress()}"
        
        prompt = f"""Classify this user request:
"{user_input}"
{context_info}

What does the user want? Reply with ONE of these formats:
- PLAN: <description> - if they want to BUILD something new (app, feature, project)
- RUN: <file_or_command> - if they want to START/RUN/LAUNCH the application
- FIX: <filename> - if they want to FIX a bug or error in a specific file
- EDIT: <filename> <what_to_change> - if they want to MODIFY/CHANGE/ADD to existing code
- COMMAND: <shell_command> - if they want to run a SPECIFIC command (pip, npm, etc)
- EXPLAIN: <topic> - if they're asking a QUESTION or need explanation
- CONTINUE: - if they want to resume previous unfinished work

Reply with ONLY the classification, nothing else."""

        response = await llm_call(prompt, "You classify user intents accurately. Be concise.")
        response = response.strip()
        
        # Parse response
        if ":" in response:
            parts = response.split(":", 1)
            intent = parts[0].strip().upper()
            details = parts[1].strip() if len(parts) > 1 else ""
        else:
            intent = response.upper()
            details = ""
        
        return intent, details
    
    async def orchestrate(user_input: str):
        nonlocal approval_result
        
        ctx.add_message("user", user_input)
        out(f"\n[white bold]👤 You:[/] {escape(user_input)}\n")
        
        # Classify intent using LLM
        out(f"[#9966ff]🎯 ORCHESTRATOR: Analyzing request...[/]")
        intent, details = await classify_intent(user_input)
        out(f"[dim]  Intent: {intent} → {details[:50] if details else '(none)'}[/]")
        
        # Route to appropriate agent based on intent
        if intent == "RUN":
            await handle_run(details)
        elif intent == "FIX":
            await handle_fix(details, user_input)
        elif intent == "EDIT":
            await handle_edit(details, user_input)
        elif intent == "COMMAND":
            await handle_command(details)
        elif intent == "EXPLAIN":
            await handle_explain(user_input)
        elif intent == "CONTINUE":
            await handle_continue()
        else:  # PLAN or unknown
            await handle_plan(user_input)
    
    async def handle_run(details: str):
        """Handle RUN intent - start the application."""
        out(f"[#9966ff]🎯 ORCHESTRATOR: Running application...[/]")
        
        run_cmd = None
        
        # If details specifies a file, use it
        if details and details.endswith(".py"):
            run_cmd = f"python {details}"
        elif details and details.endswith(".js"):
            run_cmd = f"node {details}"
        
        # Auto-detect from project files
        if not run_cmd and ctx.files:
            # Flask
            for fname, content in ctx.files.items():
                if fname.endswith(".py") and ("Flask(" in content or "from flask" in content.lower()):
                    if "app.run" in content or "__main__" in content:
                        run_cmd = f"python {fname}"
                        break
            
            # Django
            if not run_cmd and "manage.py" in ctx.files:
                run_cmd = "python manage.py runserver"
            
            # Node
            if not run_cmd and "package.json" in ctx.files:
                run_cmd = "npm start"
            
            # Generic Python
            if not run_cmd:
                for fname in ["app.py", "main.py", "server.py", "run.py"]:
                    if fname in ctx.files:
                        run_cmd = f"python {fname}"
                        break
        
        if run_cmd:
            out(f"[dim]  Command: {run_cmd}[/]")
            approved = await ask_approval(f"Run: {run_cmd}?")
            if approved:
                await run_in_terminal(run_cmd)
                out(f"[green]✓ Started in terminal[/]")
        else:
            out(f"[yellow]Could not detect run command. Files: {list(ctx.files.keys())}[/]")
    
    async def handle_fix(details: str, user_input: str):
        """Handle FIX intent - fix a bug in code."""
        out(f"[#9966ff]🎯 ORCHESTRATOR: Fixing issue...[/]")
        
        # Find the file to fix
        target_file = details.strip()
        if target_file not in ctx.files:
            # Try to find it
            for fname in ctx.files:
                if target_file in fname or fname in target_file:
                    target_file = fname
                    break
        
        if target_file in ctx.files:
            code = ctx.files[target_file]
            fixed = await fixer(target_file, code, user_input)
            approved = await ask_approval(f"Update {target_file}?")
            if approved:
                ctx.save_file(target_file, fixed, sandbox)
                out(f"[green]✓ Fixed {target_file}[/]")
        else:
            out(f"[yellow]File not found: {target_file}[/]")
            out(f"[dim]Available: {list(ctx.files.keys())}[/]")
    
    async def handle_edit(details: str, user_input: str):
        """Handle EDIT intent - modify existing code."""
        out(f"[#9966ff]🎯 ORCHESTRATOR: Editing code...[/]")
        
        # Parse filename from details
        parts = details.split(maxsplit=1)
        target_file = parts[0] if parts else ""
        edit_desc = parts[1] if len(parts) > 1 else user_input
        
        if target_file not in ctx.files:
            for fname in ctx.files:
                if target_file in fname:
                    target_file = fname
                    break
        
        if target_file in ctx.files:
            code = ctx.files[target_file]
            # Use coder to regenerate with modifications
            new_code = await coder(target_file, f"Modify existing code: {edit_desc}")
            
            out_diff(target_file, code, new_code)
            approved = await ask_approval(f"Update {target_file}?")
            if approved:
                ctx.save_file(target_file, new_code, sandbox)
                out(f"[green]✓ Updated {target_file}[/]")
        else:
            out(f"[yellow]File not found, creating new...[/]")
            await handle_plan(user_input)
    
    async def handle_command(cmd: str):
        """Handle COMMAND intent - run shell command."""
        out(f"[#9966ff]🎯 ORCHESTRATOR: Running command...[/]")
        
        approved = await ask_approval(f"Run: {cmd}?")
        if approved:
            await run_in_terminal(cmd)
    
    async def handle_explain(question: str):
        """Handle EXPLAIN intent - answer a question."""
        out(f"[#9966ff]🎯 ORCHESTRATOR: Answering question...[/]")
        
        context = ctx.get_context_for_llm() if ctx.files else ""
        prompt = f"""{context}

User question: {question}

Provide a helpful, concise answer."""
        
        response = await llm_call(prompt, "You are a helpful coding assistant.")
        out_md(response)
    
    async def handle_continue():
        """Handle CONTINUE intent - resume previous work."""
        out(f"[#9966ff]🎯 ORCHESTRATOR: Continuing previous work...[/]")
        out(f"[dim]{ctx.get_progress()}[/]")
        
        if ctx.current_step < len(ctx.plan_steps):
            await execute_remaining()
        else:
            out("[yellow]No pending work to continue[/]")
    
    async def handle_plan(goal: str):
        """Handle PLAN intent - create new project/feature."""
        ctx.goal = goal
        ctx.phase = "planning"
        
        out(f"\n[#9966ff]🎯 ORCHESTRATOR: Creating plan...[/]\n")
        
        await planner()
        
        ctx.phase = "awaiting_approval"
        approved = await ask_approval(f"Approve plan? ({len(ctx.plan_steps)} steps)")
        
        if approved:
            ctx.plan_approved = True
            out("[green]✓ Plan approved[/]\n")
            await execute_remaining()
        else:
            out("[yellow]Plan rejected[/]")
            ctx.phase = "idle"

    
    async def execute_remaining():
        ctx.phase = "executing"
        
        while ctx.current_step < len(ctx.plan_steps):
            step = ctx.plan_steps[ctx.current_step]
            
            if step.get("done"):
                ctx.current_step += 1
                continue
            
            if step["type"] == "file":
                filename = step["name"]
                
                # Code
                code = await coder(filename, step.get("desc", ctx.goal))
                
                # Review
                passed, feedback = await reviewer(filename, code)
                
                # Fix loop
                attempts = 0
                while not passed and attempts < 2:
                    code = await fixer(filename, code, feedback)
                    passed, feedback = await reviewer(filename, code)
                    attempts += 1
                
                # Approval
                approved = await ask_approval(f"Create {filename}?")
                
                if approved:
                    ctx.save_file(filename, code, sandbox)
                    out(f"[green]✓ Created {filename}[/]")
                    if tui:
                        tui.post_message(UpdateFiles())
                else:
                    out(f"[dim]Skipped {filename}[/]")
                
                step["done"] = True
            
            elif step["type"] == "cmd":
                cmd = step["cmd"]
                
                approved = await ask_approval(f"Run: {cmd}?")
                
                if approved:
                    await runner(cmd)
                else:
                    out(f"[dim]Skipped: {cmd}[/]")
                
                step["done"] = True
            
            ctx.current_step += 1
        
        ctx.phase = "complete"
        out(f"\n[green bold]✅ COMPLETE! Created {len(ctx.files)} files.[/]")
        out("[dim]Type to continue or add features.[/]\n")
    
    # ========================================
    # TUI
    # ========================================
    class BlackboardCode(App):
        TITLE = "Blackboard Code"
        
        CSS = """
        Screen { background: #0a0a14; }
        #main { layout: horizontal; height: 100%; }
        #content { width: 80%; }
        #sidebar { width: 20%; background: #0f0f1a; border-left: solid #333; padding: 1; }
        
        #output { height: 1fr; padding: 1; }
        
        #approval-bar {
            height: 3;
            background: #2a2a40;
            border-top: solid #ffd700;
            padding: 0 1;
            display: none;
        }
        #approval-bar.show { display: block; }
        #approval-text { width: 1fr; }
        #approval-btns { layout: horizontal; width: auto; }
        #btn-yes { background: #00ff88; color: #000; }
        #btn-no { background: #ff4444; }
        
        #input-box { height: auto; background: #1a1a2e; padding: 1; border-top: solid #333; }
        #prompt { width: 100%; }
        
        .section-title { color: #ffd700; text-style: bold; margin-bottom: 1; }
        #agents-box { height: auto; }
        #files-box { height: auto; margin-top: 1; }
        
        #btn-box { margin-top: 1; }
        #btn-box Button { width: 100%; margin-bottom: 1; }
        #preview { background: #00ff88; color: #000; }
        #folder { background: #00d4ff; color: #000; }
        
        RichLog { background: transparent; }
        Header { background: #9966ff; }
        Footer { background: #0f0f1a; }
        """
        
        BINDINGS = [Binding("q", "quit", "Quit"), Binding("ctrl+o", "folder", "Folder")]
        
        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal(id="main"):
                with Vertical(id="content"):
                    yield RichLog(id="output", markup=True, wrap=True, auto_scroll=True, highlight=True)
                    with Horizontal(id="approval-bar"):
                        yield Static("", id="approval-text")
                        with Horizontal(id="approval-btns"):
                            yield Button("✅ Yes", id="btn-yes")
                            yield Button("❌ No", id="btn-no")
                    with Container(id="input-box"):
                        yield Input(placeholder="What should I build?", id="prompt")
                
                with Vertical(id="sidebar"):
                    yield Static("🤖 AGENTS", classes="section-title")
                    yield Static("", id="agents-box")
                    yield Static("📁 FILES", classes="section-title")
                    yield Static("[dim]No files[/]", id="files-box")
                    with Container(id="btn-box"):
                        yield Button("👁️ Preview", id="preview")
                        yield Button("📂 Folder", id="folder")
            yield Footer()
        
        def on_mount(self):
            nonlocal tui
            tui = self
            self.update_agents({})
            
            log = self.query_one("#output", RichLog)
            log.write(Panel(
                "[bold #9966ff]🔧 BLACKBOARD CODE[/]\n\n"
                "Multi-agent coding with context persistence.\n\n"
                "[dim]Try: Create a todo app[/]",
                border_style="#9966ff"
            ))
        
        def update_agents(self, status: Dict[str, str]):
            icons = {"idle": "⚫", "working": "🟢", "waiting": "🟡", "done": "✅"}
            lines = []
            for name in ["planner", "coder", "reviewer", "fixer", "runner"]:
                s = status.get(name, "idle")
                lines.append(f"{icons.get(s, '⚫')} {name.title()}")
            self.query_one("#agents-box", Static).update("\n".join(lines))
        
        def on_input_submitted(self, e):
            if e.input.id == "prompt":
                text = e.value.strip()
                if text:
                    e.input.value = ""
                    self.run_task(text)
        
        @work(exclusive=True)
        async def run_task(self, text: str):
            await orchestrate(text)
        
        def on_write_output(self, m: WriteOutput):
            log = self.query_one("#output", RichLog)
            if isinstance(m.content, str):
                log.write(m.content)
            else:
                log.write(m.content)
        
        def on_write_markdown(self, m: WriteMarkdown):
            log = self.query_one("#output", RichLog)
            log.write(Markdown(m.md))
        
        def on_write_code(self, m: WriteCode):
            log = self.query_one("#output", RichLog)
            if m.filename:
                log.write(f"[dim]─── {m.filename} ───[/]")
            log.write(Syntax(m.code, m.lang, theme="monokai", line_numbers=True, word_wrap=True))
        
        def on_write_diff(self, m: WriteDiff):
            log = self.query_one("#output", RichLog)
            log.write(f"[#ff6b6b]─── Diff: {m.filename} ───[/]")
            
            old_lines = m.old.splitlines(keepends=True)
            new_lines = m.new.splitlines(keepends=True)
            
            diff = difflib.unified_diff(old_lines, new_lines, lineterm="")
            for line in diff:
                line = line.rstrip()
                if line.startswith("+") and not line.startswith("+++"):
                    log.write(f"[green]{escape(line)}[/]")
                elif line.startswith("-") and not line.startswith("---"):
                    log.write(f"[red]{escape(line)}[/]")
                elif line.startswith("@@"):
                    log.write(f"[cyan]{escape(line)}[/]")
        
        def on_show_approval(self, m: ShowApproval):
            self.query_one("#approval-text", Static).update(f"[#ffd700]⚠️ {m.text}[/]")
            self.query_one("#approval-bar", Horizontal).add_class("show")
        
        def on_hide_approval(self, m: HideApproval):
            self.query_one("#approval-bar", Horizontal).remove_class("show")
        
        def on_update_agents(self, m: UpdateAgents):
            self.update_agents(m.status)
        
        def on_update_files(self, m: UpdateFiles):
            icons = {"py": "🐍", "html": "🌐", "css": "🎨", "js": "⚡"}
            if ctx.files:
                lines = []
                for f in sorted(ctx.files.keys()):
                    ext = f.split(".")[-1] if "." in f else ""
                    lines.append(f"{icons.get(ext, '📄')} {f}")
                self.query_one("#files-box", Static).update("\n".join(lines))
        
        def on_button_pressed(self, e):
            nonlocal approval_result
            if e.button.id == "btn-yes":
                approval_result = True
                approval_event.set()
            elif e.button.id == "btn-no":
                approval_result = False
                approval_event.set()
            elif e.button.id == "preview":
                for f in ctx.files:
                    if f.endswith(".html"):
                        webbrowser.open(f"file://{sandbox / f}")
                        return
            elif e.button.id == "folder":
                if sys.platform == "win32":
                    os.startfile(str(sandbox))
        
        def action_folder(self):
            if sys.platform == "win32":
                os.startfile(str(sandbox))
    
    BlackboardCode().run()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, lambda s, f: sys.exit(0))
    main()
