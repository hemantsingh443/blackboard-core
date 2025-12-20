"""
Browser Worker

Wraps Playwright for extracting content from web pages.
"""

import logging
from typing import Any, Dict, Optional

from pydantic import Field

from blackboard.protocols import Worker, WorkerInput, WorkerOutput
from blackboard.state import Blackboard, Artifact

logger = logging.getLogger("blackboard.stdlib.browser")


class BrowserInput(WorkerInput):
    """Input schema for BrowserWorker."""
    url: str = Field(..., description="The URL to navigate to")
    selector: Optional[str] = Field(default=None, description="CSS selector to extract (optional)")
    wait_for: Optional[str] = Field(default=None, description="CSS selector to wait for before extraction")
    timeout: int = Field(default=30000, description="Timeout in milliseconds")
    extract_links: bool = Field(default=False, description="Whether to extract all links on the page")
    screenshot: bool = Field(default=False, description="Whether to take a screenshot")


class BrowserWorker(Worker):
    """
    Browser automation worker using Playwright.
    
    Navigates to URLs, extracts content, and can take screenshots.
    Requires playwright to be installed and browsers to be set up:
        pip install playwright
        playwright install chromium
    
    .. warning::
        Resource Management: By default, BrowserWorker creates a new browser
        instance on first use and reuses it for subsequent calls. In long-running
        processes (like `blackboard serve`), you should:
        
        1. Create ONE BrowserWorker instance and reuse it across all runs
        2. Use `async with browser_worker:` context manager for automatic cleanup
        3. Or call `await browser_worker._cleanup()` when done
        
        The browser instance is NOT automatically closed between runs to avoid
        the overhead of repeatedly launching Chromium.
    
    Args:
        name: Worker name (default: "Browser")
        description: Worker description
        headless: Run browser in headless mode (default: True)
        browser_type: Browser to use ("chromium", "firefox", "webkit")
        close_browser_after_run: If True, close browser after each run (slower but safer)
        
    Example:
        # Option 1: Persistent browser (recommended for API servers)
        browser = BrowserWorker()
        orchestrator = Orchestrator(llm=my_llm, workers=[browser])
        # Browser stays open across multiple runs
        
        # Option 2: Auto-cleanup with context manager
        async with BrowserWorker() as browser:
            orchestrator = Orchestrator(llm=my_llm, workers=[browser])
            await orchestrator.run(goal="...")
        # Browser is automatically closed
        
        # Option 3: Close after each run (slower, for memory-constrained envs)
        browser = BrowserWorker(close_browser_after_run=True)
    """
    
    name = "Browser"
    description = "Navigates to web pages and extracts content"
    input_schema = BrowserInput
    parallel_safe = False  # Browser state is shared
    
    def __init__(
        self,
        name: str = "Browser",
        description: str = "Navigates to web pages and extracts content",
        headless: bool = True,
        browser_type: str = "chromium",
        close_browser_after_run: bool = False
    ):
        self.name = name
        self.description = description
        self.headless = headless
        self.browser_type = browser_type
        self.close_browser_after_run = close_browser_after_run
        self._playwright = None
        self._browser = None
        self._browser_launch_count = 0
    
    async def _ensure_browser(self):
        """Ensure browser is started."""
        if self._browser is not None:
            return
        
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "playwright is required for BrowserWorker. "
                "Install with: pip install playwright && playwright install chromium"
            )
        
        self._playwright = await async_playwright().start()
        
        if self.browser_type == "chromium":
            self._browser = await self._playwright.chromium.launch(headless=self.headless)
        elif self.browser_type == "firefox":
            self._browser = await self._playwright.firefox.launch(headless=self.headless)
        elif self.browser_type == "webkit":
            self._browser = await self._playwright.webkit.launch(headless=self.headless)
        else:
            raise ValueError(f"Unknown browser type: {self.browser_type}")
        
        self._browser_launch_count += 1
        logger.info(f"[{self.name}] Launched {self.browser_type} (launch #{self._browser_launch_count})")
    
    async def _cleanup(self):
        """Clean up browser resources."""
        if self._browser:
            try:
                await self._browser.close()
            except Exception as e:
                logger.warning(f"[{self.name}] Error closing browser: {e}")
            self._browser = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception as e:
                logger.warning(f"[{self.name}] Error stopping playwright: {e}")
            self._playwright = None
        logger.debug(f"[{self.name}] Browser resources cleaned up")
    
    async def run(
        self,
        state: Blackboard,
        inputs: Optional[BrowserInput] = None
    ) -> WorkerOutput:
        """Navigate to URL and extract content."""
        if not inputs or not inputs.url:
            return WorkerOutput(
                metadata={"error": "url is required"}
            )
        
        logger.info(f"[{self.name}] Navigating to {inputs.url}")
        
        try:
            await self._ensure_browser()
            
            page = await self._browser.new_page()
            
            try:
                # Navigate to URL
                await page.goto(inputs.url, timeout=inputs.timeout)
                
                # Wait for selector if specified
                if inputs.wait_for:
                    await page.wait_for_selector(inputs.wait_for, timeout=inputs.timeout)
                
                # Extract content
                if inputs.selector:
                    elements = await page.query_selector_all(inputs.selector)
                    content_parts = []
                    for el in elements:
                        text = await el.text_content()
                        if text:
                            content_parts.append(text.strip())
                    content = "\n\n".join(content_parts)
                else:
                    # Get full page text content
                    content = await page.evaluate("() => document.body.innerText")
                
                # Extract links if requested
                links = []
                if inputs.extract_links:
                    link_elements = await page.query_selector_all("a[href]")
                    for link in link_elements:
                        href = await link.get_attribute("href")
                        text = await link.text_content()
                        if href:
                            links.append({
                                "url": href,
                                "text": (text or "").strip()[:100]
                            })
                
                # Take screenshot if requested
                screenshot_data = None
                if inputs.screenshot:
                    screenshot_data = await page.screenshot(type="png", full_page=False)
                
                # Get page metadata
                title = await page.title()
                url = page.url
                
                metadata: Dict[str, Any] = {
                    "title": title,
                    "url": url,
                    "selector": inputs.selector,
                    "content_length": len(content)
                }
                
                if links:
                    metadata["links"] = links[:50]  # Limit links
                    metadata["total_links"] = len(links)
                
                if screenshot_data:
                    metadata["has_screenshot"] = True
                    # Store screenshot as base64
                    import base64
                    metadata["screenshot_base64"] = base64.b64encode(screenshot_data).decode()
                
                logger.info(f"[{self.name}] Extracted {len(content)} chars from {title}")
                
                return WorkerOutput(
                    artifact=Artifact(
                        type="web_content",
                        content=content,
                        creator=self.name,
                        metadata=metadata
                    )
                )
                
            finally:
                await page.close()
                
                # Optionally close browser after each run (for memory-constrained envs)
                if self.close_browser_after_run:
                    await self._cleanup()
                
        except Exception as e:
            logger.error(f"[{self.name}] Browser error: {e}")
            return WorkerOutput(
                metadata={
                    "error": str(e),
                    "url": inputs.url
                }
            )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_browser()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup()
