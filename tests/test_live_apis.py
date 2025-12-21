"""
Live Test Script for Blackboard SDK with Real APIs

Tests:
1. WebSearchWorker with Tavily API
2. Orchestrator with Gemini via LiteLLM
3. Simple multi-worker scenario

Requirements:
- TAVILY_API_KEY in environment
- GEMINI_API_KEY in environment
"""

import asyncio
import os
import sys

# Add parent to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from blackboard import Orchestrator, worker, Blackboard
from blackboard.llm import LiteLLMClient


def check_api_keys():
    """Check that required API keys are set."""
    keys = {
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    }
    
    missing = [k for k, v in keys.items() if not v]
    if missing:
        print(f"❌ Missing API keys: {', '.join(missing)}")
        print("Please set them in your .env file")
        return False
    
    print("✅ All API keys found")
    return True


async def test_gemini_llm():
    """Test basic LLM connection with Gemini."""
    print("\n" + "="*50)
    print("TEST 1: Gemini LLM Connection")
    print("="*50)
    
    try:
        llm = LiteLLMClient(model="gemini/gemini-2.0-flash-exp")
        response = await llm.generate("Say 'Hello from Gemini!' in exactly 5 words.")
        
        print(f"✅ Gemini Response: {response.content[:100]}...")
        print(f"   Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
        return True
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        return False


async def test_websearch_worker():
    """Test WebSearchWorker with Tavily."""
    print("\n" + "="*50)
    print("TEST 2: WebSearchWorker with Tavily")
    print("="*50)
    
    try:
        from blackboard.stdlib import WebSearchWorker
        
        worker = WebSearchWorker()
        state = Blackboard(goal="Test search")
        
        # Create a mock input
        from blackboard.protocols import WorkerInput
        
        class SearchInput(WorkerInput):
            query: str = "Python programming language"
        
        result = await worker.run(state, SearchInput(query="Python 3.12 new features"))
        
        if result.artifact:
            print(f"✅ Search successful!")
            print(f"   Content preview: {result.artifact.content[:200]}...")
            return True
        else:
            print(f"⚠️ No artifact returned")
            return False
            
    except ImportError as e:
        print(f"⚠️ WebSearchWorker import failed (missing httpx?): {e}")
        return False
    except Exception as e:
        print(f"❌ WebSearchWorker test failed: {e}")
        return False


async def test_simple_orchestrator():
    """Test a simple orchestrator with a custom worker."""
    print("\n" + "="*50)
    print("TEST 3: Simple Orchestrator with Custom Worker")
    print("="*50)
    
    try:
        @worker(name="Calculator", description="Performs simple math calculations")
        async def calculator(expression: str = "2+2") -> str:
            """Safely evaluate a math expression."""
            try:
                # Only allow safe math operations
                allowed = set("0123456789+-*/.(). ")
                if all(c in allowed for c in expression):
                    result = eval(expression)
                    return f"Result: {expression} = {result}"
                return f"Invalid expression: {expression}"
            except Exception as e:
                return f"Error: {e}"
        
        llm = LiteLLMClient(model="gemini/gemini-2.0-flash-exp")
        orchestrator = Orchestrator(llm=llm, workers=[calculator])
        
        result = await orchestrator.run(
            goal="Calculate 15 * 7 + 23",
            max_steps=3
        )
        
        print(f"✅ Orchestrator completed!")
        print(f"   Status: {result.status}")
        print(f"   Steps: {result.step_count}")
        if result.artifacts:
            print(f"   Last artifact: {result.artifacts[-1].content[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multi_worker_scenario():
    """Test orchestrator with multiple workers including web search."""
    print("\n" + "="*50)
    print("TEST 4: Multi-Worker Scenario (WebSearch + Summarizer)")
    print("="*50)
    
    try:
        from blackboard.stdlib import WebSearchWorker
        
        @worker(name="Summarizer", description="Summarizes text into key points")
        async def summarizer(text: str = "") -> str:
            """Summarize the provided text."""
            # Simple summarization - just return first 200 chars
            if len(text) > 200:
                return f"Summary: {text[:200]}... (truncated)"
            return f"Summary: {text}"
        
        llm = LiteLLMClient(model="gemini/gemini-2.0-flash-exp")
        
        orchestrator = Orchestrator(
            llm=llm,
            workers=[WebSearchWorker(), summarizer]
        )
        
        result = await orchestrator.run(
            goal="Search for 'what is Python asyncio' and summarize the findings",
            max_steps=5
        )
        
        print(f"✅ Multi-worker orchestrator completed!")
        print(f"   Status: {result.status}")
        print(f"   Steps: {result.step_count}")
        print(f"   Artifacts: {len(result.artifacts)}")
        return True
        
    except ImportError as e:
        print(f"⚠️ Skipping (missing dependencies): {e}")
        return True  # Not a failure, just skip
    except Exception as e:
        print(f"❌ Multi-worker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("="*60)
    print("BLACKBOARD SDK LIVE TESTS")
    print("="*60)
    
    if not check_api_keys():
        return 1
    
    results = []
    
    # Run tests
    results.append(("Gemini LLM", await test_gemini_llm()))
    results.append(("WebSearch Worker", await test_websearch_worker()))
    results.append(("Simple Orchestrator", await test_simple_orchestrator()))
    results.append(("Multi-Worker", await test_multi_worker_scenario()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
