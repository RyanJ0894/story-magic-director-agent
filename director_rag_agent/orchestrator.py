"""
Director Orchestrator - Simplified Version

Focus: Ensure the Director Agent completes all steps including save_to_supabase
"""

import logging
import re
from typing import AsyncGenerator
from typing_extensions import override

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DirectorOrchestrator(BaseAgent):
    """Orchestrator that routes to appropriate agent."""
    
    director: LlmAgent
    max_retries: int
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(
        self,
        name: str,
        director: LlmAgent,
        max_retries:int
    ):
        super().__init__(
            name=name,
            director=director,
            sub_agents=[director],
            max_retries=max_retries
        )
    
    
    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Main orchestration."""
        async for event in self.director.run_async(ctx):
            # Log tool calls
            if hasattr(event, 'content') and event.content:
                content_str = str(event.content)
            yield event

    @override
    async def _run_live_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Live mode."""
        async for event in self.director.run_live(ctx):
            yield event