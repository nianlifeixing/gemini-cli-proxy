"""
Gemini client module

Handles interaction with Gemini CLI tool
"""

import asyncio
import logging
from typing import List, Optional, AsyncGenerator
from .models import ChatMessage
from .config import config

logger = logging.getLogger(__name__)


class GeminiClient:
    """Gemini CLI client"""
    
    def __init__(self):
        self.semaphore = asyncio.Semaphore(config.max_concurrency)
    
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Execute chat completion request
        
        Args:
            messages: List of chat messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            **kwargs: Other parameters
            
        Returns:
            Response text from Gemini CLI
            
        Raises:
            asyncio.TimeoutError: Timeout error
            subprocess.CalledProcessError: Command execution error
        """
        async with self.semaphore:
            return await self._execute_gemini_command(
                messages, temperature, max_tokens, **kwargs
            )
    
    async def chat_completion_stream(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Execute streaming chat completion request (fake streaming implementation)
        
        Args:
            messages: List of chat messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            **kwargs: Other parameters
            
        Yields:
            Response text chunks split by lines
        """
        # First get complete response
        full_response = await self.chat_completion(
            messages, temperature, max_tokens, **kwargs
        )
        
        # Split by lines and yield one by one
        lines = full_response.split('\n')
        for line in lines:
            if line.strip():  # Skip empty lines
                yield line.strip()
                # Add small delay to simulate streaming effect
                await asyncio.sleep(0.05)
    
    async def _execute_gemini_command(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Execute Gemini CLI command
        
        Args:
            messages: List of chat messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            **kwargs: Other parameters
            
        Returns:
            Command output result
        """
        # Build command arguments
        cmd_args = [config.gemini_command]
        
        # Build prompt text (simplified implementation: combine all messages)
        prompt = self._build_prompt(messages)
        
        # Use --prompt parameter to pass prompt text
        cmd_args.extend(["--prompt", prompt])
        
        # Note: Real gemini CLI doesn't support temperature and max_tokens parameters
        # We ignore these parameters here but log them
        if temperature is not None:
            logger.debug(f"Ignoring temperature parameter: {temperature} (gemini CLI doesn't support)")
        if max_tokens is not None:
            logger.debug(f"Ignoring max_tokens parameter: {max_tokens} (gemini CLI doesn't support)")
        
        logger.debug(f"Executing command: {' '.join(cmd_args)}")
        
        try:
            # Use asyncio to execute subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for command execution to complete with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout
            )
            
            # Check return code
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8').strip()
                raise RuntimeError(f"Gemini CLI execution failed (exit code: {process.returncode}): {error_msg}")
            
            # Return standard output
            result = stdout.decode('utf-8').strip()
            logger.debug(f"Command executed successfully, output length: {len(result)}")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Gemini CLI command timeout ({config.timeout}s)")
            raise
        except Exception as e:
            logger.error(f"Error executing Gemini CLI command: {e}")
            raise
    
    def _build_prompt(self, messages: List[ChatMessage]) -> str:
        """
        Build prompt text
        
        Args:
            messages: List of chat messages
            
        Returns:
            Formatted prompt text
        """
        # Simplified implementation: format all messages by role
        prompt_parts = []
        
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"System: {message.content}")
            elif message.role == "user":
                prompt_parts.append(f"User: {message.content}")
            elif message.role == "assistant":
                prompt_parts.append(f"Assistant: {message.content}")
        
        return "\n".join(prompt_parts)


# Global client instance
gemini_client = GeminiClient() 