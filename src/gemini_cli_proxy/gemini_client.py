"""
Gemini client module

Handles interaction with Gemini CLI tool
"""

import asyncio
import logging
import os
import tempfile
import uuid
import base64
from typing import List, Optional, AsyncGenerator, Tuple
from .models import ChatMessage
from .config import config

logger = logging.getLogger('gemini_cli_proxy')


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
        # Build command arguments and get temporary files
        prompt, temp_files = self._build_prompt_with_images(messages)
        
        cmd_args = [config.gemini_command]
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
            logger.debug(f"Gemini CLI response: {result}")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Gemini CLI command timeout ({config.timeout}s)")
            raise
        except Exception as e:
            logger.error(f"Error executing Gemini CLI command: {e}")
            raise
        finally:
            # Clean up temporary files (skip in debug mode)
            if not config.debug:
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
    
    def _build_prompt_with_images(self, messages: List[ChatMessage]) -> Tuple[str, List[str]]:
        """
        Build prompt text with image processing
        
        Args:
            messages: List of chat messages
            
        Returns:
            Tuple of (formatted prompt text, list of temporary file paths)
        """
        prompt_parts = []
        temp_files = []
        
        for i, message in enumerate(messages):
            if isinstance(message.content, str):
                # Simple string content
                if message.role == "system":
                    prompt_parts.append(f"System: {message.content}")
                elif message.role == "user":
                    prompt_parts.append(f"User: {message.content}")
                elif message.role == "assistant":
                    prompt_parts.append(f"Assistant: {message.content}")
            else:
                # List of content parts (vision support)
                content_parts = []
                
                for j, part in enumerate(message.content):
                    if part.type == "text" and part.text:
                        content_parts.append(part.text)
                    elif part.type == "image_url" and part.image_url:
                        url = part.image_url.get("url", "")
                        if url.startswith("data:"):
                            # Process base64 image
                            temp_file_path = self._save_base64_image(url)
                            temp_files.append(temp_file_path)
                            content_parts.append(f"<image>{temp_file_path}</image>")
                        else:
                            # For regular URLs, we'll just pass them through for now
                            # TODO: Download and save remote images if needed
                            content_parts.append(f"<image>{url}</image>")
                
                combined_content = " ".join(content_parts)
                if message.role == "system":
                    prompt_parts.append(f"System: {combined_content}")
                elif message.role == "user":
                    prompt_parts.append(f"User: {combined_content}")
                elif message.role == "assistant":
                    prompt_parts.append(f"Assistant: {combined_content}")
        
        final_prompt = "\n".join(prompt_parts)
        logger.debug(f"Prompt sent to Gemini CLI: {final_prompt}")
        
        return final_prompt, temp_files
    
    def _save_base64_image(self, data_url: str) -> str:
        """
        Save base64 image data to temporary file
        
        Args:
            data_url: Data URL in format "data:image/type;base64,..."
            
        Returns:
            Path to temporary file
            
        Raises:
            ValueError: Invalid data URL format
        """
        try:
            # Parse data URL
            if not data_url.startswith("data:"):
                raise ValueError("Invalid data URL format")
            
            # Extract MIME type and base64 data
            header, data = data_url.split(",", 1)
            mime_info = header.split(";")[0].split(":")[1]  # e.g., "image/png"
            
            # Determine file extension
            if "png" in mime_info.lower():
                ext = ".png"
            elif "jpeg" in mime_info.lower() or "jpg" in mime_info.lower():
                ext = ".jpg"
            elif "gif" in mime_info.lower():
                ext = ".gif"
            elif "webp" in mime_info.lower():
                ext = ".webp"
            else:
                ext = ".png"  # Default to PNG
            
            # Decode base64 data
            image_data = base64.b64decode(data)
            
            # Create .gemini-cli-proxy directory in project root
            project_root = os.getcwd()
            temp_dir = os.path.join(project_root, ".gemini-cli-proxy")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create temporary file in project directory
            filename = f"gemini_img_{uuid.uuid4().hex[:8]}{ext}"
            temp_file_path = os.path.join(temp_dir, filename)
            
            # Write image data
            with open(temp_file_path, 'wb') as f:
                f.write(image_data)
            
            return temp_file_path
            
        except Exception as e:
            logger.error(f"Error saving base64 image: {e}")
            raise ValueError(f"Failed to save base64 image: {e}")

    def _build_prompt(self, messages: List[ChatMessage]) -> str:
        """
        Build prompt text (legacy method, kept for compatibility)
        
        Args:
            messages: List of chat messages
            
        Returns:
            Formatted prompt text
        """
        prompt, _ = self._build_prompt_with_images(messages)
        return prompt


# Global client instance
gemini_client = GeminiClient() 