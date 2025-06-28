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
import re
from typing import List, Optional, AsyncGenerator, Tuple
from .models import ChatMessage
from .config import config

logger = logging.getLogger('gemini_cli_proxy')


class GeminiClient:
    """Gemini CLI client"""
    
    def __init__(self):
        self.semaphore = asyncio.Semaphore(config.max_concurrency)
    
    def _simplify_error_message(self, raw_error: str) -> Optional[str]:
        """
        Convert Gemini CLI error messages to more readable user-friendly messages
        
        Args:
            raw_error: Raw error message from Gemini CLI
            
        Returns:
            Simplified error message, or None if the error cannot be recognized
        """
        if not raw_error:
            return None
            
        lower_err = raw_error.lower()
        
        # Check for rate limiting related keywords
        rate_limit_indicators = [
            "code\": 429",
            "status 429", 
            "ratelimitexceeded",
            "resource_exhausted",
            "quota exceeded",
            "quota metric",
            "requests per day",
            "requests per minute",
            "limit exceeded"
        ]
        
        if any(keyword in lower_err for keyword in rate_limit_indicators):
            return "Gemini CLI rate limit exceeded. Please run `gemini` directly to check."
        
        return None

    async def chat_completion(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        user_locale: str = "en-US",
        **kwargs
    ) -> str:
        """
        Execute chat completion request
        
        Args:
            messages: List of chat messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            user_locale: User locale for response language guidance
            **kwargs: Other parameters
            
        Returns:
            Response text from Gemini CLI
            
        Raises:
            asyncio.TimeoutError: Timeout error
            subprocess.CalledProcessError: Command execution error
        """
        async with self.semaphore:
            return await self._execute_gemini_command(
                messages, temperature, max_tokens, user_locale, **kwargs
            )
    
    async def chat_completion_stream(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        user_locale: str = "en-US",
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Execute streaming chat completion request (fake streaming implementation)
        
        Args:
            messages: List of chat messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            user_locale: User locale for response language guidance
            **kwargs: Other parameters
            
        Yields:
            Response text chunks split by lines
        """
        # First get complete response
        full_response = await self.chat_completion(
            messages, temperature, max_tokens, user_locale, **kwargs
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
        user_locale: str = "en-US",
        **kwargs
    ) -> str:
        """
        Execute Gemini CLI command
        
        Args:
            messages: List of chat messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            user_locale: User locale for response language guidance
            **kwargs: Other parameters
            
        Returns:
            Command output result
        """
        # Build command arguments and get temporary files
        prompt, temp_files = self._build_prompt_with_images(messages, user_locale)
        
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
                
                # Try to simplify error message to more user-friendly format
                simplified_msg = self._simplify_error_message(error_msg)
                if simplified_msg:
                    logger.warning(f"Gemini CLI error (simplified): {simplified_msg}")
                    raise RuntimeError(simplified_msg)
                else:
                    logger.warning(f"Gemini CLI execution failed: {error_msg}")
                    raise RuntimeError(f"Gemini CLI execution failed (exit code: {process.returncode}): {error_msg}")
            
            # Return standard output
            result = stdout.decode('utf-8').strip()
            logger.debug(f"Gemini CLI response: {result}")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Gemini CLI command timeout ({config.timeout}s)")
            raise RuntimeError(f"Gemini CLI execution timeout ({config.timeout} seconds), please retry later or check your network connection")
        except RuntimeError:
            # Re-raise already processed RuntimeError
            raise
        except Exception as e:
            logger.error(f"Error executing Gemini CLI command: {e}")
            raise RuntimeError(f"Error executing Gemini CLI command: {str(e)}")
        finally:
            # Clean up temporary files (skip in debug mode)
            if not config.debug:
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
    
    def _build_prompt_with_images(self, messages: List[ChatMessage], user_locale: str) -> Tuple[str, List[str]]:
        """
        Build prompt text with image processing
        
        Args:
            messages: List of chat messages
            user_locale: User locale for response language guidance
            
        Returns:
            Tuple of (formatted prompt text, list of temporary file paths)
        """
        prompt_parts = []
        temp_files = []
        
        has_image_file_tags = False
        
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
                            content_parts.append(f"<image_file>{temp_file_path}</image_file>")
                            has_image_file_tags = True
                        else:
                            # For regular URLs, we'll just pass them through for now
                            # TODO: Download and save remote images if needed
                            content_parts.append(f"<image_url>{url}</image_url>")
                
                combined_content = " ".join(content_parts)
                if message.role == "system":
                    prompt_parts.append(f"System: {combined_content}")
                elif message.role == "user":
                    prompt_parts.append(f"User: {combined_content}")
                elif message.role == "assistant":
                    prompt_parts.append(f"Assistant: {combined_content}")

        # Add the system prompt at the beginning
        system_prompt = ""
        if user_locale:
            system_prompt += f"Please respond in locale `{user_locale}`."
        if has_image_file_tags:
            system_prompt += "For `image_file` tags in the conversation, please read the image from the specified file path instead of taking screenshots."

        if system_prompt:
            prompt_parts.insert(0, f"System: {system_prompt}")
        
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