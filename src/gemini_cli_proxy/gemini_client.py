"""
Gemini client module

Handles interaction with Gemini CLI tool
"""

import asyncio
import logging
import os
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
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Execute chat completion request
        
        Args:
            messages: List of chat messages
            model: Model name to use
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
                messages, model, temperature, max_tokens, **kwargs
            )
    
    async def chat_completion_stream(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Execute streaming chat completion request (fake streaming implementation)
        
        Args:
            messages: List of chat messages
            model: Model name to use
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            **kwargs: Other parameters
            
        Yields:
            Response text chunks split by lines
        """
        # First get complete response
        full_response = await self.chat_completion(
            messages, model, temperature, max_tokens, **kwargs
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
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Execute Gemini CLI command
        
        Args:
            messages: List of chat messages
            model: Model name to use
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            **kwargs: Other parameters
            
        Returns:
            Command output result
        """
        # Build command arguments and get temporary files
        prompt, temp_files = self._build_prompt_with_images(messages)
        
        cmd_args = [config.gemini_command]
        cmd_args.extend(["-m", model])
        cmd_args.extend(["-p", prompt])

        # Add proxy support using gemini CLI's built-in --proxy option
        if config.use_proxy:
            proxy_url = f"http://{config.proxy_host}:{config.proxy_port}"
            cmd_args.extend(["--proxy", proxy_url])

        # Note: Real gemini CLI doesn't support temperature and max_tokens parameters
        # We ignore these parameters here but log them
        if temperature is not None:
            logger.debug(f"Ignoring temperature parameter: {temperature} (gemini CLI doesn't support)")
        if max_tokens is not None:
            logger.debug(f"Ignoring max_tokens parameter: {max_tokens} (gemini CLI doesn't support)")

        # Check if gemini command exists and get full path
        import shutil
        gemini_path = shutil.which(config.gemini_command)
        logger.debug(f"Gemini command path: {gemini_path}")

        # Use full path if available, otherwise use original command
        if gemini_path:
            cmd_args[0] = gemini_path

        logger.debug(f"Executing command: {' '.join(cmd_args)}")

        try:
            # Prepare environment
            env = os.environ.copy()

            logger.debug(f"About to create subprocess with args: {cmd_args}")

            # Use subprocess to execute command (fallback for Windows asyncio issues)
            try:
                import subprocess
                import platform

                logger.debug(f"Using subprocess.run for command execution")

                # Run subprocess synchronously but in thread pool to avoid blocking
                import concurrent.futures

                def run_subprocess():
                    if platform.system() == 'Windows':
                        # Use shell on Windows
                        shell_cmd = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd_args)
                        logger.debug(f"Running shell command: {shell_cmd}")
                        result = subprocess.run(
                            shell_cmd,
                            shell=True,
                            capture_output=True,
                            text=True,
                            env=env,
                            timeout=config.timeout
                        )
                    else:
                        # Use exec on Unix-like systems
                        result = subprocess.run(
                            cmd_args,
                            capture_output=True,
                            text=True,
                            env=env,
                            timeout=config.timeout
                        )
                    return result

                # Run in thread pool to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(executor, run_subprocess)

                logger.debug(f"Subprocess completed with return code: {result.returncode}")
                logger.debug(f"Stdout length: {len(result.stdout)}, Stderr length: {len(result.stderr)}")

                # Check return code
                if result.returncode != 0:
                    error_msg = result.stderr.strip()
                    logger.debug(f"Raw stderr: {repr(error_msg)}")

                    # Try to simplify error message to more user-friendly format
                    simplified_msg = self._simplify_error_message(error_msg)
                    if simplified_msg:
                        logger.warning(f"Gemini CLI error (simplified): {simplified_msg}")
                        raise RuntimeError(simplified_msg)
                    else:
                        logger.warning(f"Gemini CLI execution failed: {error_msg}")
                        raise RuntimeError(f"Gemini CLI execution failed (exit code: {result.returncode}): {error_msg}")

                # Return standard output
                response_text = result.stdout.strip()
                logger.debug(f"Gemini CLI response: {response_text}")
                return response_text

            except subprocess.TimeoutExpired:
                logger.error(f"Gemini CLI command timeout ({config.timeout}s)")
                raise RuntimeError(f"Gemini CLI execution timeout ({config.timeout} seconds), please retry later or check your network connection") from None
            except Exception as subprocess_error:
                logger.error(f"Failed to execute subprocess: {subprocess_error}")
                logger.error(f"Subprocess error type: {type(subprocess_error)}")
                raise

        except RuntimeError:
            # Re-raise already processed RuntimeError
            raise
        except Exception as e:
            logger.error(f"Error executing Gemini CLI command: {e}")
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Exception args: {e.args}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Error executing Gemini CLI command: {str(e)}") from e
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

        For Gemini CLI, we only use the last user message since it's stateless.
        System messages are included for context.

        Args:
            messages: List of chat messages

        Returns:
            Tuple of (formatted prompt text, list of temporary file paths)
        """
        prompt_parts = []
        temp_files = []

        # Find system messages and the last user message
        system_messages = []
        last_user_message = None

        for message in messages:
            if message.role == "system":
                system_messages.append(message)
            elif message.role == "user":
                last_user_message = message  # Keep updating to get the last one

        # Add system messages first
        for message in system_messages:
            if isinstance(message.content, str):
                prompt_parts.append(f"System: {message.content}")
            else:
                # Handle complex content for system messages
                content_parts = []
                for part in message.content:
                    if part.type == "text" and part.text:
                        content_parts.append(part.text)
                combined_content = " ".join(content_parts)
                prompt_parts.append(f"System: {combined_content}")

        # Add the last user message
        if last_user_message:
            if isinstance(last_user_message.content, str):
                prompt_parts.append(f"User: {last_user_message.content}")
            else:
                # List of content parts (vision support)
                content_parts = []

                for part in last_user_message.content:
                    if part.type == "text" and part.text:
                        content_parts.append(part.text)
                    elif part.type == "image_url" and part.image_url:
                        url = part.image_url.get("url", "")
                        if url.startswith("data:"):
                            # Process base64 image
                            temp_file_path = self._save_base64_image(url)
                            temp_files.append(temp_file_path)
                            content_parts.append(f"@{temp_file_path}")
                        else:
                            # For regular URLs, we'll just pass them through for now
                            # TODO: Download and save remote images if needed
                            content_parts.append(f"<image_url>{url}</image_url>")

                combined_content = " ".join(content_parts)
                prompt_parts.append(f"User: {combined_content}")

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
            temp_dir = ".gemini-cli-proxy"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create temporary file with simplified name
            filename = f"{uuid.uuid4().hex[:8]}{ext}"
            temp_file_path = os.path.join(temp_dir, filename)
            
            # Write image data
            with open(temp_file_path, 'wb') as f:
                f.write(image_data)
            
            return temp_file_path
            
        except Exception as e:
            logger.error(f"Error saving base64 image: {e}")
            raise ValueError(f"Failed to save base64 image: {e}") from e

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