"""
OpenAI adapter module

Handles format conversion and compatibility
"""

import time
import uuid
import logging
from typing import AsyncGenerator
from fastapi.responses import StreamingResponse

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ChatMessage
)
from .gemini_client import gemini_client

logger = logging.getLogger('gemini_cli_proxy')


class OpenAIAdapter:
    """OpenAI format adapter"""
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Handle chat completion request (non-streaming)
        
        Args:
            request: OpenAI format chat completion request
            
        Returns:
            OpenAI format chat completion response
        """
        logger.info(f"Processing chat completion request, model: {request.model}, messages: {len(request.messages)}")
        
        try:
            # Call Gemini CLI
            response_text = await gemini_client.chat_completion(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Build OpenAI format response
            response = ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=response_text
                        ),
                        finish_reason="stop"
                    )
                ]
            )
            
            logger.info(f"Chat completion request processed successfully, response length: {len(response_text)}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing chat completion request: {e}")
            raise
    
    async def chat_completion_stream(self, request: ChatCompletionRequest) -> StreamingResponse:
        """
        Handle streaming chat completion request
        
        Args:
            request: OpenAI format chat completion request
            
        Returns:
            Streaming response
        """
        logger.info(f"Processing streaming chat completion request, model: {request.model}, messages: {len(request.messages)}")
        
        async def generate_stream():
            """Generate streaming response data"""
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created_time = int(time.time())
            
            try:
                # Get streaming data generator
                stream_generator = gemini_client.chat_completion_stream(
                    messages=request.messages,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
                
                # Send data chunks one by one
                async for chunk in stream_generator:
                    stream_response = ChatCompletionStreamResponse(
                        id=completion_id,
                        created=created_time,
                        model=request.model,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0,
                                delta={"content": chunk},
                                finish_reason=None
                            )
                        ]
                    )
                    
                    # Send data chunk
                    yield f"data: {stream_response.model_dump_json()}\n\n"
                
                # Send end marker
                final_response = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created_time,
                    model=request.model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta={},
                            finish_reason="stop"
                        )
                    ]
                )
                yield f"data: {final_response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                
                logger.info("Streaming chat completion request processed successfully")
                
            except Exception as e:
                logger.error(f"Error processing streaming chat completion request: {e}")
                # Send error information
                error_response = {
                    "error": {
                        "message": str(e),
                        "type": "internal_error"
                    }
                }
                yield f"data: {error_response}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )


# Global adapter instance
openai_adapter = OpenAIAdapter() 