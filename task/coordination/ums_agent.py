import ast
import json
from typing import Optional

import httpx
from aidial_sdk.chat_completion import Request, Message, Stage, Choice, Role, CustomContent


_UMS_CONVERSATION_ID = "ums_conversation_id"


class UMSAgentGateway:

    def __init__(self, ums_agent_endpoint: str):
        self.ums_agent_endpoint = ums_agent_endpoint

    async def response(
            self,
            choice: Choice,
            stage: Stage,
            request: Request,
            additional_instructions: Optional[str]
    ) -> Message:
        conversation_id = self.__get_ums_conversation_id(request)
        if conversation_id is None:
            conversation_id = await self.__create_ums_conversation()

        self.__set_choice_state(choice=choice, state={_UMS_CONVERSATION_ID: conversation_id})

        last_message = request.messages[-1] if request.messages else Message(role=Role.USER, content="")
        user_message = last_message.content or ""
        if additional_instructions:
            user_message = f"{additional_instructions}\n\nUser request:\n{user_message}"

        content = await self.__call_ums_agent(
            conversation_id=conversation_id,
            user_message=user_message,
            stage=stage
        )
        return Message(role=Role.ASSISTANT, content=content)

    def __get_ums_conversation_id(self, request: Request) -> Optional[str]:
        """Extract UMS conversation ID from previous messages if it exists"""
        for msg in reversed(request.messages):
            custom_content = getattr(msg, "custom_content", None)
            state = getattr(custom_content, "state", None)
            if isinstance(state, dict) and _UMS_CONVERSATION_ID in state:
                return state[_UMS_CONVERSATION_ID]
        return None

    async def __create_ums_conversation(self) -> str:
        """Create a new conversation on UMS agent side"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{self.ums_agent_endpoint}/conversations")
            response.raise_for_status()
            data = response.json()
            return data["id"]

    async def __call_ums_agent(
            self,
            conversation_id: str,
            user_message: str,
            stage: Stage
    ) -> str:
        """Call UMS agent and stream the response"""
        accumulated_content = ""
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                    "POST",
                    f"{self.ums_agent_endpoint}/conversations/{conversation_id}/chat",
                    json={
                        "message": {"role": "user", "content": user_message},
                        "stream": True
                    }
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    raw = line[6:].strip() if line.startswith("data: ") else line.strip()
                    if raw == "[DONE]":
                        break

                    payload = self.__parse_chunk(raw)
                    if payload is None:
                        continue

                    choices = payload.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    chunk_content = delta.get("content")
                    if chunk_content:
                        accumulated_content += chunk_content
                        stage.append_content(chunk_content)

        return accumulated_content

    @staticmethod
    def __parse_chunk(raw: str) -> Optional[dict]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                return None
            return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def __set_choice_state(choice: Choice, state: dict) -> None:
        if hasattr(choice, "set_state"):
            choice.set_state(state)
            return
        if hasattr(choice, "append_custom_content"):
            choice.append_custom_content(CustomContent(state=state))
