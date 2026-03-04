import json
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, Stage

from task.coordination.gpa import GPAGateway
from task.coordination.ums_agent import UMSAgentGateway
from task.logging_config import get_logger
from task.models import CoordinationRequest, AgentName
from task.prompts import COORDINATION_REQUEST_SYSTEM_PROMPT, FINAL_RESPONSE_SYSTEM_PROMPT
from task.stage_util import StageProcessor

logger = get_logger(__name__)


class MASCoordinator:

    def __init__(self, endpoint: str, deployment_name: str, ums_agent_endpoint: str):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.ums_agent_endpoint = ums_agent_endpoint

    async def handle_request(self, choice: Choice, request: Request) -> Message:
        client = AsyncDial(base_url=self.endpoint, api_version='2025-01-01-preview')

        coordination_stage = StageProcessor.open_stage(choice, name="Coordination Request")
        try:
            coordination_request = await self.__prepare_coordination_request(client=client, request=request)
            coordination_stage.append_content(json.dumps(coordination_request.model_dump(exclude_none=True), indent=2))
        finally:
            StageProcessor.close_stage_safely(coordination_stage)

        agent_stage = StageProcessor.open_stage(choice, name=f"{coordination_request.agent_name.value} Agent")
        try:
            agent_message = await self.__handle_coordination_request(
                coordination_request=coordination_request,
                choice=choice,
                stage=agent_stage,
                request=request
            )
        finally:
            StageProcessor.close_stage_safely(agent_stage)

        return await self.__final_response(
            client=client,
            choice=choice,
            request=request,
            agent_message=agent_message
        )

    async def __prepare_coordination_request(self, client: AsyncDial, request: Request) -> CoordinationRequest:
        messages = self.__prepare_messages(request=request, system_prompt=COORDINATION_REQUEST_SYSTEM_PROMPT)
        response = await client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": CoordinationRequest.model_json_schema()
                    }
                }
            }
        )

        content = response.choices[0].message.content or "{}"
        if not isinstance(content, str):
            content = str(content)

        parsed = json.loads(content)
        return CoordinationRequest.model_validate(parsed)

    def __prepare_messages(self, request: Request, system_prompt: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        for msg in request.messages:
            if msg.role == Role.USER and getattr(msg, "custom_content", None):
                messages.append({"role": "user", "content": msg.content})
            else:
                messages.append(msg.dict(exclude_none=True))

        return messages

    async def __handle_coordination_request(
            self,
            coordination_request: CoordinationRequest,
            choice: Choice,
            stage: Stage,
            request: Request
    ) -> Message:
        if coordination_request.agent_name == AgentName.UMS:
            gateway = UMSAgentGateway(ums_agent_endpoint=self.ums_agent_endpoint)
            return await gateway.response(
                choice=choice,
                stage=stage,
                request=request,
                additional_instructions=coordination_request.additional_instructions
            )

        gateway = GPAGateway(endpoint=self.endpoint)
        return await gateway.response(
            choice=choice,
            stage=stage,
            request=request,
            additional_instructions=coordination_request.additional_instructions
        )

    async def __final_response(
            self, client: AsyncDial,
            choice: Choice,
            request: Request,
            agent_message: Message
    ) -> Message:
        messages = self.__prepare_messages(request=request, system_prompt=FINAL_RESPONSE_SYSTEM_PROMPT)

        user_content = request.messages[-1].content if request.messages else ""
        context = agent_message.content or ""
        augmented_prompt = (
            "Use the provided context to answer the user request.\n\n"
            f"Context from called agent:\n{context}\n\n"
            f"User request:\n{user_content}"
        )

        if messages:
            messages[-1]["content"] = augmented_prompt

        stream = await client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            stream=True
        )

        content = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta
            delta_content = getattr(delta, "content", None)
            if delta_content:
                content += delta_content
                choice.append_content(delta_content)

        return Message(role=Role.ASSISTANT, content=content)
