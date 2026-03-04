from copy import deepcopy
from typing import Optional, Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, CustomContent, Stage, Attachment

from task.stage_util import StageProcessor

_IS_GPA = "is_gpa"
_GPA_MESSAGES = "gpa_messages"


class GPAGateway:

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def response(
            self,
            choice: Choice,
            stage: Stage,
            request: Request,
            additional_instructions: Optional[str]
    ) -> Message:
        client = AsyncDial(base_url=self.endpoint, api_version='2025-01-01-preview')
        headers = {"x-conversation-id": request.headers.get("x-conversation-id")} if request.headers else {}
        messages = self.__prepare_gpa_messages(request=request, additional_instructions=additional_instructions)

        stream = await client.chat.completions.create(
            model="general-purpose-agent",
            messages=messages,
            stream=True,
            extra_headers=headers
        )

        content = ""
        result_custom_content = CustomContent(attachments=[])
        stages_map: dict[int, Stage] = {}

        async for chunk in stream:
            delta = chunk.choices[0].delta
            print(delta)

            delta_content = getattr(delta, "content", None)
            if delta_content:
                content += delta_content
                stage.append_content(delta_content)

            delta_custom_content = getattr(delta, "custom_content", None)
            if not delta_custom_content:
                continue

            if getattr(delta_custom_content, "attachments", None):
                result_custom_content.attachments.extend(delta_custom_content.attachments)

            if getattr(delta_custom_content, "state", None):
                result_custom_content.state = delta_custom_content.state

            custom_content_dict = delta_custom_content.dict(exclude_none=True)
            if "stages" not in custom_content_dict:
                continue

            for stg in custom_content_dict["stages"]:
                idx = stg.get("index")
                if idx is None:
                    continue

                current_stage = stages_map.get(idx)
                if current_stage is None:
                    current_stage = StageProcessor.open_stage(choice, name=stg.get("name"))
                    stages_map[idx] = current_stage

                stg_content = stg.get("content")
                if stg_content:
                    current_stage.append_content(stg_content)

                for attachment in stg.get("attachments", []):
                    current_stage.append_attachment(Attachment(**attachment))

                if stg.get("status") == "completed":
                    StageProcessor.close_stage_safely(current_stage)

        for propagated_stage in stages_map.values():
            StageProcessor.close_stage_safely(propagated_stage)

        converted_attachments = [
            Attachment(**attachment.dict(exclude_none=True))
            for attachment in (result_custom_content.attachments or [])
        ]
        choice.append_custom_content(
            CustomContent(attachments=converted_attachments, state=result_custom_content.state)
        )

        self.__set_choice_state(
            choice=choice,
            state={
                _IS_GPA: True,
                _GPA_MESSAGES: result_custom_content.state
            }
        )

        return Message(role=Role.ASSISTANT, content=content)

    def __prepare_gpa_messages(self, request: Request, additional_instructions: Optional[str]) -> list[dict[str, Any]]:
        res_messages: list[dict[str, Any]] = []

        for idx in range(len(request.messages)):
            msg = request.messages[idx]
            if msg.role != Role.ASSISTANT:
                continue

            custom_content = getattr(msg, "custom_content", None)
            state = getattr(custom_content, "state", None)
            if not isinstance(state, dict) or not state.get(_IS_GPA):
                continue

            if idx - 1 >= 0:
                res_messages.append(request.messages[idx - 1].dict(exclude_none=True))

            copied_message = deepcopy(msg)
            copied_message.custom_content.state = state.get(_GPA_MESSAGES)
            res_messages.append(copied_message.dict(exclude_none=True))

        last_message = deepcopy(request.messages[-1]).dict(exclude_none=True)
        if additional_instructions:
            last_message["content"] = (
                f"{additional_instructions}\n\n"
                f"User request:\n{last_message.get('content', '')}"
            )

        res_messages.append(last_message)
        return res_messages

    @staticmethod
    def __set_choice_state(choice: Choice, state: dict) -> None:
        if hasattr(choice, "set_state"):
            choice.set_state(state)
            return
        if hasattr(choice, "append_custom_content"):
            choice.append_custom_content(CustomContent(state=state))
