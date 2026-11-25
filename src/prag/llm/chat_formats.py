"""Chat formats for target LLMs.

FastChat `vicuna_v1.1` — copied verbatim from lm-sys/FastChat v0.2.36 (latest tag),
fastchat/conversation.py L473-483 and the ADD_COLON_TWO branch of Conversation.get_prompt (L79-90):

    register_conv_template(
        Conversation(
            name="vicuna_v1.1",
            system_message="A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.",
            roles=("USER", "ASSISTANT"),
            sep_style=SeparatorStyle.ADD_COLON_TWO,
            sep=" ",
            sep2="</s>",
        )
    )

        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ...
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret

(system_template defaults to "{system_message}".)

Resolution path in upstream: Vicuna.query -> fastchat.model.get_conversation_template("lmsys/vicuna-7b-v1.3")
-> get_model_adapter matches VicunaAdapter ("vicuna" in basename; PeftModelAdapter / StableVicunaAdapter don't match)
-> get_default_conv_template: "v0" not in "vicuna-7b-v1.3" -> get_conv_template("vicuna_v1.1").
VicunaAdapter also sets use_fast_tokenizer = False, and load_model on cuda uses torch_dtype=float16.

Upstream builds a single-turn conversation: append_message(USER, msg); append_message(ASSISTANT, None).
"""
from __future__ import annotations

VICUNA_V1_1_SYSTEM_MESSAGE = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
VICUNA_V1_1_ROLES = ("USER", "ASSISTANT")
VICUNA_V1_1_SEP = " "
VICUNA_V1_1_SEP2 = "</s>"


def fastchat_vicuna_v1_1(messages: list[tuple[str, str | None]]) -> str:
    """Verbatim port of Conversation.get_prompt() for sep_style=ADD_COLON_TWO (text-only messages)."""
    seps = [VICUNA_V1_1_SEP, VICUNA_V1_1_SEP2]
    ret = VICUNA_V1_1_SYSTEM_MESSAGE + seps[0]
    for i, (role, message) in enumerate(messages):
        if message:
            ret += role + ": " + message + seps[i % 2]
        else:
            ret += role + ":"
    return ret


def vicuna_single_turn(msg: str) -> str:
    """Exactly what upstream Vicuna.query feeds to the tokenizer."""
    return fastchat_vicuna_v1_1([(VICUNA_V1_1_ROLES[0], msg), (VICUNA_V1_1_ROLES[1], None)])


CHAT_FORMATS = {"fastchat_vicuna_v1_1": vicuna_single_turn}
