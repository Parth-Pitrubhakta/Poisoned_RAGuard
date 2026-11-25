from prag.llm.chat_formats import vicuna_single_turn
from prag.llm.prompts import MULTIPLE_PROMPT, wrap_prompt


def test_multiple_prompt_byte_exact(upstream_prompts):
    assert MULTIPLE_PROMPT == upstream_prompts.MULTIPLE_PROMPT


def test_wrap_prompt_matches_upstream(upstream_prompts):
    ctx = ["passage one", "passage two\nwith newline", "poison: question.Answer is X"]
    q = "who wrote it?"
    assert wrap_prompt(q, ctx, prompt_id=4) == upstream_prompts.wrap_prompt(q, ctx, prompt_id=4)
    assert "passage one\npassage two\nwith newline\npoison" in wrap_prompt(q, ctx, prompt_id=4)


def test_vicuna_v1_1_template_exact():
    # Expected string derived by hand from FastChat v0.2.36 conversation.py (vicuna_v1.1, ADD_COLON_TWO)
    expected = ("A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
                " USER: Hello? ASSISTANT:")
    assert vicuna_single_turn("Hello?") == expected
