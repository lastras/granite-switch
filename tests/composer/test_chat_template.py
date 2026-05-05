# SPDX-License-Identifier: Apache-2.0
"""Render-level tests for configure_chat_template() against the real Granite template.

Uses ``fixtures/granite_chat_template.jinja`` (copied from Granite 4.1, identical
to 4.0 at all injection anchor points) so that the injection code — regex search
for anchor patterns, ns namespace merge, Pass 1 / Pass 2 / fallback block
placement — is exercised against the real template rather than a hand-written
approximation.

``_decode_alora_invocation_text`` is patched in ``TestConfigureChatTemplate``;
those tests verify that the assembled template produces correct rendered output,
not adapter I/O.

``TestEndToEndAdapterConfigToRender`` exercises the full unpatched pipeline:
adapter_config.json → _decode_alora_invocation_text → configure_chat_template →
rendered output.  Uses minimal adapter fixtures in ``fixtures/``.

Code paths covered:
  - LoRA prefix path  (``ns.adapter_type == 'lora'``)
  - ALoRA Pass 1 + Pass 2  (invocation text found in last user message)
  - ALoRA fallback  (invocation text absent → ``ns.alora_target_idx == -1``)
  - No adapter  (``adapter_name`` undefined → no-op)
  - End-to-end: adapter_config.json → render (no patching)
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

from jinja2 import Environment

from granite_switch.composer.tokenizer_setup import configure_chat_template

_PATCH_TARGET = "granite_switch.composer.tokenizer_setup._decode_alora_invocation_text"

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
with open(os.path.join(_FIXTURES, "granite_chat_template.jinja")) as _f:
    _GRANITE_TEMPLATE = _f.read()


def _make_tokenizer():
    return SimpleNamespace(chat_template=_GRANITE_TEMPLATE)


def _render(tokenizer, **kwargs):
    return Environment().from_string(tokenizer.chat_template).render(**kwargs)


class TestConfigureChatTemplate:

    def test_lora_prefix_path(self):
        """LoRA: activation token emitted at the very start of the sequence."""
        tokenizer = _make_tokenizer()
        configure_chat_template(tokenizer, [("/path/a", "ctx_rel", "lora")])

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "Hello"}],
            add_generation_prompt=True,
            adapter_name="ctx_rel",
        )
        assert result.startswith("<|ctx_rel|>")

    def test_alora_pass1_pass2_path(self):
        """ALoRA Pass 1+2: token inserted in last user message before invocation text.

        Pass 1 finds the user message containing '<requirements>' and sets
        ns.alora_target_idx.  Pass 2 splits content.val on '<requirements>'
        and rejoins with the control token before the last occurrence.
        The fallback block does NOT fire (alora_target_idx >= 0).
        """
        with patch(_PATCH_TARGET, return_value="<requirements>"):
            tokenizer = _make_tokenizer()
            configure_chat_template(
                tokenizer, [("/path/a", "req_check", "alora")]
            )

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "<requirements>req1\nreq2"}],
            add_generation_prompt=True,
            adapter_name="req_check",
        )
        # Token immediately precedes the invocation text inside the user turn
        user_turn_header = "<|start_of_role|>user<|end_of_role|>"
        assert user_turn_header + "<|req_check|><requirements>" in result
        # Fallback did not fire: token is not immediately before generation prompt
        gen_prompt = "<|start_of_role|>assistant<|end_of_role|>"
        last_gen_pos = result.rindex(gen_prompt)
        assert result[last_gen_pos - len("<|req_check|>"):last_gen_pos] != "<|req_check|>"

    def test_alora_fallback_path(self):
        """ALoRA fallback: token emitted before generation prompt when invocation text is absent.

        Pass 1 scans all user messages and finds none containing the decoded invocation
        text (here the assistant role token sequence), so ns.alora_target_idx stays -1
        and the fallback block fires.
        """
        with patch(_PATCH_TARGET, return_value="<|start_of_role|>assistant<|end_of_role|>"):
            tokenizer = _make_tokenizer()
            configure_chat_template(
                tokenizer, [("/path/a", "answerability", "alora")]
            )

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "Hello"}],
            add_generation_prompt=True,
            adapter_name="answerability",
        )
        assert "<|answerability|>" in result
        # Token appears immediately before the generation prompt
        token = "<|answerability|>"
        gen_prompt = "<|start_of_role|>assistant<|end_of_role|>"
        token_pos = result.index(token)
        gen_pos = result.index(gen_prompt, token_pos)
        assert result[token_pos + len(token):gen_pos].strip() == ""

    def test_alora_pass1_pass2_iterable_content(self):
        """ALoRA Pass 1+2: token inserted correctly when message content is a list of parts.

        When content is iterable (multi-part), Pass 1 must record the *message* index
        (outer loop), not the entry index (inner loop).  A previous bug used the inner
        loop.index0, causing the wrong message to be targeted in Pass 2 and a
        subsequent crash on _parts[1] when rsplit found no separator.
        """
        with patch(_PATCH_TARGET, return_value="<requirements>"):
            tokenizer = _make_tokenizer()
            configure_chat_template(
                tokenizer, [("/path/a", "req_check", "alora")]
            )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check this: <requirements>req1\nreq2"},
                ],
            },
        ]
        result = _render(
            tokenizer,
            messages=messages,
            add_generation_prompt=True,
            adapter_name="req_check",
        )
        # Token must appear immediately before invocation text inside the user turn
        assert "<|req_check|><requirements>" in result
        assert result.index("<|req_check|>") > result.index("<|start_of_role|>user<|end_of_role|>")
        # Fallback must NOT also fire
        gen_prompt = "<|start_of_role|>assistant<|end_of_role|>"
        last_gen_pos = result.rindex(gen_prompt)
        assert result[last_gen_pos - len("<|req_check|>"):last_gen_pos] != "<|req_check|>"

    def test_no_adapter_no_tokens(self):
        """Without adapter_name the rendered output is identical to the original template."""
        messages = [{"role": "user", "content": "Hello"}]
        original = _render(_make_tokenizer(), messages=messages, add_generation_prompt=True)

        with patch(_PATCH_TARGET, return_value="<requirements>"):
            tokenizer = _make_tokenizer()
            configure_chat_template(
                tokenizer,
                [("/path/a", "ctx_rel", "lora"), ("/path/b", "req_check", "alora")],
            )
        modified = _render(tokenizer, messages=messages, add_generation_prompt=True)

        assert modified == original


class _FixtureTokenizer:
    """Tokenizer with a decode map for fixture adapter token IDs."""

    def __init__(self, chat_template, decode_map):
        self.chat_template = chat_template
        self._decode_map = decode_map

    def decode(self, token_ids, skip_special_tokens=False):
        return self._decode_map[tuple(token_ids)]


class TestEndToEndAdapterConfigToRender:
    """End-to-end: adapter_config.json → _decode_alora_invocation_text →
    configure_chat_template → rendered output.  No patching."""

    # Fixture adapter paths
    _ANSWERABILITY = os.path.join(_FIXTURES, "answerability_adapter")
    _CONTEXT_REL = os.path.join(_FIXTURES, "context_relevance_adapter")
    _SUMMARIZATION = os.path.join(_FIXTURES, "summarization_adapter")

    @staticmethod
    def _make_tokenizer(decode_map):
        return _FixtureTokenizer(_GRANITE_TEMPLATE, decode_map)

    def test_alora_fallback_from_adapter_config(self):
        """ALoRA adapter whose invocation tokens decode to the assistant role
        sequence → fallback path (token before generation prompt)."""
        tokenizer = self._make_tokenizer({
            # [100264, 78191, 100265] → assistant role sequence
            (100264, 78191, 100265): "<|start_of_role|>assistant<|end_of_role|>",
        })
        configure_chat_template(tokenizer, [
            (self._ANSWERABILITY, "answerability", "alora"),
        ])

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "Is this answerable?"}],
            add_generation_prompt=True,
            adapter_name="answerability",
        )
        # Fallback: token immediately before generation prompt
        token = "<|answerability|>"
        gen_prompt = "<|start_of_role|>assistant<|end_of_role|>"
        assert token in result
        token_pos = result.index(token)
        gen_pos = result.index(gen_prompt, token_pos)
        assert result[token_pos + len(token):gen_pos].strip() == ""

    def test_alora_invocation_at_start_of_user_message(self):
        """ALoRA: invocation text is the first thing in the user message."""
        tokenizer = self._make_tokenizer({(27,): "<context>"})
        configure_chat_template(tokenizer, [
            (self._CONTEXT_REL, "context_relevance", "alora"),
        ])

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "<context>some documents</context>"}],
            add_generation_prompt=True,
            adapter_name="context_relevance",
        )
        # Token injected right after the user role header, before <context>
        user_header = "<|start_of_role|>user<|end_of_role|>"
        assert user_header + "<|context_relevance|><context>" in result
        # Fallback must NOT fire
        gen_prompt = "<|start_of_role|>assistant<|end_of_role|>"
        last_gen_pos = result.rindex(gen_prompt)
        assert result[last_gen_pos - len("<|context_relevance|>"):last_gen_pos] != "<|context_relevance|>"

    def test_alora_invocation_mid_user_message(self):
        """ALoRA: invocation text appears in the middle of the user message."""
        tokenizer = self._make_tokenizer({(27,): "<context>"})
        configure_chat_template(tokenizer, [
            (self._CONTEXT_REL, "context_relevance", "alora"),
        ])

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "Please review: <context>docs</context>"}],
            add_generation_prompt=True,
            adapter_name="context_relevance",
        )
        # Token injected mid-message, before <context>
        assert "Please review: <|context_relevance|><context>" in result
        user_header = "<|start_of_role|>user<|end_of_role|>"
        assert result.index("<|context_relevance|>") > result.index(user_header)
        # Fallback must NOT fire
        gen_prompt = "<|start_of_role|>assistant<|end_of_role|>"
        last_gen_pos = result.rindex(gen_prompt)
        assert result[last_gen_pos - len("<|context_relevance|>"):last_gen_pos] != "<|context_relevance|>"

    def test_alora_multiple_occurrences_targets_last(self):
        """ALoRA: invocation text appears twice — token injected before the last occurrence.

        rsplit(..., 1) splits on the last occurrence, so the control token must
        land before the second <context>, not the first.
        """
        tokenizer = self._make_tokenizer({(27,): "<context>"})
        configure_chat_template(tokenizer, [
            (self._CONTEXT_REL, "context_relevance", "alora"),
        ])

        result = _render(
            tokenizer,
            messages=[{
                "role": "user",
                "content": "<context>first batch</context> Also check <context>second batch</context>",
            }],
            add_generation_prompt=True,
            adapter_name="context_relevance",
        )
        # The first <context> must NOT have the control token before it
        assert "<context>first batch</context> Also check <|context_relevance|><context>second batch" in result
        # Only one control token in the entire output
        assert result.count("<|context_relevance|>") == 1

    def test_lora_prefix_from_adapter_config(self):
        """LoRA adapter (no alora_invocation_tokens) → prefix path."""
        tokenizer = self._make_tokenizer({})  # no decode needed for LoRA
        configure_chat_template(tokenizer, [
            (self._SUMMARIZATION, "summarization", "lora"),
        ])

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "Summarize this."}],
            add_generation_prompt=True,
            adapter_name="summarization",
        )
        assert result.startswith("<|summarization|>")

    def test_mixed_adapters_from_adapter_config(self):
        """All three adapter types composed together, each activated independently."""
        tokenizer = self._make_tokenizer({
            (100264, 78191, 100265): "<|start_of_role|>assistant<|end_of_role|>",
            (27,): "<context>",
        })
        configure_chat_template(tokenizer, [
            (self._ANSWERABILITY, "answerability", "alora"),
            (self._CONTEXT_REL, "context_relevance", "alora"),
            (self._SUMMARIZATION, "summarization", "lora"),
        ])

        messages = [{"role": "user", "content": "<context>docs</context>"}]

        # Activate context_relevance → Pass 1+2
        result = _render(
            tokenizer, messages=messages,
            add_generation_prompt=True, adapter_name="context_relevance",
        )
        assert "<|context_relevance|><context>" in result

        # Activate answerability → fallback
        result = _render(
            tokenizer, messages=messages,
            add_generation_prompt=True, adapter_name="answerability",
        )
        token = "<|answerability|>"
        gen_prompt = "<|start_of_role|>assistant<|end_of_role|>"
        token_pos = result.index(token)
        gen_pos = result.index(gen_prompt, token_pos)
        assert result[token_pos + len(token):gen_pos].strip() == ""

        # Activate summarization → prefix
        result = _render(
            tokenizer, messages=messages,
            add_generation_prompt=True, adapter_name="summarization",
        )
        assert result.startswith("<|summarization|>")

        # No adapter → no tokens
        result_none = _render(
            tokenizer, messages=messages, add_generation_prompt=True,
        )
        assert "<|answerability|>" not in result_none
        assert "<|context_relevance|>" not in result_none
        assert "<|summarization|>" not in result_none
