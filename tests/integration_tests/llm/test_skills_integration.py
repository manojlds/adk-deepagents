"""Integration tests — skills across root, static, and dynamic delegation.

Run with: uv run pytest -m llm tests/integration_tests/llm/test_skills_integration.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adk_deepagents import create_deep_agent
from adk_deepagents.types import SubAgentSpec
from tests.integration_tests.conftest import make_litellm_model, run_agent_with_events

pytest.importorskip("adk_skills_agent")

pytestmark = [pytest.mark.integration, pytest.mark.llm]


def _write_skill(
    skills_root: Path,
    *,
    skill_name: str,
    description: str,
    body: str,
) -> None:
    skill_dir = skills_root / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_doc = f"---\nname: {skill_name}\ndescription: {description}\n---\n\n{body.strip()}\n"
    (skill_dir / "SKILL.md").write_text(skill_doc, encoding="utf-8")


@pytest.mark.timeout(180)
async def test_root_agent_skills_are_invoked(tmp_path: Path):
    """Root agent can discover and invoke a skill via use_skill."""
    model = make_litellm_model()
    token = "ROOT_SKILL_TOKEN_97341"
    skills_root = tmp_path / "skills"

    _write_skill(
        skills_root,
        skill_name="root-skill-token",
        description="Returns a hidden token for integration testing.",
        body=(
            "# Root Skill Token\n"
            f"When asked for the hidden token, return exactly: {token}\n"
            "Return only the token."
        ),
    )

    agent = create_deep_agent(
        model=model,
        name="skills_root_agent_test",
        skills=[str(skills_root)],
        instruction=(
            "You are a strict test agent. For hidden-token requests, you MUST call use_skill "
            "with skill 'root-skill-token'. After loading the skill, return only the token "
            "from the skill."
        ),
    )

    texts, function_calls, function_responses, _runner, _session = await run_agent_with_events(
        agent,
        "For this hidden-token request, call use_skill for root-skill-token and return the token "
        "exactly.",
    )

    response_text = " ".join(texts)
    assert "use_skill" in function_calls, f"Expected use_skill call, got: {function_calls}"
    assert "use_skill" in function_responses, (
        f"Expected use_skill response, got: {function_responses}"
    )
    assert token in response_text, f"Expected token in response, got: {response_text}"


@pytest.mark.timeout(180)
async def test_static_subagent_can_use_its_own_skills(tmp_path: Path):
    """Sub-agent skills work when delegation is static AgentTool-based."""
    model = make_litellm_model()
    token = "SUBAGENT_SKILL_TOKEN_64027"
    skills_root = tmp_path / "skills"

    _write_skill(
        skills_root,
        skill_name="subagent-skill-token",
        description="Returns a hidden token for sub-agent skill tests.",
        body=(
            "# Sub-Agent Skill Token\n"
            f"When asked for the hidden token, return exactly: {token}\n"
            "Return only the token."
        ),
    )

    researcher: SubAgentSpec = SubAgentSpec(
        name="researcher",
        description="Uses skill tools to retrieve hidden test tokens.",
        system_prompt=(
            "You are the researcher sub-agent. For hidden-token tasks, call use_skill with "
            "'subagent-skill-token' and return only the token from the skill."
        ),
        skills=[str(skills_root)],
    )

    agent = create_deep_agent(
        model=model,
        name="skills_static_subagent_test",
        instruction=(
            "Delegate hidden-token requests to the researcher sub-agent using the researcher "
            "tool. Do not answer directly."
        ),
        subagents=[researcher],
    )

    texts, function_calls, function_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Delegate this hidden-token task to researcher. Ask it to load subagent-skill-token and "
        "return the token.",
    )

    response_text = " ".join(texts)
    assert "researcher" in function_calls, f"Expected researcher call, got: {function_calls}"
    assert "researcher" in function_responses, (
        f"Expected researcher response, got: {function_responses}"
    )
    assert token in response_text, f"Expected token in response, got: {response_text}"


@pytest.mark.timeout(180)
async def test_dynamic_task_subagent_can_use_skills(tmp_path: Path):
    """Dynamic task delegation supports skills on the selected sub-agent profile."""
    model = make_litellm_model()
    token = "DYNAMIC_SUBAGENT_SKILL_TOKEN_31852"
    skills_root = tmp_path / "skills"

    _write_skill(
        skills_root,
        skill_name="dynamic-subagent-skill",
        description="Returns a hidden token for dynamic sub-agent tests.",
        body=(
            "# Dynamic Sub-Agent Skill\n"
            f"When asked for the hidden token, return exactly: {token}\n"
            "Return only the token."
        ),
    )

    skill_worker: SubAgentSpec = SubAgentSpec(
        name="skill_worker",
        description="Dynamic worker that uses a dedicated skill.",
        system_prompt=(
            "For hidden-token tasks, call use_skill with 'dynamic-subagent-skill' and return "
            "only the token from the skill."
        ),
        skills=[str(skills_root)],
    )

    agent = create_deep_agent(
        model=model,
        name="skills_dynamic_subagent_test",
        instruction=(
            "You must delegate hidden-token requests through the task tool using "
            "subagent_type='skill_worker'. Return the delegated result unchanged."
        ),
        subagents=[skill_worker],
        delegation_mode="dynamic",
    )

    texts, function_calls, function_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Use task with subagent_type skill_worker. In the delegated prompt, ask the worker to "
        "load dynamic-subagent-skill and return the hidden token only.",
    )

    response_text = " ".join(texts)
    assert "task" in function_calls, f"Expected task call, got: {function_calls}"
    assert "task" in function_responses, f"Expected task response, got: {function_responses}"
    assert token in response_text, f"Expected token in response, got: {response_text}"


@pytest.mark.timeout(180)
async def test_skill_instructions_can_trigger_dynamic_task_delegation(tmp_path: Path):
    """A root skill can instruct the parent agent to spin up dynamic task delegation."""
    model = make_litellm_model()
    skills_root = tmp_path / "skills"

    _write_skill(
        skills_root,
        skill_name="dynamic-delegation-playbook",
        description="Playbook that requires task-tool delegation.",
        body=(
            "# Dynamic Delegation Playbook\n"
            "When handling a playbook request, call the task tool with "
            "subagent_type='math_expert'.\n"
            "Use delegated prompt: 'Compute 21 + 21. Return only the number.'\n"
            "Return only the delegated numeric result."
        ),
    )

    math_subagent: SubAgentSpec = SubAgentSpec(
        name="math_expert",
        description="Solves arithmetic and returns concise numeric results.",
        system_prompt="You are a math expert. Return the final numeric answer only.",
    )

    agent = create_deep_agent(
        model=model,
        name="skills_dynamic_chain_test",
        instruction=(
            "For playbook requests, first call use_skill with 'dynamic-delegation-playbook', "
            "then follow that skill exactly. Never solve arithmetic directly in the parent."
        ),
        skills=[str(skills_root)],
        subagents=[math_subagent],
        delegation_mode="dynamic",
    )

    texts, function_calls, function_responses, _runner, _session = await run_agent_with_events(
        agent,
        "Handle this playbook request: use dynamic-delegation-playbook and return the final "
        "result only.",
    )

    response_text = " ".join(texts)
    assert "use_skill" in function_calls, f"Expected use_skill call, got: {function_calls}"
    assert "use_skill" in function_responses, (
        f"Expected use_skill response, got: {function_responses}"
    )
    assert "task" in function_calls, f"Expected task call, got: {function_calls}"
    assert "task" in function_responses, f"Expected task response, got: {function_responses}"
    assert "42" in response_text, f"Expected delegated result 42, got: {response_text}"
