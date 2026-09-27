import asyncio
import json

import numpy as np
import pytest

from flux_action.serving import protocol
from flux_action.serving.gpt6_vla import (
    AstraPlanner,
    JsonlLogger,
    PlannerResult,
    _proxy_connection,
    compose_vla_prompt,
    parse_planner_response,
)


def response(arguments, *, name="select_vla_subtask"):
    return {
        "id": "resp_test",
        "output": [{"type": "function_call", "name": name, "arguments": json.dumps(arguments)}],
    }


def observation():
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    return {
        "observation/wrist_image_left": image,
        "observation/exterior_image_1_left": image,
        "observation/exterior_image_2_left": image,
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
    }


def test_parse_planner_response_and_compose_prompt():
    plan = parse_planner_response(
        response(
            {
                "subtask": "Move above the yellow cup",
                "phase": "approach",
                "scene_summary": "The cup is left of the bowl.",
                "success_criteria": "The gripper is centered above the cup.",
            }
        )
    )
    assert plan.response_id == "resp_test"
    assert compose_vla_prompt("Put the cup in the bowl", plan) == (
        "Put the cup in the bowl. Current immediate step: Move above the yellow cup"
    )


def test_planner_builds_forced_structured_multiview_request(monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", "not-a-real-key")
    captured = {}

    def transport(url, payload, headers, timeout):
        captured.update(
            url=url,
            body=json.loads(payload),
            authorization=headers["Authorization"],
            timeout=timeout,
        )
        return response(
            {
                "subtask": "Center the gripper over the object",
                "phase": "approach",
                "scene_summary": "The object is visible.",
                "success_criteria": "The gripper is centered.",
            }
        )

    planner = AstraPlanner(api_key_env="TEST_OPENAI_KEY", transport=transport, max_retries=0)
    plan = planner.plan(observation(), "Pick up the object")
    assert plan.subtask == "Center the gripper over the object"
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["authorization"] == "Bearer not-a-real-key"
    assert captured["body"]["model"] == "gpt-6-astra"
    assert captured["body"]["store"] is False
    assert captured["body"]["parallel_tool_calls"] is False
    assert captured["body"]["tool_choice"]["name"] == "select_vla_subtask"
    content = captured["body"]["input"][0]["content"]
    assert sum(item["type"] == "input_image" for item in content) == 3
    assert all(
        item["image_url"].startswith("data:image/jpeg;base64,")
        for item in content
        if item["type"] == "input_image"
    )


@pytest.mark.parametrize("phase", ["done", None])
def test_planner_rejects_invalid_phase(phase):
    with pytest.raises(ValueError, match="phase"):
        parse_planner_response(
            response(
                {
                    "subtask": "Move",
                    "phase": phase,
                    "scene_summary": "Visible",
                    "success_criteria": "Moved",
                }
            )
        )


def test_planner_requires_exactly_one_tool_call():
    with pytest.raises(ValueError, match="exactly one"):
        parse_planner_response({"output": []})


def test_prompt_rejects_empty_overall_task():
    plan = PlannerResult("Move", "approach", "Visible", "Moved")
    with pytest.raises(ValueError, match="overall_task"):
        compose_vla_prompt("", plan)


def test_websocket_pipeline_calls_planner_before_vla():
    import websockets.asyncio.client
    import websockets.asyncio.server

    events = []

    class RecordingPlanner:
        model = "test-planner"

        def plan(self, observation, overall_task, previous_plan=None):
            del observation, previous_plan
            events.append("planner")
            assert overall_task == "Put the cup in the bowl"
            return PlannerResult(
                "Move above the cup", "approach", "Cup visible", "Gripper above cup"
            )

    async def fake_vla(websocket):
        packer = protocol.Packer()
        await websocket.send(packer.pack({"model": "fake-vla"}))
        request = protocol.unpackb(await websocket.recv())
        events.append("vla")
        assert request["prompt"].endswith("Current immediate step: Move above the cup")
        await websocket.send(
            packer.pack(
                {
                    "action": np.zeros((32, 8), dtype=np.float32),
                    "server_timing": {"infer_ms": 1.0},
                }
            )
        )

    async def run():
        packer = protocol.Packer()
        async with websockets.asyncio.server.serve(fake_vla, "127.0.0.1", 0) as vla_server:
            vla_port = vla_server.sockets[0].getsockname()[1]

            async def proxy_handler(websocket):
                await _proxy_connection(
                    websocket,
                    upstream=f"ws://127.0.0.1:{vla_port}",
                    planner=RecordingPlanner(),
                    logger=JsonlLogger(None),
                )

            async with websockets.asyncio.server.serve(
                proxy_handler, "127.0.0.1", 0
            ) as proxy_server:
                proxy_port = proxy_server.sockets[0].getsockname()[1]
                async with websockets.asyncio.client.connect(
                    f"ws://127.0.0.1:{proxy_port}", compression=None, max_size=None
                ) as client:
                    metadata = protocol.unpackb(await client.recv())
                    assert metadata["serial"] is True
                    request = observation()
                    request["prompt"] = "Put the cup in the bowl"
                    await client.send(packer.pack(request))
                    result = protocol.unpackb(await client.recv())
                    assert np.asarray(result["action"]).shape == (32, 8)
                    assert result["gpt6_plan"]["subtask"] == "Move above the cup"
                    assert result["pipeline_timing"]["total_ms"] >= 0

    asyncio.run(run())
    assert events == ["planner", "vla"]
