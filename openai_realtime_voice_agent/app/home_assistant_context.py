"""Helpers for fetching Home Assistant area and entity context."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import websockets

logger = logging.getLogger(__name__)


@dataclass
class HomeAssistantContextSnapshot:
    """Compact Home Assistant metadata used to guide the LLM."""

    areas: List[str]
    entities_by_area: Dict[str, List[str]]
    unassigned_entities: List[str]

    def to_instructions(self) -> str:
        """Render the snapshot as a compact system instruction suffix."""
        lines = [
            "",
            "Home Assistant context:",
            "Use exact Home Assistant area names when calling tools. Do not invent area names.",
        ]

        if self.areas:
            lines.append(f"Known areas: {', '.join(self.areas)}")
        else:
            lines.append("Known areas: none")

        if self.entities_by_area:
            lines.append("Area hints:")
            for area_name in self.areas:
                area_entities = self.entities_by_area.get(area_name)
                if area_entities:
                    lines.append(f"- {area_name}: {', '.join(area_entities)}")

        if self.unassigned_entities:
            lines.append(f"Unassigned entity examples: {', '.join(self.unassigned_entities)}")

        lines.append(
            "If the user mentions an area that is not in the known areas list, ask for clarification instead of calling a tool with that area."
        )
        return "\n".join(lines)


class HomeAssistantContextService:
    """Fetches area and entity metadata from Home Assistant via WebSocket API."""

    def __init__(self, mcp_url: str, access_token: str):
        self._access_token = access_token
        self._request_id = 0
        self._websocket_url = self._build_websocket_url(mcp_url)

    async def fetch_snapshot(self) -> Optional[HomeAssistantContextSnapshot]:
        """Fetch a compact Home Assistant metadata snapshot."""
        try:
            async with websockets.connect(self._websocket_url, max_size=2**20) as websocket:
                await self._authenticate(websocket)

                areas = await self._send_command(websocket, {"type": "config/area_registry/list"})
                devices = await self._send_command(websocket, {"type": "config/device_registry/list"})
                entities = await self._send_command(websocket, {"type": "config/entity_registry/list"})

            snapshot = self._build_snapshot(areas, devices, entities)
            logger.info(
                "Loaded Home Assistant context: %s areas, %s area groups, %s unassigned entities",
                len(snapshot.areas),
                len(snapshot.entities_by_area),
                len(snapshot.unassigned_entities),
            )
            return snapshot
        except Exception as exc:
            logger.warning("⚠️ Failed to fetch Home Assistant context: %s", exc)
            return None

    async def _authenticate(self, websocket: websockets.ClientConnection) -> None:
        initial_message = json.loads(await websocket.recv())
        if initial_message.get("type") != "auth_required":
            raise RuntimeError(f"Unexpected Home Assistant auth prelude: {initial_message}")

        await websocket.send(
            json.dumps({"type": "auth", "access_token": self._access_token})
        )
        auth_response = json.loads(await websocket.recv())
        if auth_response.get("type") != "auth_ok":
            raise RuntimeError(f"Home Assistant auth failed: {auth_response}")

    async def _send_command(
        self, websocket: websockets.ClientConnection, payload: Dict[str, Any]
    ) -> Any:
        self._request_id += 1
        request_id = self._request_id

        message = {"id": request_id, **payload}
        await websocket.send(json.dumps(message))

        while True:
            response = json.loads(await websocket.recv())
            if response.get("id") != request_id:
                continue
            if not response.get("success", False):
                raise RuntimeError(response.get("error") or response)
            return response.get("result")

    def _build_snapshot(
        self,
        areas: Iterable[Dict[str, Any]],
        devices: Iterable[Dict[str, Any]],
        entities: Iterable[Dict[str, Any]],
    ) -> HomeAssistantContextSnapshot:
        area_id_to_name = {
            area["area_id"]: area["name"].strip()
            for area in areas
            if area.get("area_id") and area.get("name")
        }
        device_id_to_area_id = {
            device["id"]: device.get("area_id")
            for device in devices
            if device.get("id")
        }

        entities_by_area_id: Dict[str, List[str]] = defaultdict(list)
        unassigned_entities: List[str] = []

        for entity in entities:
            entity_id = entity.get("entity_id")
            if not entity_id or entity.get("hidden_by") or entity.get("disabled_by"):
                continue

            if not self._should_include_entity(entity_id):
                continue

            area_id = entity.get("area_id") or device_id_to_area_id.get(entity.get("device_id"))
            rendered_entity = self._render_entity(entity)
            if area_id and area_id in area_id_to_name:
                entities_by_area_id[area_id].append(rendered_entity)
            else:
                unassigned_entities.append(rendered_entity)

        area_names = sorted(area_id_to_name.values())[:12]
        entities_by_area = {
            area_id_to_name[area_id]: self._dedupe_and_limit(entities_by_area_id[area_id], limit=3)
            for area_id in sorted(entities_by_area_id, key=lambda item: area_id_to_name.get(item, item))
            if area_id in area_id_to_name and area_id_to_name[area_id] in area_names
        }

        return HomeAssistantContextSnapshot(
            areas=area_names,
            entities_by_area=entities_by_area,
            unassigned_entities=self._dedupe_and_limit(unassigned_entities, limit=4),
        )

    def _render_entity(self, entity: Dict[str, Any]) -> str:
        entity_id = entity["entity_id"]
        original_name = entity.get("original_name")
        if original_name:
            return f"{original_name} ({entity_id})"
        return entity_id

    def _should_include_entity(self, entity_id: str) -> bool:
        domain = entity_id.split(".", 1)[0]
        return domain in {
            "light",
            "switch",
            "climate",
            "cover",
            "fan",
            "media_player",
            "vacuum",
            "scene",
            "script",
            "lock",
            "input_boolean",
        }

    def _dedupe_and_limit(self, values: Iterable[str], limit: int) -> List[str]:
        deduped: List[str] = []
        seen = set()
        for value in sorted(values):
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
            if len(deduped) >= limit:
                break
        return deduped

    def _build_websocket_url(self, mcp_url: str) -> str:
        parsed = urlparse(mcp_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid Home Assistant MCP URL: {mcp_url}")

        websocket_scheme = "wss" if parsed.scheme == "https" else "ws"
        if parsed.path.endswith("/api/mcp"):
            websocket_path = f"{parsed.path[:-len('/api/mcp')]}/api/websocket"
        else:
            websocket_path = "/api/websocket"
        return f"{websocket_scheme}://{parsed.netloc}{websocket_path}"