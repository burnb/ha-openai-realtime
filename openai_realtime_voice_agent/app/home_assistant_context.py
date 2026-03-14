"""Helpers for fetching exposed Home Assistant voice-assistant context."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import websockets

logger = logging.getLogger(__name__)

CONVERSATION_ASSISTANT = "conversation"


@dataclass
class HomeAssistantEntitySummary:
    """Compact metadata for one exposed Home Assistant entity."""

    names: List[str]
    area_names: List[str]

    def to_instruction_lines(self) -> List[str]:
        """Render the entity as a compact instruction block."""
        lines = [f"- names: {', '.join(self.names)}"]
        if self.area_names:
            lines.append(f"  areas: {', '.join(self.area_names)}")
        return lines


@dataclass
class HomeAssistantContextSnapshot:
    """Compact Home Assistant metadata used to guide the LLM."""

    areas: List[str]
    entities: List[HomeAssistantEntitySummary]

    def to_instructions(self) -> str:
        """Render the snapshot as a compact system instruction suffix."""
        lines = [
            "Use exact Home Assistant entity and area names when calling tools. Do not invent entity names and area names.",
            "Home Assistant static context:",
        ]

        if self.areas:
            lines.append(f"Known areas: {', '.join(self.areas)}")
        else:
            lines.append("Known areas: none")

        if self.entities:
            lines.append("Exposed entities:")
            for entity in self.entities:
                lines.extend(entity.to_instruction_lines())
        else:
            lines.append("Exposed entities: none")

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
                exposed_entities = await self._send_command(
                    websocket,
                    {"type": "homeassistant/expose_entity/list"},
                )
                expose_new = await self._send_command(
                    websocket,
                    {
                        "type": "homeassistant/expose_new_entities/get",
                        "assistant": CONVERSATION_ASSISTANT,
                    },
                )

                included_entity_ids = self._select_candidate_entity_ids(
                    entities,
                    exposed_entities,
                    bool(expose_new.get("expose_new")),
                )
                entity_details = await self._send_command(
                    websocket,
                    {
                        "type": "config/entity_registry/get_entries",
                        "entity_ids": included_entity_ids,
                    },
                ) if included_entity_ids else {}

            snapshot = self._build_snapshot(
                areas,
                devices,
                entity_details,
                exposed_entities,
                bool(expose_new.get("expose_new")),
            )
            logger.info(
                "Loaded Home Assistant context: %s areas, %s exposed entities",
                len(snapshot.areas),
                len(snapshot.entities),
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
        entity_details: Dict[str, Optional[Dict[str, Any]]],
        exposed_entities: Dict[str, Any],
        expose_new: bool,
    ) -> HomeAssistantContextSnapshot:
        area_lookup = {
            area["area_id"]: area
            for area in areas
            if area.get("area_id") and area.get("name")
        }
        device_lookup = {
            device["id"]: device
            for device in devices
            if device.get("id")
        }
        explicitly_exposed_ids = self._get_explicitly_exposed_entity_ids(exposed_entities)

        summaries: List[HomeAssistantEntitySummary] = []
        seen_entity_ids = set()

        for entity_id in sorted(entity_details):
            entity = entity_details.get(entity_id)
            if not entity or not self._is_exposed_to_conversation(
                entity,
                explicitly_exposed_ids=explicitly_exposed_ids,
                expose_new=expose_new,
            ):
                continue

            summary = self._build_entity_summary(entity, area_lookup, device_lookup)
            if not summary:
                continue
            summaries.append(summary)
            seen_entity_ids.add(entity_id)

        for entity_id in sorted(explicitly_exposed_ids - seen_entity_ids):
            summaries.append(
                HomeAssistantEntitySummary(
                    names=[entity_id],
                    area_names=[],
                )
            )

        known_areas = self._dedupe(
            [area_name for summary in summaries for area_name in summary.area_names],
        )

        return HomeAssistantContextSnapshot(areas=known_areas, entities=summaries)

    def _build_entity_summary(
        self,
        entity: Dict[str, Any],
        area_lookup: Dict[str, Dict[str, Any]],
        device_lookup: Dict[str, Dict[str, Any]],
    ) -> Optional[HomeAssistantEntitySummary]:
        entity_id = entity.get("entity_id")
        if not entity_id:
            return None

        device = device_lookup.get(entity.get("device_id")) if entity.get("device_id") else None
        area = None
        area_id = entity.get("area_id")
        if area_id:
            area = area_lookup.get(area_id)
        elif device and device.get("area_id"):
            area = area_lookup.get(device.get("area_id"))

        names = self._dedupe(
            [
                self._clean_text(entity.get("name")),
                *self._clean_string_list(entity.get("aliases")),
            ],
        )
        area_names = self._dedupe(
            [
                self._clean_text(area.get("name")) if area else None,
                *self._clean_string_list(area.get("aliases") if area else None),
            ],
        )

        return HomeAssistantEntitySummary(
            names=names,
            area_names=area_names,
        )

    def _select_candidate_entity_ids(
        self,
        entities: Iterable[Dict[str, Any]],
        exposed_entities: Dict[str, Any],
        expose_new: bool,
    ) -> List[str]:
        explicit_ids = self._get_explicitly_exposed_entity_ids(exposed_entities)
        candidates = set(explicit_ids)

        if expose_new:
            for entity in entities:
                entity_id = entity.get("entity_id")
                if not entity_id or entity.get("hidden_by") or entity.get("disabled_by"):
                    continue
                candidates.add(entity_id)

        return sorted(candidates)

    def _get_explicitly_exposed_entity_ids(self, exposed_entities: Dict[str, Any]) -> set[str]:
        exposed = exposed_entities.get("exposed_entities", {}) if isinstance(exposed_entities, dict) else {}
        return {
            entity_id
            for entity_id, assistants in exposed.items()
            if isinstance(assistants, dict) and assistants.get(CONVERSATION_ASSISTANT)
        }

    def _is_exposed_to_conversation(
        self,
        entity: Dict[str, Any],
        explicitly_exposed_ids: set[str],
        expose_new: bool,
    ) -> bool:
        entity_id = entity.get("entity_id")
        if not entity_id or entity.get("hidden_by") or entity.get("disabled_by"):
            return False

        conversation_options = (entity.get("options") or {}).get(CONVERSATION_ASSISTANT) or {}
        should_expose = conversation_options.get("should_expose")
        if should_expose is True:
            return True
        if should_expose is False:
            return False
        if entity_id in explicitly_exposed_ids:
            return True
        return expose_new

    def _clean_string_list(self, values: Any) -> List[str]:
        if not values:
            return []
        return [cleaned for value in values if (cleaned := self._clean_text(value))]

    def _clean_text(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _dedupe(self, values: Iterable[Any]) -> List[str]:
        deduped: List[str] = []
        seen = set()
        normalized_values = sorted(
            normalized
            for value in values
            if (normalized := self._clean_text(value))
        )
        for value in normalized_values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
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