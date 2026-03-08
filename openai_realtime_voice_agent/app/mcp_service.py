"""MCP service integration using Pipecat's MCPClient with StreamableHTTP."""
import logging
from typing import List, Optional
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.mcp_service import MCPClient, StreamableHttpParameters

logger = logging.getLogger(__name__)


class HomeAssistantMCPService:
    """Home Assistant MCP service using Pipecat's MCPClient."""
    
    def __init__(self, url: str, access_token: str, excluded_tools: Optional[List[str]] = None):
        """
        Initialize Home Assistant MCP service.
        
        Args:
            url: Home Assistant MCP Server URL (e.g., http://supervisor/core/api/mcp)
            access_token: Long-lived access token for Home Assistant
            excluded_tools: Optional list of MCP tool names to exclude
        """
        self.url = url
        self.access_token = access_token
        self.excluded_tools = set(excluded_tools or [])
        self.mcp_client: Optional[MCPClient] = None
        
    async def initialize(self) -> MCPClient:
        """Initialize and return the MCP client."""
        try:
            logger.info(f"🔗 Initializing Home Assistant MCP Client at {self.url}")
            
            # Create StreamableHTTP parameters with authentication
            server_params = StreamableHttpParameters(
                url=self.url,
                headers={
                    "Authorization": f"Bearer {self.access_token}"
                }
            )
            
            # Create MCP client
            self.mcp_client = MCPClient(server_params=server_params)
            
            logger.info("✅ Home Assistant MCP Client initialized")
            return self.mcp_client
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Home Assistant MCP Client: {e}", exc_info=True)
            raise
    
    def get_client(self) -> Optional[MCPClient]:
        """Get the MCP client instance."""
        return self.mcp_client

    def filter_tools_schema(self, tools_schema: ToolsSchema) -> ToolsSchema:
        """Remove excluded tools from the fetched MCP schema."""
        if not self.excluded_tools:
            return tools_schema

        filtered_tools = [
            tool for tool in tools_schema.standard_tools if tool.name not in self.excluded_tools
        ]
        excluded_count = len(tools_schema.standard_tools) - len(filtered_tools)
        if excluded_count > 0:
            logger.info(
                "Excluded %s MCP tools via HA_MCP_TOOL_FILTER: %s",
                excluded_count,
                sorted(self.excluded_tools),
            )
        return ToolsSchema(standard_tools=filtered_tools)






