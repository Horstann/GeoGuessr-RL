import base64
from typing import Dict, List
from openai import OpenAI
import httpx


class UnifiedAIClient:
    """
    Unified AI client that can handle both open source (AK/SK-based) and closed source (API key-based) models.
    Provides OpenAI-compatible interface for both types.
    """
    
    def __init__(self, ai_config, ai_keys):
        """
        Initialize the unified AI client.
        
        Args:
            ai_config: Configuration object with model parameters
            ai_keys: Keys object with API credentials
        """
        self.config = ai_config
        self.keys = ai_keys
        
        # Determine model type and initialize appropriate client
        self.model_type = getattr(ai_config, 'model_type', 'closed_source')
        
        if self.model_type == "closed_source":
            self._init_closed_source_client()
        else:
            self._init_open_source_client()
    
    
    def _init_closed_source_client(self):
        """Initialize OpenAI client for closed source models."""
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.keys.api_key
        )
        self.is_async = False
    
    def _init_open_source_client(self):
        """Initialize OpenAI client for open source models with AK/SK Basic auth."""
        # Get server_keys from config
        server_keys = getattr(self.keys, 'server_keys', {}) or {}
        ak = (
            server_keys.get("ak")
            if isinstance(server_keys, dict)
            else getattr(server_keys, "ak", None)
        )
        sk = (
            server_keys.get("sk")
            if isinstance(server_keys, dict)
            else getattr(server_keys, "sk", None)
        )

        if not ak or not sk:
            raise ValueError(
                "ai_keys.server_keys.ak and ai_keys.server_keys.sk are required for open_source models"
            )

        # Create Basic auth token
        auth_token = base64.b64encode(f"{ak}:{sk}".encode()).decode()
        http_client = httpx.Client(trust_env=False)

        # Ensure base_url ends with /
        base_url = self.config.base_url.rstrip("/") + "/"

        # Use OpenAI client with Basic auth header
        self.client = OpenAI(
            base_url=base_url,
            api_key="unused",  # required by SDK; Basic header is used instead
            default_headers={"Authorization": f"Basic {auth_token}"},
            http_client=http_client,
        )
        self.is_async = False


class ChatCompletions:
    """Mock chat completions object that provides OpenAI-compatible interface."""
    
    def __init__(self, client: UnifiedAIClient):
        self.client = client
    
    def create(self, model: str = None, messages: List[Dict] = None, **kwargs):
        """
        Create chat completion. Uses OpenAI client for both model types.
        """
        # Use OpenAI client directly for both closed_source and open_source
        data = self.client.client.chat.completions.create(
            model=model or self.client.config.model,
            messages=messages,
            **kwargs
        )
        return data


class Chat:
    """Mock chat object that provides OpenAI-compatible interface."""
    
    def __init__(self, client: UnifiedAIClient):
        self.completions = ChatCompletions(client)


def get_openai_client(ai_config, ai_keys):
    """
    Get OpenAI-compatible client that can be used as a drop-in replacement for OpenAI().
    
    Usage:
        from ai_client import get_openai_client
        client = get_openai_client(ai_config=cfg.ai_config, ai_keys=cfg.ai_keys)
        
        # Use exactly like OpenAI client
        response = client.chat.completions.create(
            model="model-name",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """
    client = UnifiedAIClient(ai_config, ai_keys)
    # Add OpenAI-compatible chat interface
    client.chat = Chat(client)
    return client
