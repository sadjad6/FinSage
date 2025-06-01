"""
Model Context Protocol (MCP) Utilities for FinSage

This module provides the core functionality for creating, updating, and managing
context objects that are passed between agents in the FinSage system.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar, Generic, Type
from pydantic import BaseModel, Field, create_model

# Type variable for the context model
T = TypeVar('T', bound=BaseModel)

class ContextMetadata(BaseModel):
    """Metadata for context objects"""
    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context_type: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    creator_agent: str
    last_updated_by: str
    version: int = 1

class ContextWrapper(BaseModel, Generic[T]):
    """Generic wrapper for context objects with metadata"""
    metadata: ContextMetadata
    content: T

    @classmethod
    def create(
        cls, 
        context_type: str, 
        creator_agent: str, 
        content_model: Type[T], 
        content_data: Dict[str, Any]
    ) -> "ContextWrapper[T]":
        """Create a new context wrapper with initial content"""
        metadata = ContextMetadata(
            context_type=context_type,
            creator_agent=creator_agent,
            last_updated_by=creator_agent
        )
        
        content = content_model(**content_data)
        
        return cls(metadata=metadata, content=content)
    
    def update(
        self, 
        updated_by: str, 
        content_updates: Dict[str, Any]
    ) -> None:
        """Update the content and metadata of this context"""
        # Update the content with the provided updates
        for key, value in content_updates.items():
            if hasattr(self.content, key):
                setattr(self.content, key, value)
        
        # Update the metadata
        self.metadata.updated_at = datetime.now()
        self.metadata.last_updated_by = updated_by
        self.metadata.version += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the context wrapper to a dictionary"""
        return {
            "metadata": self.metadata.dict(),
            "content": self.content.dict()
        }
    
    def to_json(self) -> str:
        """Convert the context wrapper to a JSON string"""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(
        cls, 
        data: Dict[str, Any], 
        content_model: Type[T]
    ) -> "ContextWrapper[T]":
        """Create a context wrapper from a dictionary"""
        metadata = ContextMetadata(**data["metadata"])
        content = content_model(**data["content"])
        
        return cls(metadata=metadata, content=content)
    
    @classmethod
    def from_json(
        cls, 
        json_str: str, 
        content_model: Type[T]
    ) -> "ContextWrapper[T]":
        """Create a context wrapper from a JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data, content_model)


class ContextRegistry:
    """Registry for managing context objects"""
    def __init__(self):
        self.contexts: Dict[str, Dict[str, ContextWrapper]] = {}
    
    def register_context(self, context_wrapper: ContextWrapper) -> None:
        """Register a context object with the registry"""
        context_type = context_wrapper.metadata.context_type
        context_id = context_wrapper.metadata.context_id
        
        if context_type not in self.contexts:
            self.contexts[context_type] = {}
        
        self.contexts[context_type][context_id] = context_wrapper
    
    def get_context(self, context_type: str, context_id: str) -> Optional[ContextWrapper]:
        """Get a context object from the registry"""
        if context_type not in self.contexts:
            return None
        
        return self.contexts[context_type].get(context_id)
    
    def get_latest_context(self, context_type: str) -> Optional[ContextWrapper]:
        """Get the most recently updated context of a given type"""
        if context_type not in self.contexts or not self.contexts[context_type]:
            return None
        
        # Return the context with the most recent updated_at timestamp
        return max(
            self.contexts[context_type].values(),
            key=lambda ctx: ctx.metadata.updated_at
        )
    
    def list_contexts(self, context_type: Optional[str] = None) -> List[ContextWrapper]:
        """List all contexts or contexts of a specific type"""
        if context_type:
            if context_type not in self.contexts:
                return []
            return list(self.contexts[context_type].values())
        
        # Return all contexts from all types
        all_contexts = []
        for type_contexts in self.contexts.values():
            all_contexts.extend(type_contexts.values())
        
        return all_contexts


# Global registry instance
global_registry = ContextRegistry()

def get_registry() -> ContextRegistry:
    """Get the global context registry"""
    return global_registry
