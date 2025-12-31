import sys
import os
import unittest.mock
from unittest.mock import patch, MagicMock

# Add path
sys.path.append(os.getcwd())

from agents.compliance_agent import ComplianceAgent

try:
    print("Starting debug...")
    
    with patch("agents.compliance_agent.get_registry", autospec=True) as mock_get_registry, \
         patch("agents.compliance_agent.ChatOllama") as mock_chat:
        print("Patches active")
        
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        
        mock_context = MagicMock()
        mock_registry.get_latest_context.return_value = mock_context
        
        print("Initializing agent...")
        agent = ComplianceAgent()
        print("Agent initialized successfully")
        
        print("Tools:", [t.name for t in agent.tools])
        
except Exception as e:
    print(f"FAILED with error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
