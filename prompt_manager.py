from typing import Dict, List
import json
from datetime import datetime
import os

class PromptVersion:
    def __init__(self, content: str, version: str, created_at: str, description: str = ""):
        self.content = content
        self.version = version
        self.created_at = created_at
        self.description = description

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "version": self.version,
            "created_at": self.created_at,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PromptVersion':
        return cls(
            content=data["content"],
            version=data["version"],
            created_at=data["created_at"],
            description=data.get("description", "")
        )

class PromptManager:
    def __init__(self):
        self.prompts_file = "prompts.json"
        self.prompts = self._load_prompts()
        self._ensure_default_prompts()

    def _load_prompts(self) -> Dict:
        """Load prompts from JSON file or return empty dict if file doesn't exist."""
        if os.path.exists(self.prompts_file):
            try:
                with open(self.prompts_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_prompts(self):
        """Save prompts to JSON file."""
        with open(self.prompts_file, 'w') as f:
            json.dump(self.prompts, f, indent=2)

    def _ensure_default_prompts(self):
        """Ensure default prompts exist for each model."""
        default_prompts = {
            "gemini": {
                "versions": [{
                    "content": "You are an advanced AI assistant powered by Google's Gemini model. You excel at providing detailed, accurate, and helpful responses. Please analyze queries thoroughly and provide comprehensive answers.",
                    "version": "v1",
                    "created_at": datetime.now().isoformat(),
                    "description": "Default Gemini prompt"
                }]
            },
            "gemma": {
                "versions": [{
                    "content": "You are an AI assistant powered by Google's Gemma model. You are focused on providing clear, concise, and accurate responses. Please maintain a helpful and informative tone while being direct and efficient.",
                    "version": "v1",
                    "created_at": datetime.now().isoformat(),
                    "description": "Default Gemma prompt"
                }]
            }
        }

        # Add default prompts only if the model doesn't exist or has no versions
        for model, data in default_prompts.items():
            if model not in self.prompts or not self.prompts[model].get("versions", []):
                self.prompts[model] = data
                self._save_prompts()

    def add_prompt_version(self, model: str, content: str, description: str = "") -> None:
        """Add a new prompt version for a model."""
        if model not in self.prompts:
            self.prompts[model] = {"versions": []}
        
        versions = self.prompts[model]["versions"]
        new_version = {
            "content": content,
            "version": f"v{len(versions) + 1}",
            "created_at": datetime.now().isoformat(),
            "description": description
        }
        
        versions.append(new_version)
        self._save_prompts()

    def get_latest_prompt(self, model: str) -> str:
        """Get the latest prompt version for a model."""
        if model not in self.prompts or not self.prompts[model]["versions"]:
            return "You are a helpful AI assistant. Please provide a detailed and accurate response."
        return self.prompts[model]["versions"][-1]["content"]

    def get_prompt_versions(self, model: str) -> List[Dict]:
        """Get all prompt versions for a model."""
        if model not in self.prompts:
            return []
        return self.prompts[model]["versions"]

    def get_available_models(self) -> List[str]:
        """Get list of models that have prompts."""
        return list(self.prompts.keys())

# Initialize with some default prompts
if __name__ == "__main__":
    manager = PromptManager()
    
    # Add default prompts if the file doesn't exist
    if not os.path.exists(manager.prompts_file):
        # Default Gemini prompt
        manager.add_prompt_version(
            "gemini",
            "You are an advanced AI assistant powered by Google's Gemini model. "
            "You excel at providing detailed, accurate, and helpful responses. "
            "Please analyze queries thoroughly and provide comprehensive answers.",
            "Initial Gemini prompt"
        )
        
        # Default Gemma prompt
        manager.add_prompt_version(
            "gemma",
            "You are an AI assistant powered by Google's Gemma model. "
            "You are focused on providing clear, concise, and accurate responses. "
            "Please maintain a helpful and informative tone while being direct and efficient.",
            "Initial Gemma prompt"
        ) 