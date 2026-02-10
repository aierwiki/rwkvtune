"""
Chat Template Formatter - unified formatting for training and inference
"""
import os
from typing import List, Dict, Optional
from jinja2 import Template, TemplateError


class ChatFormatter:
    """Format conversations using Jinja2 templates"""
    
    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize ChatFormatter
        
        Args:
            template_path: Path to Jinja2 template file. If None, uses default template.
        """
        if template_path is None:
            default_template_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "configs", "chat_templates", "rwkv_default.jinja"
            )
            template_path = default_template_path
        
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Chat template file not found: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        try:
            self.template = Template(template_content)
            self.template_path = template_path
        except TemplateError as e:
            raise ValueError(f"Invalid Jinja2 template: {e}")
    
    def format_messages(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False
    ) -> str:
        """
        Format a list of messages
        
        Args:
            messages: List of messages, each containing 'role' and 'content'.
                     Role can be 'user' or 'assistant' (no 'system' support).
            add_generation_prompt: Whether to add generation prompt (for inference)
        
        Returns:
            Formatted string
        """
        for msg in messages:
            if 'role' not in msg or 'content' not in msg:
                raise ValueError(f"Message must contain 'role' and 'content' fields: {msg}")
            
            if msg['role'] not in ['user', 'assistant']:
                raise ValueError(
                    f"Unsupported role: {msg['role']}. "
                    f"Only 'user' and 'assistant' are supported (no 'system')."
                )
        
        try:
            result = self.template.render(
                messages=messages,
                add_generation_prompt=add_generation_prompt
            )
            return result
        except Exception as e:
            raise RuntimeError(f"Template rendering failed: {e}")
    
    def convert_sharegpt_to_messages(
        self,
        conversations: List[Dict[str, str]],
        include_gpt_ignore: bool = True
    ) -> List[Dict[str, str]]:
        """
        Convert ShareGPT format to standard message format
        
        Args:
            conversations: ShareGPT format conversation list.
                          Each element contains 'from' (human/gpt/gpt_ignore) and 'value'.
            include_gpt_ignore: Whether to include gpt_ignore content
        
        Returns:
            Standard message list, each containing 'role' and 'content'
        """
        messages = []
        
        for conv in conversations:
            from_role = conv.get('from', '')
            value = conv.get('value', '')
            
            if from_role == 'human':
                messages.append({
                    'role': 'user',
                    'content': value
                })
            elif from_role == 'gpt':
                messages.append({
                    'role': 'assistant',
                    'content': value
                })
            elif from_role == 'gpt_ignore' and include_gpt_ignore:
                messages.append({
                    'role': 'assistant',
                    'content': value,
                    '_ignore': True
                })
        
        return messages
    
    def format_sharegpt(
        self,
        conversations: List[Dict[str, str]],
        add_generation_prompt: bool = False,
        include_gpt_ignore: bool = True
    ) -> str:
        """
        Directly format ShareGPT data
        
        Args:
            conversations: ShareGPT format conversation list
            add_generation_prompt: Whether to add generation prompt
            include_gpt_ignore: Whether to include gpt_ignore content
        
        Returns:
            Formatted string
        """
        messages = self.convert_sharegpt_to_messages(conversations, include_gpt_ignore)
        return self.format_messages(messages, add_generation_prompt)
    
    def __repr__(self):
        return f"ChatFormatter(template_path='{self.template_path}')"


def get_default_formatter() -> ChatFormatter:
    """Get default ChatFormatter"""
    return ChatFormatter()


def get_formatter(template_path: Optional[str] = None) -> ChatFormatter:
    """
    Get ChatFormatter instance
    
    Args:
        template_path: Template file path. None uses default template.
    
    Returns:
        ChatFormatter instance
    """
    return ChatFormatter(template_path)
