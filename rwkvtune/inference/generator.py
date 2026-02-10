"""
Text generation utilities
"""

from typing import List, Optional, Dict, Any
from rwkvtune.inference.pipeline import InferencePipeline


class TextGenerator:
    """Text generator - provides higher-level generation interface"""
    
    PRESETS = {
        "greedy": {
            "temperature": 1.0,
            "top_p": 0.0,
            "top_k": 0,
        },
        "creative": {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 0,
        },
        "precise": {
            "temperature": 0.3,
            "top_p": 0.5,
            "top_k": 0,
        },
        "balanced": {
            "temperature": 0.6,
            "top_p": 0.8,
            "top_k": 0,
        },
    }
    
    def __init__(self, pipeline: InferencePipeline):
        """
        Initialize generator
        
        Args:
            pipeline: Inference pipeline
        """
        self.pipeline = pipeline
    
    def generate(
        self,
        prompt: str,
        preset: str = "balanced",
        max_tokens: int = 100,
        **kwargs
    ) -> str:
        """
        Generate text using preset configuration
        
        Args:
            prompt: Input prompt
            preset: Preset name (greedy/creative/precise/balanced)
            max_tokens: Maximum tokens to generate
            **kwargs: Other parameters to override preset
        
        Returns:
            Generated text
        """
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(self.PRESETS.keys())}")
        
        gen_kwargs = {**self.PRESETS[preset], **kwargs}
        return self.pipeline.generate(prompt, max_tokens=max_tokens, **gen_kwargs)
    
    def complete(
        self,
        prompt: str,
        max_tokens: int = 50,
        n: int = 1,
        **kwargs
    ) -> List[str]:
        """
        Generate multiple completions
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            n: Number of completions
            **kwargs: Other parameters passed to generate
        
        Returns:
            List of completions
        """
        result = self.pipeline.generate(
            prompt, 
            max_tokens=max_tokens, 
            num_generations=n,
            **kwargs
        )
        return result if isinstance(result, list) else [result]
    
    def interactive_chat(self, system_prompt: Optional[str] = None):
        """
        Interactive chat
        
        Args:
            system_prompt: System prompt
        """
        print("=" * 60)
        print("RWKV-Kit Interactive Chat")
        print("Type 'quit' or 'exit' to exit")
        print("=" * 60)
        
        if system_prompt:
            print(f"\nSystem: {system_prompt}\n")
        
        history = []
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("\nAssistant: ", end="")
                response = self.pipeline.chat(
                    user_input,
                    system_prompt=system_prompt,
                    history=history,
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=200,
                    stream=True,
                )
                
                history.append((user_input, response))
                
                if len(history) > 5:
                    history = history[-5:]
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue
