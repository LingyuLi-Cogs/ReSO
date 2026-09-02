"""
Implementation of the Response Attack method.

This attack generates adversarial dialogues through a two-phase process
to bypass model safeguards.
"""

import os
from pathlib import Path
from typing import Literal, Optional, Dict, List
from .dialogue_generator import ResponseAttackGenerator

from ...base import BaseBlackBoxAttack
from ....base_attack import AttackResult
from .....core.registry import attack_registry
from .....judges.base_judge import BaseJudge

@attack_registry.register("response_attack")
class ResponseAttack(BaseBlackBoxAttack):
    """
    Implementation of the Response Attack.
    
    This attack generates adversarial dialogues through a two-phase process:
    Phase 1: Rewrites harmful instructions into benign forms
    Phase 2: Creates dialogue contexts (DRI or SRI) to elicit harmful responses
    """
    
    # Defaults
    DEFAULT_P1_TEMPLATE_NAME = "rewrite.md"
    DEFAULT_P2_T1_TEMPLATE_NAME = "dri.md"
    DEFAULT_P2_T2_TEMPLATE_NAME = "sri.md"
    DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent / "template"
    
    def __init__(
        self,
        model,
        helper_model,
        judge: Optional[BaseJudge] = None,
        attack_type: Literal['type1', 'type2'] = 'type1',
        template_dir: str = DEFAULT_TEMPLATE_DIR,
        p1_template: str = DEFAULT_P1_TEMPLATE_NAME,
        p2_t1_template: str = DEFAULT_P2_T1_TEMPLATE_NAME,
        p2_t2_template: str = DEFAULT_P2_T2_TEMPLATE_NAME,
        max_retries: int = 3,
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize the Response Attack.
        
        Args:
            model: The target model to attack.
            helper_model: The model used to generate attack dialogues.
            judge: Optional judge to evaluate success.
            attack_type: Type of attack ('type1' for DRI, 'type2' for SRI).
            template_dir: Directory containing prompt template files.
            p1_template: Phase 1 template filename.
            p2_t1_template: Phase 2 type1 template filename.
            p2_t2_template: Phase 2 type2 template filename.
            max_retries: Maximum number of retries for generation.
            verbose: Whether to print detailed progress information.
        """
        super().__init__(model, **kwargs)
        assert helper_model is not None, "Helper model must be provided for Response Attack"
        self.helper_model = helper_model
        self.judge = judge
        self.attack_type = attack_type
        self.max_retries = max_retries
        self.verbose = verbose
        
        # Build template paths
        self.prompt1_template_path = os.path.join(template_dir, p1_template)
        self.prompt2_template_paths = {
            'type1': os.path.join(template_dir, p2_t1_template),
            'type2': os.path.join(template_dir, p2_t2_template),
        }
        
        # Validate template paths
        if not os.path.isfile(self.prompt1_template_path):
            raise FileNotFoundError(f"Phase 1 template not found: {self.prompt1_template_path}")
        
        required_p2_path = self.prompt2_template_paths.get(attack_type)
        if not required_p2_path or not os.path.isfile(required_p2_path):
            raise FileNotFoundError(f"Phase 2 template for '{attack_type}' not found: {required_p2_path}")
        
        # Initialize generator
        self.generator = ResponseAttackGenerator(
            model=self.helper_model,
            prompt1_template_path=self.prompt1_template_path,
            prompt2_template_paths=self.prompt2_template_paths,
            verbose=self.verbose
        )
    
    def _generate_attack_dialogue(self, instruction: str) -> Optional[Dict]:
        """
        Generate an attack dialogue for the given instruction.
        
        Args:
            instruction: The harmful instruction to process.
            
        Returns:
            Dictionary containing the attack dialogue, or None if failed.
        """
        for retry in range(self.max_retries):
            try:
                attack_data = self.generator.generate_attack_json(instruction, self.attack_type)
                
                if not attack_data:
                    if self.verbose:
                        print(f"Generation failed, retry {retry + 1}/{self.max_retries}")
                    continue
                
                return attack_data
                
            except Exception as e:
                if self.verbose:
                    print(f"Error during generation: {e}, retry {retry + 1}/{self.max_retries}")
                continue
        
        return None
    
    def attack(self, target: str) -> AttackResult:
        """
        Execute the Response Attack on the target prompt.
        
        Args:
            target: The harmful prompt or objective to achieve.
            
        Returns:
            AttackResult: The result of the attack.
        """
        if self.verbose:
            print(f"\n=== Starting Response Attack ({self.attack_type}) for: {target} ===")
        
        result = AttackResult(target=target)
        
        # Generate attack dialogue
        attack_dialogue = self._generate_attack_dialogue(target)
        
        if not attack_dialogue:
            if self.verbose:
                print("\n=== Attack failed: Could not generate valid dialogue ===")
            result.success = False
            return result
        
        if self.verbose:
            print(f"Generated attack dialogue with {len(attack_dialogue.get('dialogue', []))} turns")
        
        # Extract the final adversarial prompt from the dialogue
        dialogue = attack_dialogue.get('dialogue', [])
        if dialogue:
            # The last user message is the adversarial prompt
            adversarial_prompt = dialogue[-1].get('content', '') if dialogue[-1].get('role') == 'user' else target
        else:
            adversarial_prompt = target
        
        # Query the target model with the adversarial prompt
        response = self.model.query(adversarial_prompt)
        
        if self.verbose:
            print(f"Response: {response[:200]}...")
        
        # Store results
        result.final_prompt = adversarial_prompt
        result.output_text = response
        result.history.append({
            'attack_type': self.attack_type,
            'dialogue': dialogue,
            'adversarial_prompt': adversarial_prompt,
            'response': response
        })
        
        # Evaluate success using judge
        if self.judge:
            is_success = self.judge.is_query_successful(target, response)
            result.success = is_success
            
            if self.verbose:
                print(f"Judge evaluation: {'✓ Success' if is_success else '✗ Failed'}")
                print(f"\n=== Attack {'succeeded' if is_success else 'failed'} ===")
        else:
            result.success = None
            if self.verbose:
                print("\n=== Attack completed (no judge specified) ===")
        
        return result
