"""
Implementation of the ArtPrompt Attack method.

This attack uses ASCII art to mask sensitive words and bypass model safeguards.
"""

import re
from typing import Literal, Optional

import nltk
from nltk.corpus import stopwords


from . import base_prompt as bp
from ...base import BaseBlackBoxAttack
from ....base_attack import AttackResult
from .....core.registry import attack_registry
from .....judges.base_judge import BaseJudge


_OFFLINE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "what", "when", "where", "which", "who",
    "will", "with", "write", "you", "your",
}


def _english_stopwords():
    """Return NLTK stopwords without ever attempting a network download."""
    try:
        return set(stopwords.words('english'))
    except LookupError:
        return _OFFLINE_STOPWORDS


def _ensure_nltk_data():
    """Validate the offline fallback during construction."""
    _english_stopwords()


@attack_registry.register("artprompt")
class ArtPromptAttack(BaseBlackBoxAttack):
    """
    Implementation of the ArtPrompt Attack.
    
    This attack uses ASCII art representations of sensitive words to bypass
    content moderation systems by obfuscating harmful keywords.
    """
    
    SUPPORTED_FONTS = ['gen', 'alphabet', 'letters', 'keyboard', 'cards', 'puzzle']
    SUPPORTED_DIRECTIONS = ['h', 'v']  # horizontal, vertical
    SUPPORTED_MASKING = ['stop', 'gpt']  # stop words filtering, GPT-based masking
    
    def __init__(
        self,
        model,
        judge: Optional[BaseJudge] = None,
        font_name: Literal['gen', 'alphabet', 'letters', 'keyboard', 'cards', 'puzzle'] = 'gen',
        direction: Literal['h', 'v'] = 'h',
        masking_strategy: Literal['stop', 'gpt'] = 'stop',
        attack_model = None,
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize the ArtPrompt Attack.
        
        Args:
            model: The target model to attack.
            judge: Optional judge to evaluate success.
            font_name: ASCII art font to use for encoding. Options:
                      'gen', 'alphabet', 'letters', 'keyboard', 'cards', 'puzzle'
            direction: Direction of ASCII art. Options: 'h' (horizontal), 'v' (vertical)
            masking_strategy: Strategy for masking sensitive words. Options:
                            'stop' (stop words filtering), 'gpt' (GPT-based masking)
            attack_model: Model to use for GPT-based masking (required if masking_strategy='gpt')
            verbose: Whether to print detailed progress information.
        """
        super().__init__(model, **kwargs)
        self.judge = judge
        self.font_name = font_name
        self.direction = direction
        self.masking_strategy = masking_strategy
        self.attack_model = attack_model
        self.verbose = verbose
        
        # Ensure NLTK stopwords data is available (needed for stop-word masking)
        _ensure_nltk_data()
        
        if font_name not in self.SUPPORTED_FONTS:
            raise ValueError(
                f"Unsupported font: {font_name}. "
                f"Supported fonts: {', '.join(self.SUPPORTED_FONTS)}"
            )
        
        if direction not in self.SUPPORTED_DIRECTIONS:
            raise ValueError(
                f"Unsupported direction: {direction}. "
                f"Supported directions: {', '.join(self.SUPPORTED_DIRECTIONS)}"
            )
        
        if masking_strategy not in self.SUPPORTED_MASKING:
            raise ValueError(
                f"Unsupported masking strategy: {masking_strategy}. "
                f"Supported strategies: {', '.join(self.SUPPORTED_MASKING)}"
            )
        
        if masking_strategy == 'gpt' and attack_model is None:
            raise ValueError("attack_model is required when masking_strategy='gpt'")
        
        # Initialize the VITC attack prompt generator based on direction
        if direction == 'h':
            self.prompt_generator = bp.vitc_horizontal()
        else:  # direction == 'v'
            self.prompt_generator = bp.vitc_vertical()
        
        self.prompt_gen_func = getattr(self.prompt_generator, font_name)
        
        # Initialize GPT masking prompts if needed
        if masking_strategy == 'gpt':
            self.gpt_mask_prompt = bp.GPTMaskPrompt()
            self.gpt_mask_response = bp.GPTMaskResponse()
        
    def _mask_word_stop(self, sentence: str, mask_token: str = '[MASK]'):
        """
        Mask words in a sentence by filtering stop words.
        
        Args:
            sentence: The sentence to mask.
            mask_token: The token to use for masking.
            
        Returns:
            List of tuples: [(masked_sentence, masked_word), ...]
        """
        words = sentence.split(' ')
        masked_s_kw = []
        filtered_sentence = [
            word for word in words 
            if word.lower() not in _english_stopwords() and word.isalpha()
        ]

        for word in filtered_sentence:
            masked_s_kw.append((sentence.replace(word, mask_token), word))
        
        return masked_s_kw
    
    def _mask_word_gpt(self, sentence: str):
        """
        Mask words in a sentence using GPT-based masking.
        
        Args:
            sentence: The sentence to mask.
            
        Returns:
            List of tuples: [(masked_sentence, masked_word), ...]
        """
        # Use GPT to identify and mask sensitive words
        prompt = self.gpt_mask_prompt.get_prompt(sentence)
        response = self.attack_model.query(prompt)
        
        if self.verbose:
            print(f"GPT Masking Response: {response}")
        
        # Parse the response to get masked words and instruction
        try:
            msk_words, msk_instruction = self.gpt_mask_prompt.parse(response)
            
            # Local instruction models sometimes return a phrase (or quoted
            # punctuation) where ArtPrompt's 5x5 renderer expects A-Z only.
            # Keep the phrase semantics while compacting it to renderable
            # letters, and normalize numbered mask placeholders.
            masked_instruction = re.sub(r"\[MASK\d*\]", "[MASK]", msk_instruction)
            masked_s_kw = []
            for word in msk_words:
                renderable_word = "".join(
                    character for character in str(word) if character.isascii()
                    and character.isalpha()
                )
                if renderable_word:
                    masked_s_kw.append((masked_instruction, renderable_word))
            return masked_s_kw or self._mask_word_stop(sentence)
        except Exception as e:
            if self.verbose:
                print(f"Error parsing GPT masking response: {e}")
            # Fallback to stop words filtering
            return self._mask_word_stop(sentence)
    
    def _generate_attack_prompts(self, target: str):
        """
        Generate attack prompts by masking sensitive words with ASCII art.
        
        Args:
            target: The original harmful instruction.
            
        Returns:
            List of tuples: [(attack_prompt, masked_word), ...]
        """
        # Step 1: Mask words based on strategy
        if self.masking_strategy == 'stop':
            masked_s_kw = self._mask_word_stop(target)
        else:  # masking_strategy == 'gpt'
            masked_s_kw = self._mask_word_gpt(target)
        
        if self.verbose:
            print(f"Masked words: {[word for _, word in masked_s_kw]}")
        
        # Step 2: Generate attack prompts for each masked word
        attack_prompt_list = []
        for masked_instruction, masked_word in masked_s_kw:
            attack_prompt = self.prompt_gen_func(masked_instruction, masked_word)
            attack_prompt_list.append((attack_prompt, masked_word))
        
        return attack_prompt_list
    
    def attack(self, target: str) -> AttackResult:
        """
        Execute the ArtPrompt Attack on the target prompt.
        
        Args:
            target: The harmful prompt or objective to achieve.
            
        Returns:
            AttackResult: The result of the attack.
        """
        if self.verbose:
            print(f"\n=== Starting ArtPrompt Attack ===")
            print(f"Font: {self.font_name}, Direction: {self.direction}, Masking: {self.masking_strategy}")
            print(f"Target: {target}")
        
        result = AttackResult(target=target)
        
        # Generate all attack prompts
        attack_prompts = self._generate_attack_prompts(target)
        
        if self.verbose:
            print(f"Generated {len(attack_prompts)} attack prompts")
        
        # Try each attack prompt
        for idx, (attack_prompt, masked_word) in enumerate(attack_prompts, 1):
            if self.verbose:
                print(f"\n--- Attack Prompt {idx}/{len(attack_prompts)}: (masked word: '{masked_word}') ---")
            
            # Query the target model
            response = self.model.query(attack_prompt)
            
            if self.verbose:
                print(f"Response: {response[:200]}...")
            
            # Store in history
            result.history.append({
                'attempt': idx,
                'masked_word': masked_word,
                'attack_prompt': attack_prompt,
                'response': response
            })
            
            # Check if successful using judge
            if self.judge:
                is_success = self.judge.is_query_successful(target, response)
                if self.verbose:
                    print(f"Judge evaluation: {'✓ Success' if is_success else '✗ Failed'}")
                
                if is_success:
                    result.final_prompt = attack_prompt
                    result.output_text = response
                    result.success = True
                    if self.verbose:
                        print(f"\n=== Attack succeeded at attempt {idx} ===")
                    return result
            else:
                # Without a judge, we consider the first response as the result
                result.final_prompt = attack_prompt
                result.output_text = response
                result.success = None
                if self.verbose:
                    print("\n=== Attack completed (no judge specified) ===")
                return result
        
        # Attack failed after all attempts
        if result.history:
            last_result = result.history[-1]
            result.final_prompt = last_result['attack_prompt']
            result.output_text = last_result['response']
        
        result.success = False
        if self.verbose:
            print("\n=== Attack failed after all attempts ===")
        
        return result
