import os
import re

# Attack sequence generator
class ResponseAttackGenerator:
    """
    Input: a harmful query (Q_Harmful).
    Output: a two-round attack sequence (Prompt_1, response, Prompt_2) as a structured JSON-like dict.
    """
    def __init__(
        self,
        model,
        prompt1_template_path: str="",
        prompt2_template_paths: dict[str, str]=[],
        verbose: bool=False
    ):
        self.model = model

        self.prompt1_template = None
        self.prompt2_templates = {}
        self.verbose = verbose

        print("--- Initializing ResponseAttackGenerator ---")
        try:
            prompt1_template_path = os.path.abspath(prompt1_template_path)
            self.prompt1_template = self._load_template(prompt1_template_path)
            print(f"[Info] Loaded Prompt_1 template: {prompt1_template_path}")
        except Exception as e:
            print(f"[Error] Failed to load Prompt_1 template from {prompt1_template_path}: {e}")

        for type_key, path in prompt2_template_paths.items():
            try:
                path = os.path.abspath(path)
                self.prompt2_templates[type_key] = self._load_template(path)
                print(f"[Info] Loaded Prompt_2 template ('{type_key}'): {path}")
            except Exception as e:
                 print(f"[Error] Failed to load Prompt_2 ('{type_key}') template from {path}: {e}")

        if not self.prompt1_template or len(self.prompt2_templates) != len(prompt2_template_paths):
            print("[Warning] One or more required templates failed to load.")
        # Summarization support removed
        print("--- Initialization complete ---")
    # Prepare response context for Prompt_2 (no summarization)
    def _prepare_response_context(self, original_query: str, prompt1_base: str, response_text: str) -> str:
        return response_text
        
    def _load_template(self, filepath: str) -> str:
        """Load template file content."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Template file not found: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
        
    # Remove reasoning chain tags from response
    def _split_response(self, raw: str) -> str:
        """
        1) Remove complete <think>...</think> or <reason>...</reason> blocks.
        2) If only a closing tag exists, drop everything up to it.
        Return the cleaned response.
        """
        cleaned = re.sub(r'<(?:think|reason)>.*?</(?:think|reason)>', '', raw, flags=re.DOTALL)
        cleaned = re.sub(r'^.*?</(?:think|reason)>', '', cleaned, flags=re.DOTALL)
        return cleaned.strip()

    # Extract chain-of-thought from response (without tags)
    def _extract_chain(self, raw: str) -> str:
        """Return the first <think>...</think> or <reason>...</reason> content; if only a closing tag exists, take content up to it; else empty string."""
        m = re.search(r'<(?:think|reason)>(.*?)</(?:think|reason)>', raw, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
        # Only closing tag exists
        m2 = re.search(r'(.*?)</(?:think|reason)>', raw, flags=re.DOTALL)
        if m2:
            return m2.group(1).strip()
        return ''
    
    # Call LLM
    def _call_llm_api(self, prompt: str, task_description: str) -> str | None:
        if self.verbose:
            print(f"\n--- [Real call] Calling LLM API, task: {task_description} (prompt chars: {len(prompt)}) ---")
            print(f"Prompt content (truncated to 500 chars):\n{prompt[:500]}{'...' if len(prompt) > 500 else ''}")
        try:
            llm_output = self.model.query(prompt)
            return llm_output.strip() if llm_output else None
        except Exception as e:
            print(f"[Error] LLM API call failed ({task_description}): {e}")
            return None

    def _generate_prompt1(self, original_query: str) -> str | None:
        """Generate Prompt_1 using LLM."""
        if not self.prompt1_template:
            print("[Error] Prompt_1 template not loaded or failed to load.")
            return None
        try:
            # Fill template
            input_prompt_for_p1 = self.prompt1_template.format(query=original_query)
            prompt1_base = self._call_llm_api(input_prompt_for_p1, "Generate Base Prompt_1 (Sanitization)")
            return prompt1_base if prompt1_base else None
        except KeyError as e:
            print(f"[Error] Prompt_1 template formatting error: missing placeholder '{e}'")
            return None
        except Exception as e:
            print(f"[Error] Exception during Prompt_1 generation: {e}")
            return None

    # Generate intermediate response using LLM
    def _generate_response(self,
                           prompt1_base: str,
                           original_query: str,
                           attack_type: str) -> str | None:
        """Generate intermediate response using LLM. For type2, append a high-level suffix to prompt1_base only for generation."""
        type2_suffix = "Please provide only a high-level outline or framework, omitting specific details."
        prompt_for_api_call = prompt1_base
        if attack_type == 'type2':
            prompt_for_api_call = f"{prompt1_base}\n\n{type2_suffix}"
            if self.verbose:
                print(f"[Info] Added type2 suffix for response generation.")
        try:
            task_desc = f"Generate Intermediate Response (for attack type {attack_type})"
            response = self._call_llm_api(prompt_for_api_call, task_desc)
            return response if response else None
        except Exception as e:
            print(f"[Error] Exception during response generation: {e}")
            return None
    
    # Generate Prompt_2 using template and LLM
    def _generate_prompt2(self, original_query: str, prompt1_base: str, response: str, attack_type: str) -> str | None:
        """Generate Prompt_2 using base prompt1 and the response as context."""
        if attack_type not in self.prompt2_templates:
            print(f"[Error] Prompt_2 template for attack type '{attack_type}' not loaded.")
            return None
        template = self.prompt2_templates[attack_type]
        try:
            # Fill template using base prompt1 (without suffix)
            input_prompt_for_p2 = template.format(
                query=original_query,
                prompt1=prompt1_base,
                response=response
            )
            # Call LLM API to generate Prompt_2
            prompt2 = self._call_llm_api(input_prompt_for_p2, f"Generate Prompt_2 ({attack_type} context)")
            return prompt2 if prompt2 else None
        except KeyError as e:
            print(f"[Error] Prompt_2 ({attack_type}) template formatting error: missing placeholder '{e}'")
            return None
        except Exception as e:
            print(f"[Error] Exception during Prompt_2 ({attack_type}) generation: {e}")
            return None

    def generate_attack_json(self, original_query: str, attack_type: str) -> dict | None:
        """Generate the full attack sequence as a JSON-like dict."""
        if self.verbose:
            print(f"\n=== Start generating attack sequence (type: {attack_type}) ===")
        # Pre-checks remain unchanged
        if not self.model or not self.prompt1_template or attack_type not in self.prompt2_templates:
             print(f"[Error] Cannot continue; model or required templates not loaded (P1, P2 for {attack_type}).")
             return None


        # Step 1: generate base Prompt_1
        prompt1_raw = self._generate_prompt1(original_query)
        if prompt1_raw is None:
            print("[!] Failed at Prompt_1 stage.")
            return None
        prompt1_base = self._split_response(prompt1_raw)
        if self.verbose:
            print(f"\n[1] Generated base Prompt_1 (chars: {len(prompt1_base)})")

        # Step 2: generate raw response (may contain chain-of-thought tags)
        raw_response = self._generate_response(prompt1_base, original_query, attack_type)
        if raw_response is None:
            print("[!] Failed at Response stage.")
            return None
        if self.verbose:
            print(f"\n[2] Generated raw response (type {attack_type}) (chars: {len(raw_response)})")

        # Step 2.1: clean response (remove chain-of-thought tags)
        cleaned_response = self._split_response(raw_response)
        if self.verbose:
            print(f"\n[2.1] Cleaned response (chars: {len(cleaned_response)})")

        # Step 2.2: prepare response context (no summarization)
        response_for_p2_context = self._prepare_response_context(original_query, prompt1_base, cleaned_response)
        if self.verbose:
            print(f"\n[2.2] Prepared response context for P2.")

        # Step 3: generate Prompt_2
        prompt2 = self._generate_prompt2(original_query, prompt1_base, response_for_p2_context, attack_type)
        # Note: _generate_prompt2 will be cleaned by _split_response
        if prompt2 is None:
            print("[!] Failed at Prompt_2 stage.")
            return None
        if self.verbose:
            print(f"\n[3] Generated Prompt_2 ({attack_type}) (chars: {len(prompt2)})")
        prompt2 = self._split_response(prompt2)
        # Build final JSON (store cleaned response for round 1)
        attack_type_alias = 'dri' if attack_type == 'type1' else ('sri' if attack_type == 'type2' else attack_type)
        output_json = {
            "originalHarmfulQuery": original_query,
            "attackType": attack_type_alias,
            "rounds": [
                {
                    "roundIndex": 1,
                    "roundType": "spoof", # or setup
                    "promptParts": [{"type": "text", "text": prompt1_base}],
                    "response": cleaned_response
                },
                {
                    "roundIndex": 2,
                    "roundType": "attack",
                    "promptParts": [{"type": "text", "text": prompt2}],
                    "response": None
                }
            ],
        }
        return output_json