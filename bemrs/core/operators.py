import ast
import os
import re
import time
import logging
from .llm import InterfaceAPI as InterfaceLLM
import re
input = lambda: ...

class Evolution:

    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode, prompts, **kwargs):
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        self._use_local_llm = False
        self._url = kwargs.get('url')
        self.prompt_task = prompts.get_task()
        self.prompt_func_name = prompts.get_func_name()
        self.prompt_func_inputs = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf = prompts.get_inout_inf()
        self.prompt_other_inf = prompts.get_other_inf()
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ', '.join(("'" + s + "'" for s in self.prompt_func_inputs))
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"
        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ', '.join(("'" + s + "'" for s in self.prompt_func_outputs))
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.last_prompt_content = None
        self.last_response = None
        self.last_llm_attempts = 0
        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode)

    def get_prompt_i1(self, initialization_context=None):
        diversity_prompt = ''
        initialization_seed_prompt = ''
        initialization_seed = os.environ.get('BEMRS_INIT_PROMPT_SEED')
        if initialization_seed is not None and str(initialization_seed).strip():
            initialization_seed_prompt = '\nInitialization prompt seed: ' + str(initialization_seed).strip() + '. Use this only as a diversity cue for this initial candidate; do not include the number in the algorithm code or description.\n'
        if initialization_context:
            candidate_index = initialization_context.get('candidate_index', 1)
            previous_ideas = initialization_context.get('previous_ideas', [])
            previous_text = ''
            if previous_ideas:
                previous_text = '\nPreviously generated descriptions, provided only to avoid duplication:\n- ' + '\n- '.join((str(idea) for idea in previous_ideas))
            diversity_prompt = f'\nThis is initialization candidate No.{candidate_index}. Its purpose is to expand the initial algorithm population.\nBased only on the task specification, independently identify a useful core mechanism that is not yet represented. Do not assume or follow any predefined family of algorithms. The result must differ in its core decision logic from the previously generated ideas, not merely in variable names, constants, comments, or equivalent syntax.{previous_text}\n'
        prompt_content = self.prompt_task + '\n' + initialization_seed_prompt + diversity_prompt + 'First, describe your new algorithm and main steps in one sentence. The description must be inside a brace. Next, implement it in Python as a function named ' + self.prompt_func_name + '. This function should accept ' + str(len(self.prompt_func_inputs)) + ' input(s): ' + self.joined_inputs + '. The function should return ' + str(len(self.prompt_func_outputs)) + ' output(s): ' + self.joined_outputs + '. ' + self.prompt_inout_inf + ' ' + self.prompt_other_inf + '\n' + 'Do not give additional explanations.'
        return prompt_content

    @staticmethod
    def _format_parent_algorithms(indivs):
        prompt_indiv = ''
        for i in range(len(indivs)):
            objective = indivs[i].get('objective')
            objective_text = 'Observed objective value: ' + str(objective) + '\n' if objective is not None else ''
            prompt_indiv += 'No.' + str(i + 1) + ' algorithm and the corresponding code are: \n' + objective_text + str(indivs[i].get('algorithm') or '') + '\n' + str(indivs[i].get('code') or '') + '\n'
        return prompt_indiv

    def get_prompt_bx(self, indivs):
        prompt_indiv = self._format_parent_algorithms(indivs)
        relation = 'The reference is an anchor from the current promising search archive.' if len(indivs) == 1 else 'The references were selected to expose meaningfully different observed decision behaviors.'
        prompt_content = self.prompt_task + '\nI have ' + str(len(indivs)) + ' existing algorithms with their codes as follows: \n' + prompt_indiv + relation + '\nCreate one coherent algorithm that expands the represented behavior space. Derive reusable decision principles from the references, resolve conflicts between them, and introduce a materially different core decision mechanism. Do not concatenate parent branches, merely average formulas, tune constants, rename variables, or perform an equivalent syntactic rewrite.\nFirst, describe your new algorithm and main steps in one sentence.         The description must be inside a brace. Next, implement it in Python as a function named ' + self.prompt_func_name + '. This function should accept ' + str(len(self.prompt_func_inputs)) + ' input(s): ' + self.joined_inputs + '. The function should return ' + str(len(self.prompt_func_outputs)) + ' output(s): ' + self.joined_outputs + '. ' + self.prompt_inout_inf + ' ' + self.prompt_other_inf + '\n' + 'Do not give additional explanations.'
        return prompt_content

    def get_prompt_br(self, indivs):
        prompt_indiv = self._format_parent_algorithms(indivs)
        comparison = 'No.1 is the local reference algorithm.' if len(indivs) == 1 else 'No.1 is the local reference algorithm. The remaining algorithm(s) are behaviorally nearby contrasts with different observed performance.'
        prompt_content = self.prompt_task + '\nI have ' + str(len(indivs)) + ' algorithm reference(s) from one promising local behavior region: \n' + prompt_indiv + comparison + '\nCreate a locally refined algorithm. Preserve the effective decision backbone of No.1 and make exactly one substantive, controlled change to the mechanism most likely to improve performance or robustness. Use the contrasts only to diagnose that change. Do not rewrite the whole algorithm, add unrelated mechanisms, or rely only on renaming and constant tuning.\nFirst, describe your new algorithm and main steps in one sentence.         The description must be inside a brace. Next, implement it in Python as a function named ' + self.prompt_func_name + '. This function should accept ' + str(len(self.prompt_func_inputs)) + ' input(s): ' + self.joined_inputs + '. The function should return ' + str(len(self.prompt_func_outputs)) + ' output(s): ' + self.joined_outputs + '. ' + self.prompt_inout_inf + ' ' + self.prompt_other_inf + '\n' + 'Do not give additional explanations.'
        return prompt_content

    def get_prompt_be(self, indivs):
        prompt_indiv = self._format_parent_algorithms(indivs)
        parent_wording = 'The supplied algorithm is a behaviorally selected reference.' if len(indivs) == 1 else 'The supplied algorithms were selected to represent different observed behaviors.'
        prompt_content = self.prompt_task + '\n' + 'I have ' + str(len(indivs)) + ' existing algorithm reference(s) with their codes as follows: \n' + prompt_indiv + parent_wording + '\n' + 'The search is repeatedly producing equivalent, invalid, or already evaluated ' + 'behaviors. Create a valid algorithm with a fundamentally different core decision ' + 'mechanism. Use the references only as evidence of mechanisms to avoid, not as ' + 'templates. Do not reuse the dominant scoring structure, concatenate their branches, ' + 'rename variables, reformat code, or make only a small constant change.\n' + 'First, describe your new algorithm and main steps in one sentence. ' + 'The description must be inside a brace. Next, implement it in Python as a ' + 'function named ' + self.prompt_func_name + '. This function should accept ' + str(len(self.prompt_func_inputs)) + ' input(s): ' + self.joined_inputs + '. The function should return ' + str(len(self.prompt_func_outputs)) + ' output(s): ' + self.joined_outputs + '. ' + self.prompt_inout_inf + ' ' + self.prompt_other_inf + '\nDo not give additional explanations.'
        return prompt_content

    def _extract_code_candidate(self, response):
        """Prefer complete fenced code; retain the legacy parser as fallback.

        The legacy BeMRS parser stops at the word ``return`` and appends the
        expected output variable. That only works when the model already wrote
        ``return <expected_name>``. A complete fenced block preserves valid
        direct-return expressions and any helper definitions without changing
        old unfenced-response behavior.
        """
        fenced_blocks = re.findall('```(?:python)?\\s*(.*?)```', str(response or ''), flags=re.IGNORECASE | re.DOTALL)
        fallback_block = None
        for block in fenced_blocks:
            block = block.strip()
            if not block:
                continue
            try:
                tree = ast.parse(block)
            except SyntaxError:
                continue
            function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
            if self.prompt_func_name in function_names:
                return (block, True)
            if function_names and fallback_block is None:
                fallback_block = block
        if fallback_block is not None:
            return (fallback_block, True)
        code = re.findall('import.*return', response, re.DOTALL)
        if len(code) == 0:
            code = re.findall('def.*return', response, re.DOTALL)
        if len(code) == 0:
            return (None, False)
        return (code[0], False)

    def _get_alg(self, prompt_content):
        self.last_prompt_content = prompt_content
        response = self.interface_llm.get_response(prompt_content)
        llm_attempts = 1
        algorithm = re.findall('\\{(.*)\\}', response, re.DOTALL)
        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall('^.*?(?=python)', response, re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall('^.*?(?=import)', response, re.DOTALL)
            else:
                algorithm = re.findall('^.*?(?=def)', response, re.DOTALL)
        code, code_is_complete = self._extract_code_candidate(response)
        n_retry = 1
        while len(algorithm) == 0 or code is None:
            if self.debug_mode:
                print('Error: algorithm or code not identified, wait 1 seconds and retrying ... ')
            response = self.interface_llm.get_response(prompt_content)
            llm_attempts += 1
            algorithm = re.findall('\\{(.*)\\}', response, re.DOTALL)
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall('^.*?(?=python)', response, re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall('^.*?(?=import)', response, re.DOTALL)
                else:
                    algorithm = re.findall('^.*?(?=def)', response, re.DOTALL)
            code, code_is_complete = self._extract_code_candidate(response)
            if n_retry > 3:
                break
            n_retry += 1
        algorithm = algorithm[0]
        code_all = code if code_is_complete else code + ' ' + ', '.join((s for s in self.prompt_func_outputs))
        self.last_response = response
        self.last_llm_attempts = llm_attempts
        return [code_all, algorithm]

    def _extract_response_parts(self, response):
        algorithm = re.findall('\\{(.*)\\}', response, re.DOTALL)
        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall('^.*?(?=python)', response, re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall('^.*?(?=import)', response, re.DOTALL)
            else:
                algorithm = re.findall('^.*?(?=def)', response, re.DOTALL)
        code, code_is_complete = self._extract_code_candidate(response)
        if len(algorithm) == 0 or code is None:
            return None
        code_all = code if code_is_complete else code + ' ' + ', '.join((output for output in self.prompt_func_outputs))
        return (code_all, algorithm[0])

    def _batch_prompt(self, operator, parents, initialization_context=None):
        if operator == 'i1':
            return self.get_prompt_i1(initialization_context)
        if operator in {'bx', 'e1', 'e2'}:
            return self.get_prompt_bx(parents)
        if operator in {'br', 'm1', 'm2'}:
            return self.get_prompt_br(parents)
        if operator in {'be', 'be1'}:
            return self.get_prompt_be(parents)
        raise ValueError(f'Unsupported evolution operator: {operator}')

    def generate_batch(self, operator, parent_batches, initialization_contexts=None):
        """Generate independent offspring concurrently without changing prompts."""
        parent_batches = list(parent_batches or [])
        if initialization_contexts is None:
            initialization_contexts = [None] * len(parent_batches)
        initialization_contexts = list(initialization_contexts)
        if len(initialization_contexts) != len(parent_batches):
            raise ValueError('initialization_contexts must align with parent_batches')
        prompts = [self._batch_prompt(operator, parents, context) for parents, context in zip(parent_batches, initialization_contexts)]
        results = [None] * len(prompts)
        attempts = [0] * len(prompts)
        responses = [None] * len(prompts)
        pending = list(range(len(prompts)))
        for _ in range(4):
            if not pending:
                break
            pending_prompts = [prompts[index] for index in pending]
            if hasattr(self.interface_llm, 'get_responses'):
                pending_responses = self.interface_llm.get_responses(pending_prompts)
            else:
                pending_responses = [self.interface_llm.get_response(prompt) for prompt in pending_prompts]
            retry = []
            for index, response in zip(pending, pending_responses):
                attempts[index] += 1
                responses[index] = response
                parsed = self._extract_response_parts(response)
                if parsed is None:
                    retry.append(index)
                    continue
                code_all, algorithm = parsed
                results[index] = {'code': code_all, 'algorithm': algorithm, 'prompt': prompts[index], 'response': response, 'llm_attempts': attempts[index]}
            pending = retry
        for index in pending:
            results[index] = {'code': None, 'algorithm': None, 'prompt': prompts[index], 'response': responses[index], 'llm_attempts': attempts[index]}
        if pending:
            logging.warning('[LLMBatch] %s/%s responses remained unparsable after retries', len(pending), len(prompts))
        if results:
            last = results[-1]
            self.last_prompt_content = last['prompt']
            self.last_response = last['response']
            self.last_llm_attempts = last['llm_attempts']
        return results

    def i1(self, initialization_context=None):
        prompt_content = self.get_prompt_i1(initialization_context)
        if self.debug_mode:
            print('\n >>> check prompt for creating algorithm using [ i1 ] : \n', prompt_content)
            print(">>> Press 'Enter' to continue")
            input()
        [code_all, algorithm] = self._get_alg(prompt_content)
        if self.debug_mode:
            print('\n >>> check designed algorithm: \n', algorithm)
            print('\n >>> check designed code: \n', code_all)
            print(">>> Press 'Enter' to continue")
            input()
        return [code_all, algorithm]

    def bx(self, parents):
        prompt_content = self.get_prompt_bx(parents)
        if self.debug_mode:
            print('\n >>> check prompt for creating algorithm using [ bx ] : \n', prompt_content)
            print(">>> Press 'Enter' to continue")
            input()
        [code_all, algorithm] = self._get_alg(prompt_content)
        if self.debug_mode:
            print('\n >>> check designed algorithm: \n', algorithm)
            print('\n >>> check designed code: \n', code_all)
            print(">>> Press 'Enter' to continue")
            input()
        return [code_all, algorithm]

    def br(self, parents):
        prompt_content = self.get_prompt_br(parents)
        if self.debug_mode:
            print('\n >>> check prompt for creating algorithm using [ br ] : \n', prompt_content)
            print(">>> Press 'Enter' to continue")
            input()
        [code_all, algorithm] = self._get_alg(prompt_content)
        if self.debug_mode:
            print('\n >>> check designed algorithm: \n', algorithm)
            print('\n >>> check designed code: \n', code_all)
            print(">>> Press 'Enter' to continue")
            input()
        return [code_all, algorithm]

    def be(self, parents):
        prompt_content = self.get_prompt_be(parents)
        [code_all, algorithm] = self._get_alg(prompt_content)
        return [code_all, algorithm]
