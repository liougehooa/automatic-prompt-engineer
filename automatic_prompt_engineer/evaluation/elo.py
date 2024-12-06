import itertools

from automatic_prompt_engineer import llm, data, template, evaluate
import numpy as np

import asyncio
import nest_asyncio

nest_asyncio.apply()

class ELOEvaluator(evaluate.Evaluator):
    """Elo evaluation with LLM."""
    
    @staticmethod
    def _expected_score(r1, r2):
        return 1 / (1 + 10**((r2 - r1) / 400))

    @staticmethod
    def _update_elo(r1, r2, score, k):
        e1 = ELOEvaluator._expected_score(r1, r2)
        e2 = ELOEvaluator._expected_score(r2, r1)
        return r1 + k * (score - e1), r2 + k * ((1 - score) - e2)

    def __init__(self, eval_config):
        self.eval_config = eval_config
        self.gen_model_config = eval_config['gen_model']
        self.score_model_config = eval_config['score_model']
        self.llm_gen_model = llm.model_from_config(eval_config['gen_model'])
        self.llm_score_model = llm.model_from_config(eval_config['score_model'])
        self.eval_template = template.EvalTemplate(eval_config['ranking_system_prompt'], eval_config['user_template'])
        
        self.score_model_config['logit_bias'] = {
              '32': 100,  # 'A' token
              '33': 100,  # 'B' token
            }
    
    def rate_prompts(self, task, prompts, inputs):
        all_scores = self.generate_prompts_output(task, prompts, inputs)
        
        prompt_ratings = self.rating_scores(prompts, all_scores)

        return EloEvaluationResult(task, prompt_ratings)

    def rating_scores(self, prompts, all_scores):
        prompt_ratings = {prompt: 100 for prompt in prompts}
        
        for score in all_scores:
            prompt1, prompt2 = score['prompt1'], score['prompt2']
            r1, r2 = prompt_ratings[prompt1], prompt_ratings[prompt2]
            
            score = score['avg_score']
            r1, r2 = ELOEvaluator._update_elo(r1, r2, score, self.eval_config['elo_k'])
            prompt_ratings[prompt1], prompt_ratings[prompt2] = r1, r2
        return prompt_ratings

    def _generate_output(self, prompt, input_):
        return self.llm_gen_model.complete_prompt(messages=self.eval_template.fill_generate_messages(prompt, input_),
                                                  config=self.gen_model_config['gpt_config'])[0].message.content


    def _score_output(self, task, input_, generation_1, generation_2):
        score = self.llm_score_model.complete_prompt(messages=self.eval_template.fill_score_messages(task, input_, generation_1, generation_2),
                                               config=self.score_model_config['gpt_config'])[0].message.content
        
        return 1.0 if score == 'A' else 0.0 if score == 'B' else 0.5

    def score_output(self, task, input_, generation_1, generation_2):
        score1 = self._score_output(task, input_, generation_1, generation_2)
        score2 = 1 - self._score_output(task, input_, generation_2, generation_1)

        return (score1 + score2)/2

    async def score_generation(self, task, prompt1, prompt2, input_, scores):
        generation_1 = self._generate_output(prompt1, input_)
        generation_2 = self._generate_output(prompt2, input_)
                                                              
        avg_score = self.score_output(task, input_, generation_1, generation_2)
        scores.append({'input': input_, 'prompt1': prompt1, 'prompt2': prompt2, 'avg_score': avg_score})
        
    async def batch_score_generation(self, task, inputs, prompt1, prompt2, scores):
        await asyncio.gather(*[self.score_generation(task, prompt1, prompt2, input_, scores) for input_ in inputs])

    def generate_prompts_output(self, task, prompts, inputs):
        batch_size = self.eval_config['batch_size']
        
        all_scores = []
        for prompt1, prompt2 in itertools.combinations(prompts, 2):
            sampled_inputs = data.subsample_single_data(
                inputs, self.eval_config['num_samples'])
            
            sampled_inputs_batches = [sampled_inputs[i:i + batch_size]
                                      for i in range(0, len(sampled_inputs), batch_size)]
            for input_batch in sampled_inputs_batches:
                scores = []
                asyncio.run(self.batch_score_generation(task, input_batch, prompt1, prompt2, scores))
                all_scores.extend(scores)
        
        return all_scores


class EloEvaluationResult(evaluate.EvaluationResult):
    """
    A class for storing the results of a likelihood evaluation. Supports
    sorting prompts by various statistics of the likelihoods.
    """

    def __init__(self, task, prompts_ratings):
        self.task = task
        self.prompts_ratings = prompts_ratings

    def sorted(self):
        """Get the results in the form of a sorted prompt and score list.
        Has a method argument to support sorting by various metrics."""
        if self.prompts_ratings is None:
            return []
        
        
        return [(prompt, rating) for prompt, rating in sorted(self.prompts_ratings.items(), key=lambda item: item[1], reverse=True)]

    
    def top_n(self, n:int):
        assert n > 0, "n must be greater than 0."
        
        if self.prompts_ratings is None:
            return []
        
        return self.sorted()[:n]
