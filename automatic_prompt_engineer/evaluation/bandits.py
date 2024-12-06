import random
from tqdm import tqdm
from abc import ABC, abstractmethod

import numpy as np

from automatic_prompt_engineer.evaluation import elo
from automatic_prompt_engineer import evaluate


        
class BanditsEvaluator(evaluate.Evaluator):
    """Elo evaluation with LLM."""

    def __init__(self, eval_config):
        self.eval_config = eval_config
        self.inner_evaluator = elo.ELOEvaluator(eval_config['base_eval_config'])
        
    def rate_prompts(self, task, prompts, inputs):
        return self.bandits_evaluate(task,
                                     prompts,
                                     inputs)
        
    def bandit_algorithm(self, prompts):
        """
        Returns the bandit method object.
        Parameters:
            bandit_method: The bandit method to use. ('epsilon-greedy')
        Returns:
            A bandit method object.
        """
        
        if self.eval_config['bandit_method']  == 'ucb':
            return UCBBanditAlgo(prompts, self.eval_config['num_prompts_per_round'], self.eval_config['bandit_config']['c'])
        else:
            raise ValueError('Invalid bandit method.')
        
        
    def bandits_evaluate(self, task, prompts, inputs):
        rounds = self.eval_config['rounds']
        warm_rounds = min(self.eval_config['warm_rounds'], rounds)

        bandit_algo = self.bandit_algorithm(prompts)
        
        num_prompts_per_round = self.eval_config['num_prompts_per_round']
        if num_prompts_per_round < 1:
            num_prompts_per_round = int(len(prompts) * num_prompts_per_round)
            
        num_prompts_per_round = min(num_prompts_per_round, len(prompts))
        
        for round_i in tqdm(range(rounds), desc='Evaluating prompts'):
            # Sample the prompts, choose the max
            # intially, we choose the first n prompts randomly
            
            sampled_prompts_idx = bandit_algo.choose(num_prompts_per_round, warm = round_i < warm_rounds)
            sampled_prompts = [prompts[i] for i in sampled_prompts_idx]
            
            # Evaluate the sampled prompts
            # get EloEvaluationResult of ['prompt': score]
            sampled_prompt_ratings = self.inner_evaluator.rate_prompts(
                                                                task,
                                                                sampled_prompts, # topn records from algo
                                                                inputs)
            
            sampled_prompt_scores = sampled_prompt_ratings.sorted()
            
            # Update the bandit algorithm
            bandit_algo.update(sampled_prompt_scores)

        return BanditsEvaluationResult(prompts, bandit_algo.get_scores())


class BanditsEvaluationResult(evaluate.EvaluationResult):

    def __init__(self, prompts, scores):
        self.prompts = prompts
        self.scores = scores

    def sorted(self):
        """Sort the prompts and scores. There is no choice of method for now."""
        idx = np.argsort(self.scores)
        prompts, scores = [self.prompts[i]
                           for i in idx], [self.scores[i] for i in idx]
        # Reverse
        ## TODO change to return prompts, scores
        prompts, scores = prompts[::-1], scores[::-1]
        return prompts, scores
    
    def __str__(self):
        s = ''
        prompts, scores = self.sorted()
        s += 'score: prompt\n'
        s += '----------------\n'
        for prompt, score in list(zip(prompts, scores))[:10]:
            s += f'{score:.2f}: {prompt}\n'
        if len(prompts) > 10:
            s +='......\n'
        return s


class BanditAlgo(ABC):

    @ abstractmethod
    def choose(self, n, warm=True):
        """Choose n prompts from the scores.
        Parameters:
            n: The number of prompts to choose.
        Returns:
            A list of indices of the chosen prompts.
        """
        pass

    @ abstractmethod
    def update(self, sampled_scores):
        """Update the scores for the chosen prompts.
        Parameters:
            sampled_scores: sampled_scores to update prompt_ratings
        """
        pass

    @ abstractmethod
    def reset(self):
        """Reset the algorithm."""
        pass

    @ abstractmethod
    def get_scores(self):
        """Get the scores for all prompts.
        Returns:
            prompt_ratings: whole dict with prompt as key, score as value
        """
        pass



class UCBBanditAlgo(BanditAlgo):

    def __init__(self, prompts, num_samples, c):
        self.prompts = prompts
        self.num_prompts = len(prompts)
        self.num_samples = num_samples
        self.c = c
        
        self.reset()
        
    def reset(self):
        self.counts = np.zeros(self.num_prompts)
        self.scores = np.zeros(self.num_prompts)

    def update(self, scores):
        """
        prompt_ratings: prompt_ratings = {prompt: 0.0 for prompt in prompts}
        """
        for prompt, score in scores:
            i = self.prompts.index(prompt)
            self.counts[i] += self.num_samples
            self.scores[i] += score * self.num_samples

    def get_scores(self):
        # Some counts may be 0, so we need to avoid division by 0.
        return np.divide(self.scores, self.counts, out=np.zeros_like(self.scores), where=self.counts != 0)

    def choose(self, n, warm=True):
        if np.sum(self.counts) == 0 or warm is True:
            # If all counts are 0, choose randomly.
            return random.sample(range(self.num_prompts), n)
        
        ## UCB(i, t) = {x}i + sqrt(2 * ln t} / n_i)
        ## x_i: is the average reward received from arm (i) so far.
        ## n_i: is the number of times arm (i) has been pulled so far.
        ## t: is the total number of plays across all arms.
        scores = self.get_scores()
        counts = self.counts + 1e-3
        ucb_scores = scores + self.c * np.sqrt(np.log(np.sum(counts)) / counts)
        # Choose the prompts with the highest UCB scores
        # print('----ucb_scores:', ucb_scores)
        results = np.argsort(ucb_scores)[::-1][:n]
        return results
