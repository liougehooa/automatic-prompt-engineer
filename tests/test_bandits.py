import numpy.testing as npt
import numpy as np

from automatic_prompt_engineer.evaluation import bandits


def test_counting():
    prompts = ['prompt1', 'prompt2', 'prompt3', 'prompt4', 'prompt5']
    num_samples = 3
    algo = bandits.UCBBanditAlgo(prompts, num_samples, c=1)
    
    algo.update(scores=[('prompt1', 5), ('prompt2', 10)])
    algo.update(scores=[('prompt1', 1), ('prompt2',1), ('prompt3',1),  ('prompt4',1),  ('prompt5',1)])
    
    scores = algo.get_scores()
    chosen_scores = algo.choose(n=2, warm=False)

    npt.assert_array_equal(scores, np.array([3, 5.5, 1, 1, 1]))
    # npt.assert_array_equal(chosen_scores, [])
    npt.assert_array_equal(chosen_scores, np.array([1, 0]))

