"""
What distribution am I thinking of? 20 questions for distributions.
"""
from dataclasses import dataclass, field
import math
from typing import List

import jax
import jax.numpy as jnp
from jax.ops import index_update
import numpyro.distributions as dist
from tqdm import tqdm


class Normal(dist.Normal):
    def __str__(self):
        return f"Normal({self.loc}, {self.scale})"


@dataclass
class Question:
    text: str

    def ask(self):
        return input(self.text + " ")

    def log_prob(self, rng_key, distribution: dist.Distribution, answer: str):
        """
        Log likelihood of observing this answer for this particular distribution
        """
        raise NotImplementedError


class SampleQuestion(Question):
    def __init__(self):
        super(SampleQuestion, self).__init__("What might a typical value look like?")

    def log_prob(self, rng_key, distribution: dist.Distribution, answer: str):
        return distribution.log_prob(float(answer))

    def sample_answer(self, rng_key, distribution: dist.Distribution):
        return distribution.sample(key=rng_key)


class IntervalQuestion(Question):
    pivot: float = 1

    def __init__(self, pivot=1):
        self.pivot = pivot
        super(IntervalQuestion, self).__init__(
            f"How likely is it that the value is < {self.pivot}?"
        )

    def true_frac(self, rng_key, distribution: dist.Distribution):
        num_samples = 100
        samples = distribution.sample(key=rng_key, sample_shape=(num_samples,))
        true_frac = jnp.sum(samples < float(self.pivot)) / num_samples
        return true_frac

    def log_prob(self, rng_key, distribution: dist.Distribution, answer: str):
        true_frac = self.true_frac(rng_key, distribution)
        reported_frac = float(answer)
        return -((true_frac - reported_frac) ** 2)

    def sample_answer(self, rng_key, distribution: dist.Distribution):
        return self.true_frac(rng_key, distribution)


@dataclass
class HyperDistribution:
    # Could just be Categorical; this class doesn't need to know
    # that we're dealing with distributions

    distributions: List[dist.Distribution] = field(
        default_factory=lambda: [Normal(loc=i, scale=1) for i in range(-5, 5)]
    )
    weights: List[float] = field(default_factory=lambda: [1 / 10 for _ in range(0, 10)])

    def update(self, rng_key, question: Question, answer: str) -> "HyperDistribution":
        new_weights = jnp.array(self.weights)
        for i, (distribution, weight) in enumerate(
            zip(self.distributions, self.weights)
        ):
            prob = jnp.exp(question.log_prob(rng_key, distribution, answer))
            new_weights = index_update(new_weights, i, weight * prob)
        new_weights = new_weights / sum(new_weights)
        return HyperDistribution(self.distributions, new_weights)

    def sample(self, rng_key):
        i = dist.Categorical(jnp.array(self.weights)).sample(key=rng_key)
        return self.distributions[i]

    def most_likely_value(self):
        _, distribution = max(
            zip(self.weights, self.distributions), key=lambda pair: pair[0]
        )
        return distribution

    def __str__(self):
        return str(self.weights)


def kl_divergence(P: HyperDistribution, Q: HyperDistribution):
    # FIXME: Handle q = 0
    # FIXME: Should apply to arbitrary discrete distributions
    kl = 0.0
    for (p, q) in zip(P.weights, Q.weights):
        kl += p * math.log(p / q)
    return kl


class State:
    hyper_dist: HyperDistribution
    questions: List[Question]
    rng_key: jax.random.PRNGKey

    def __init__(self):
        self.hyper_dist = HyperDistribution()
        self.rng_key = jax.random.PRNGKey(0)
        self.questions = [
            IntervalQuestion(pivot) for pivot in range(-5, 5)
        ]  # + [SampleQuestion()]

    def next_rng_key(self):
        current_key, next_rng_key = jax.random.split(self.rng_key, 2)
        self.rng_key = next_rng_key
        return current_key

    def expected_info_gain(self, rng_key, question):
        """
        Expected information gain of asking the given question in the current state.
        """
        num_samples = 100
        info_gains = []
        for _ in tqdm(range(num_samples)):
            dist_key, answer_key, update_key, rng_key = jax.random.split(rng_key, 4)
            # Suppose the true distribution is this
            dist = self.hyper_dist.sample(dist_key)
            # Then what answer do I expect?
            answer = question.sample_answer(answer_key, dist)
            # Given that answer, what's my new posterior on distributions?
            hyper_posterior = self.hyper_dist.update(update_key, question, answer)
            # How much information have I gained from asking this question?
            info_gain = kl_divergence(hyper_posterior, self.hyper_dist)
            info_gains.append(info_gain)
        return sum(info_gains) / len(info_gains)

    def most_informative_question(self) -> Question:
        """
        Return the question with highest expected information gain.
        """
        eigs = [
            self.expected_info_gain(self.next_rng_key(), question)
            for question in self.questions
        ]
        _, max_eig_question = max(zip(eigs, self.questions))
        return max_eig_question

    def update(self, question, answer):
        self.hyper_dist = self.hyper_dist.update(self.next_rng_key(), question, answer)

    def most_likely_distribution(self):
        return self.hyper_dist.most_likely_value()

    def __str__(self):
        return str(self.hyper_dist)


def qa_loop():
    state = State()
    while True:
        print(f"Distribution weights: {state}")
        print(f"Most likely distribution: {state.most_likely_distribution()}")
        question = state.most_informative_question()
        answer = question.ask()
        state.update(question, answer)


if __name__ == "__main__":
    qa_loop()
