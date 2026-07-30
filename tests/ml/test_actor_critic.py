from __future__ import annotations

import torch

from ml import (
    ActorCriticConfig,
    ParametricActorCritic,
    RaggedCategorical,
    batch_candidates,
)


def _config() -> ActorCriticConfig:
    return ActorCriticConfig(
        observation_size=7,
        action_size=5,
        state_hidden_size=16,
        action_hidden_size=12,
        embedding_size=8,
        scorer_hidden_size=6,
        value_hidden_size=4,
    )


def _batch():
    return batch_candidates(
        observations=(
            (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            (0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0),
        ),
        action_groups=(
            (
                (0.0, 0.1, 0.2, 0.3, 0.4),
                (0.4, 0.3, 0.2, 0.1, 0.0),
            ),
            (
                (0.5, 0.4, 0.3, 0.2, 0.1),
                (0.1, 0.2, 0.3, 0.4, 0.5),
                (0.2, 0.3, 0.4, 0.5, 0.6),
            ),
        ),
    )


def test_actor_critic_returns_one_logit_per_candidate_and_value_per_state() -> None:
    model = ParametricActorCritic(_config())

    output = model(_batch())

    assert output.logits.shape == (5,)
    assert output.values.shape == (2,)


def test_reordering_candidates_only_reorders_logits() -> None:
    torch.manual_seed(1)
    model = ParametricActorCritic(_config())
    observation = ((0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),)
    actions = (
        (0.0, 0.1, 0.2, 0.3, 0.4),
        (0.4, 0.3, 0.2, 0.1, 0.0),
        (0.2, 0.3, 0.4, 0.5, 0.6),
    )
    original = model(batch_candidates(observation, (actions,)))
    permutation = (2, 0, 1)
    reordered = model(
        batch_candidates(
            observation,
            (tuple(actions[index] for index in permutation),),
        )
    )

    torch.testing.assert_close(
        reordered.logits,
        original.logits[list(permutation)],
    )
    torch.testing.assert_close(reordered.values, original.values)


def test_actor_critic_has_finite_gradients_and_updates_parameters() -> None:
    torch.manual_seed(2)
    model = ParametricActorCritic(_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    batch = _batch()
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    output = model(batch)
    distribution = RaggedCategorical(output.logits, batch.offsets)
    loss = (
        -distribution.log_prob(torch.tensor([0, 1])).mean()
        - 0.01 * distribution.entropy().mean()
        + output.values.square().mean()
    )
    optimizer.zero_grad()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    optimizer.step()

    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )


def test_actor_critic_handles_a_large_candidate_group_without_padding() -> None:
    model = ParametricActorCritic(_config())
    batch = batch_candidates(
        observations=((0.0,) * 7,),
        action_groups=(tuple((float(index % 3),) * 5 for index in range(2184)),),
    )

    output = model(batch)

    assert output.logits.shape == (2184,)
    assert output.values.shape == (1,)
