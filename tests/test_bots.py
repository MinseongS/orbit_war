"""Smoke + property tests for the bot zoo."""

from kaggle_environments import make

from orbit_war.bots import greedy_baseline, random_bot, starter_bot


def _first_obs():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.reset(num_agents=2)
    return env.steps[0][0]["observation"]


def test_random_bot_returns_list():
    obs = _first_obs()
    moves = random_bot.agent(obs)
    assert isinstance(moves, list)
    for move in moves:
        assert len(move) == 3
        from_id, angle, ships = move
        assert isinstance(from_id, int)
        assert isinstance(angle, float)
        assert isinstance(ships, int)


def test_starter_bot_runs_a_full_game_against_random():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run([starter_bot.agent, random_bot.agent])
    final = env.steps[-1]
    assert all(s.status == "DONE" for s in final)


def test_greedy_baseline_runs_a_full_game_against_random():
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run([greedy_baseline.agent, random_bot.agent])
    final = env.steps[-1]
    assert all(s.status == "DONE" for s in final)


def test_greedy_beats_random_on_a_few_seeds():
    """Property: a sane greedy bot should beat a uniform-random bot
    in a clear majority of games. If this fails, greedy has a bug."""
    wins = 0
    for seed in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        env = make("orbit_wars", configuration={"seed": seed}, debug=True)
        env.run([greedy_baseline.agent, random_bot.agent])
        rewards = [s.reward for s in env.steps[-1]]
        if rewards[0] > rewards[1]:
            wins += 1
    assert wins >= 8, f"greedy only beat random {wins}/10 seeds"
