import inspect

from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration


def main() -> None:
    """
    Small helper script for inspecting Kaggle's Hungry Geese Configuration class.

    This is only a debugging/reference utility. It is not used by the agent,
    training loop, simulator, or submission.
    """
    print(inspect.getsource(Configuration))


if __name__ == "__main__":
    main()