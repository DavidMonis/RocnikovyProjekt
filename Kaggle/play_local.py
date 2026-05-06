from kaggle_environments import make
from other.hungry_geese_viewer import HungryGeeseReplayViewer

env = make("hungry_geese", debug=True)
env.run(["submission.py", "winning_agent/kaggle_public_agent.py", "bots/stupid_bot.py", "bots/stupid_bot.py"])

viewer = HungryGeeseReplayViewer(env)
viewer.run()