from kaggle_environments import make
from hungry_geese_viewer import HungryGeeseReplayViewer

env = make("hungry_geese", debug=True)
env.run(["clever_bot.py", "clever_bot.py", "stupid_bot.py", "stupid_bot.py"])

viewer = HungryGeeseReplayViewer(env)
viewer.run()