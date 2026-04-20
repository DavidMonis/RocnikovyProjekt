from kaggle_environments import make
from hungry_geese_viewer import HungryGeeseReplayViewer

env = make("hungry_geese", debug=True)
env.run(["bot.py", "bot.py", "smart_bot.py", "smart_bot.py"])

viewer = HungryGeeseReplayViewer(env)
viewer.run()