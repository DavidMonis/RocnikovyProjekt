from kaggle_environments import make
from hungry_geese_viewer import HungryGeeseReplayViewer

env = make("hungry_geese", debug=True)
env.run(["agent.py", "agent.py", "submission.py", "submission.py"])

viewer = HungryGeeseReplayViewer(env)
viewer.run()