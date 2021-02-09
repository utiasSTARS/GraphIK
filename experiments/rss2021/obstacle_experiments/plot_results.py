import os, sys
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

if __name__ == "__main__":
    data = pickle.load(
        open(
            os.path.dirname(os.path.realpath(__file__))
            + "/results/ur10_"
            + str(sys.argv[1])
            + ".p",
            "rb",
        )
    )

    cvl = data["Constraint Violations"]
    print(cvl[cvl["type"] == "obstacle"])

    # NOTE plot ideas:
    # - success rate
    # - when constraints are violated, which ones are they?
    # - how bad are the obstacle violations?

    # print(data[(data["Pos. Error"] < 0.001) & (data["Rot. Error"] < 0.001)])
    # print(data["Rot. Error"].mean())
    # print(data[data["Viol. Type"].isin(["obstacle"])]["Viol. Value"])
    # print(data["Viol. Type"])
