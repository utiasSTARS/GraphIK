import numpy as np
from matplotlib import pyplot as plt

# from matplotlib.cbook import get_sample_data
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import rc, cm
from graphik.robots.revolute import Revolute3dChain
from graphik.robots.robot_base import RobotRevolute
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

# mpl.rcParams['legend.fontsize'] = 10

# Use teX to render text
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
# Change the sample figure directory to the local assets/ folder in this repo
# rc('examples', directory='assets')


def plot_heatmap(
    X, Y, Z, fig=None, ax=None, title=None, xlabel=None, ylabel=None, cmap=cm.coolwarm
):
    """
    Taken from an answer at https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib by user
    Erasmus Cedernaes.
    :param X: matrix coordinates (same size as Z) from np.meshgrid
    :param Y: matrix coordinates (same size as Z) from np.meshgrid
    :param Z: matrix coordinates (same size as Z) from np.meshgrid
    :param fig: figure handle
    :param ax: axes handle
    :param title: figure title
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param cmap: colormap to use
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots()
    z = Z[:-1, :-1]
    z_min, z_max = z.min(), z.max()
    c = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=z_min, vmax=z_max)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    # set the limits of the plot to the limits of the data
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    fig.colorbar(c, ax=ax)
    return fig, ax


def plot_surface(
    X, Y, Z, fig=None, ax=None, title=None, xlabel=None, ylabel=None, cmap=cm.coolwarm
):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection="3d")

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False)
    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    return fig, ax


def get_image(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=zoom)


def plot_image(path, x, y, ax, zoom=1):
    ab = AnnotationBbox(get_image(path, zoom=zoom), (x, y), frameon=False)
    ax.add_artist(ab)


def get_tree_paths(parent_nodes):
    n_nodes = len(parent_nodes) + 1
    heads = [[0]]
    for idx, parent_node in enumerate(parent_nodes):
        for head in heads:
            if head[-1] == parent_node:
                head.append(idx + 1)
                break
        else:
            heads.append([parent_node, idx + 1])
    return heads


def plot_planar_manipulator_robot(
    robot,
    config,
    fig_handle=None,
    ax_handle=None,
    arm_color=(0.72, 0.53, 0.04, 1),
    joint_color=(0.4, 0.4, 0.4, 1),
    joint_size=20,
    joint_thickness=16.0,
    joint_edge_width=3.5,
    alpha=1.0,
):

    if fig_handle is None:
        fig_handle, ax_handle = plt.subplots()

    joint_x = []
    joint_y = []

    for i in range(robot.n + 1):
        joint_x += [robot.get_pose(config, "p" + str(i)).trans[0]]
        joint_y += [robot.get_pose(config, "p" + str(i)).trans[1]]
    joint_x = np.array(joint_x)
    joint_y = np.array(joint_y)

    for joint in list(robot.parents.keys()):
        i1 = int(joint[1:])
        for child in robot.parents[joint]:
            i2 = int(child[1:])

            ax_handle.plot(
                [joint_x[i1], joint_x[i2]],
                [joint_y[i1], joint_y[i2]],
                "o-",
                markersize=joint_size,
                markerfacecolor=joint_color,
                markeredgewidth=joint_edge_width,
                markeredgecolor=(0, 0, 0, 1),
                color=arm_color,
                linewidth=joint_thickness,
                alpha=alpha,
            )

    plt.xlim([joint_x.min() - 0.125, joint_x.max() + 0.125])
    plt.ylim([joint_y.min() - 0.125, joint_y.max() + 0.125])
    ax_handle.set_aspect("equal", adjustable="box")

    return True


def plot_planar_manipulator(
    joint_x,
    joint_y,
    parent_nodes=None,
    fig_handle=None,
    ax_handle=None,
    arm_color=(0.72, 0.53, 0.04, 1),
    joint_color=(0.4, 0.4, 0.4, 1),
    joint_size=20,
    joint_thickness=16.0,
    joint_edge_width=3.5,
    alpha=1.0,
):
    if fig_handle is None:
        fig_handle, ax_handle = plt.subplots()
        # fig_handle = plt.figure()
        # ax_handle =fig_handle.gca()

    if parent_nodes is not None:
        heads = get_tree_paths(parent_nodes)
        for head in heads:
            print("Head: {:}".format(head))
            plt.plot(
                joint_x[head],
                joint_y[head],
                "o-",
                markersize=joint_size,
                markerfacecolor=joint_color,
                markeredgewidth=joint_edge_width,
                markeredgecolor=(0, 0, 0, 1),
                color=arm_color,
                linewidth=joint_thickness,
                alpha=alpha,
            )
    else:
        plt.plot(
            joint_x,
            joint_y,
            "o-",
            markersize=joint_size,
            markerfacecolor=joint_color,
            markeredgewidth=joint_edge_width,
            markeredgecolor=(0, 0, 0, 1),
            color=arm_color,
            linewidth=joint_thickness,
            alpha=alpha,
        )
    plt.xlim([joint_x.min() - 0.125, joint_x.max() + 0.125])
    plt.ylim([joint_y.min() - 0.125, joint_y.max() + 0.125])
    ax_handle.set_aspect("equal", adjustable="box")
    return fig_handle, ax_handle


def add_fixed_point_to_graphic(fig, ax, joint_location=(0.0, 0.0), orientation=0):
    x, y = (joint_location[0], joint_location[1] - 0.042)
    plot_image("assets/fixed_point.png", x, y, ax, zoom=0.13)


def plot_3d_chain_manipulator(robot: RobotRevolute, input_angles, fig=None, ax=None):
    """
    TODO: add tree capability (same as 2D)
    :param robot:
    :param input_angles:
    :param title:
    :return:
    """
    pose_list = []
    for joint in input_angles:
        pose_list.append(robot.get_pose(input_angles, joint))
    x = []
    y = []
    z = []
    # For frame axes
    u1 = []
    v1 = []
    w1 = []
    u2 = []
    v2 = []
    w2 = []
    u3 = []
    v3 = []
    w3 = []
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.gca(projection="3d")
    # ax.set_aspect('equal', adjustable='box')
    ax.autoscale(False)
    for pose in pose_list:
        R = pose.rot.as_matrix()
        t = pose.trans
        print("t: {:}".format(t))
        print("type of t: {:}".format(type(t)))
        x.append(t[0])
        y.append(t[1])
        z.append(t[2])
        u1.append(R[0, 0])
        v1.append(R[1, 0])
        w1.append(R[2, 0])
        u2.append(R[0, 1])
        v2.append(R[1, 1])
        w2.append(R[2, 1])
        u3.append(R[0, 2])
        v3.append(R[1, 2])
        w3.append(R[2, 2])
    ax.quiver(x, y, z, u1, v1, w1, length=0.1, normalize=True, color="r")
    ax.quiver(x, y, z, u2, v2, w2, length=0.1, normalize=True, color="g")
    ax.quiver(x, y, z, u3, v3, w3, length=0.1, normalize=True, color="b")
    ax.plot(x, y, z)

    return fig, ax


if __name__ == "__main__":
    # # Test manipulator plotting
    # joint_y = np.array([0., 1., 1., 2., 2.])
    # joint_x = np.array([0., 0., 1., 0., 1.])
    # parent_joints = [0, 1, 1, 3]
    # plot_planar_manipulator(joint_x, joint_y, parent_nodes=parent_joints)
    # # add_fixed_point_to_graphic(fig, ax, joint_location=(x_joint[0], y_joint[0]))
    # plt.show()

    # Test plot_heatmap
    x = np.linspace(0.0, 1.0, 50)
    y = x
    X, Y = np.meshgrid(x, y)
    Z = x ** 2 + y * x - 2 * y + 2.0
    Z = X ** 2 + Y * X - 2 * Y + 2.0
    plot_heatmap(X, Y, Z, title="Test Computer Modern Font")
    plt.show()
