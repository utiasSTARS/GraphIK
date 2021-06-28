#!/bin/sh

NUM_PROB=3000

# python generate_problem_set.py --robot "ur10" "kuka" "schunk" --num_prob ${NUM_PROB} --env "" &
# python generate_problem_set.py --robot "ur10" "kuka" "schunk" --num_prob ${NUM_PROB} --env "octahedron" &
# python generate_problem_set.py --robot "ur10" "kuka" "schunk" --num_prob ${NUM_PROB} --env "cube" &
# python generate_problem_set.py --robot "ur10" "kuka" "schunk" --num_prob ${NUM_PROB} --env "icosahedron" &
# wait

# python local_obstacle.py --robot "ur10" --num_prob 100
# python local_obstacle.py --robot "kuka" --num_prob 100
# python local_obstacle.py --robot "schunk" --num_prob 100
python joint_obstacle.py --robot "ur10" --num_prob ${NUM_PROB}
python joint_obstacle.py --robot "kuka" --num_prob ${NUM_PROB}
python joint_obstacle.py --robot "schunk" --num_prob ${NUM_PROB}
# python riemannian_obstacle.py --robot "ur10" --num_prob ${NUM_PROB}
# python riemannian_obstacle.py --robot "kuka" --num_prob ${NUM_PROB}
# python riemannian_obstacle.py --robot "schunk" --num_prob ${NUM_PROB}

# python local_obstacle.py --robot "ur10" --num_prob 100 --env "octahedron"
# python local_obstacle.py --robot "kuka" --num_prob 100 --env "octahedron"
# python local_obstacle.py --robot "schunk" --num_prob 100 --env "octahedron"
python joint_obstacle.py --robot "ur10" --num_prob ${NUM_PROB} --env "octahedron"
python joint_obstacle.py --robot "kuka" --num_prob ${NUM_PROB} --env "octahedron"
python joint_obstacle.py --robot "schunk" --num_prob ${NUM_PROB} --env "octahedron"
python riemannian_obstacle.py --robot "ur10" --num_prob ${NUM_PROB} --env "octahedron"
python riemannian_obstacle.py --robot "kuka" --num_prob ${NUM_PROB} --env "octahedron"
python riemannian_obstacle.py --robot "schunk" --num_prob ${NUM_PROB} --env "octahedron"

# python local_obstacle.py --robot "ur10" --num_prob 100 --env "cube"
# python local_obstacle.py --robot "kuka" --num_prob 100 --env "cube"
# python local_obstacle.py --robot "schunk" --num_prob 100 --env "cube"
python joint_obstacle.py --robot "ur10" --num_prob ${NUM_PROB} --env "cube"
python joint_obstacle.py --robot "kuka" --num_prob ${NUM_PROB} --env "cube"
python joint_obstacle.py --robot "schunk" --num_prob ${NUM_PROB} --env "cube"
python riemannian_obstacle.py --robot "ur10" --num_prob ${NUM_PROB} --env "cube"
python riemannian_obstacle.py --robot "kuka" --num_prob ${NUM_PROB} --env "cube"
python riemannian_obstacle.py --robot "schunk" --num_prob ${NUM_PROB} --env "cube"

# python local_obstacle.py --robot "ur10" --num_prob 100 --env "icosahedron"
# python local_obstacle.py --robot "kuka" --num_prob 100 --env "icosahedron"
# python local_obstacle.py --robot "schunk" --num_prob 100 --env "icosahedron"
python joint_obstacle.py --robot "ur10" --num_prob ${NUM_PROB} --env "icosahedron"
python joint_obstacle.py --robot "kuka" --num_prob ${NUM_PROB} --env "icosahedron"
python joint_obstacle.py --robot "schunk" --num_prob ${NUM_PROB} --env "icosahedron"
python riemannian_obstacle.py --robot "ur10" --num_prob ${NUM_PROB} --env "icosahedron"
python riemannian_obstacle.py --robot "kuka" --num_prob ${NUM_PROB} --env "icosahedron"
python riemannian_obstacle.py --robot "schunk" --num_prob ${NUM_PROB} --env "icosahedron"
wait
