# Inverse-Pendulum-Genetic

While the inverse pendulum problem has been solved before by using genetic algorithms to train neural network weights, I have encountered some problems.

# Problems

- The pendulum is only able to balance itself if initialized in an upright position
- The pendulum is unable to upright itself when it falls over
- When at a downright position the pendulum moves as much as it can to one side, instead of swinging itself and gaining momentum to reach the upright position
# Currently

- The pendulum is able to maintain stability when initialized in an upright position

# Possible Solutions

- The most obvious problem is the unoptimized fitness function. The pendulum is being rewarded based on how close it is to the desired setpoint. This causes the pendulum to be penalized for trying to swing itself back and forth to gain momentum. The fitness function could be optimized to reward the pendulum for being above a certain threshold.
