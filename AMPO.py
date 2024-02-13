import os
import yaml
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import argparse
import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.rcParams['figure.facecolor'] = '1'
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams["legend.loc"] = 'lower right'

def main(rng):
    plt_path = '~/'
    for env in ["CartPole-v1", "Acrobot-v1"]:
        from utils.ampo import make_train
        fig, ax = plt.subplots()

        for phi, phi_inv, phi_math, lr_actor in zip([lambda x: x, lambda x: nn.relu((0.5 - 1) * x / 0.5) ** (1 / (0.5 - 1)), lambda x: nn.relu((1.5 - 1) * x / 1.5) ** (1 / (1.5 - 1))], [lambda x: x, lambda x: 0.5 * (x ** (0.5 - 1)) / (0.5 - 1), lambda x: 1.5 * (x ** (1.5 - 1)) / (1.5 - 1)], ["x$", "tsallis 0.5$", "tsallis 1.5$"], [1., 1., 1.]):
            with open('~/hyper_parameters/ampo/' + env + '.yaml', 'r') as file:
                config = yaml.safe_load(file)
            config["PHI"] = phi
            config["PHI_INV"] = phi_inv
            config["LR_ACTOR"] = lr_actor
        
            ### Run experiment
            rng, rng_ = jr.split(rng)
            rngs = jr.split(rng_, 100)
            train_fn = make_train(config)
            multi_train_fn = jax.jit(jax.vmap(train_fn))
            outputs = multi_train_fn(rngs)
            avg = outputs.mean(axis = 0)
                        
            ax.plot(avg[4:], label = "AMPO with $\\phi(x) = " + phi_math)
            

        ax.legend()
        ax.set_xlabel("Iteration $t$")
        ax.set_ylabel("$V^t$")
        ax.set_title("AMPO performance on " + env)
        fig.savefig(plt_path + env + '.pdf', bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)

    print("Done")

if __name__ == "__main__":
    rng = jr.PRNGKey(0)

    main(rng)