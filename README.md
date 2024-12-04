# Genetic Algorithm-based optimization of Dynamical Decoupling (GADD)

This is the companion repository to the paper [Empirical learning of dynamical decoupling on quantum processors](https://arxiv.org/abs/2403.02294). You can use this code to train on physical processors then run target circuits with sequences found via GADD.

## Installation

The package can be installed by running `pip install .` in the root directory of the package. If you would like to make changes to the package, you should run `pip install -e .` instead to install it in editable mode.

## Usage

This package is designed to be used on top of [Qiskit](https://github.com/Qiskit/qiskit) and the [Qiskit Runtime IBM Client](https://github.com/Qiskit/qiskit-ibm-runtime).

The core class `GADD` runs the genetic algorithm training process
on training circuits, outputting the best sequence and intermediate training data, if desired, which can be then used to run on a target circuit:

```
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=n_qubits)

with Batch(backend=backend):
    sampler = SamplerV2()
    gadd = GADD(backend=backend)

    # train
    [seq, data] = gadd.train(backend=backend,
        sampler=sampler,
        training_circuit=training_circuit,
        utility_function=utility_function,
        save_iterations=True,
        comparison_seqs=["baseline", "xy4", "cpmg","edd"])

    # visualize the training progression
    gadd.plot(seq, data)

    # run on target circuit (can run on a different backend)
    gadd.run(
        seq = seq,
        target_circuit=target_circuit,
        sampler=sampler
    )
```

