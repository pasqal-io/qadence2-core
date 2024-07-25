# Qadence 2

**Notice**: Qadence 2 is currently a *work in progress* and is under active development. Please be aware that the software is in an early stage, and frequent updates, including breaking changes, are to be expected. This means that:
* Features and functionalities may change without prior notice.
* The codebase is still evolving, and parts of the software may not function as intended.
* Documentation and user guides may be incomplete or subject to significant changes.


## Installation

*Note*: it is adviced to set up a python environment before installing the package.


To install the current version, there are a few options:

### Installation from PYPI

On the terminal, type

```bash
pip install qadence2
```

### Installation from Source

Clone this repository by typing on the terminal

```bash
git clone https://github.com/pasqal-io/qadence2-core.git
```

Go to `qadence2-core` folder and install it using [hatch](https://hatch.pypa.io/latest/)

```bash
hatch -v shell
```

## Contributing

Before making a contribution, please review our [code of conduct](docs/getting_started/CODE_OF_CONDUCT.md).

- **Submitting Issues:** To submit bug reports or feature requests, please use our [issue tracker](https://github.com/pasqal-io/qadence2-core/issues).
- **Developing in qadence:** To learn more about how to develop within `qadence`, please refer to [contributing guidelines](docs/getting_started/CONTRIBUTING.md).

### Setting up qadence in development mode

We recommend to use the [`hatch`](https://hatch.pypa.io/latest/) environment manager to install `qadence` from source:

```bash
python -m pip install hatch

# get into a shell with all the dependencies
python -m hatch shell

# run a command within the virtual environment with all the dependencies
python -m hatch run python my_script.py
```

## License

Qadence 2 is a free and open source software package, released under the [Apache License, Version 2.0](docs/getting_started/LICENSE.md).
