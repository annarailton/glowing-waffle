## Running this repo

Chose your favourite Python environment management system to make an environment (I like pyenv)

```bash
pyenv virtualenv 3.12 sentiment
pyenv activate sentiment
python -m pip install .
```

Problems with VSCode not using the correct environment when running the notebook can be resolved with 

```bash
python -m ipykernel install --user --name=sentiment
```