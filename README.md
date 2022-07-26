# pipenv-docker and M_simulation for simulated data prediction

This is a docker template for fast development with pipenv.
You can do this in the environment you already have after launching docker.
We hope you can experience simulated medical data analysis in the M_simulation directory!

## About M_simulation

After navigating to the M_simulation directory

python Learning.py command will execute training data creation.
M_simulation\learned_model\model The training data will be saved in the directory.
A diagram of the result of the EDA is shown in
M_simulation\learned_model\EDA_data The data will be saved in the directory
Adjusting the scr module to suit your case
It should contribute to feature selection

Using the learned model with the python Prediction.py command
The system performs predictions on the test data
If there is no data in the M_simulation_input_Test_data_input
If there is no data in M_simulation\inputTest_data_input
Move the input_data.csv in M_simulation_input to the Test_data_input directory and try
Please note that input_data.csv data used for forecasting will be deleted.
The predicted values will be stored in the M_simulationConversion output directory under the name Result.csv.

In actual operation, we would appreciate it if you could monitor the folder and move the data by shell or other means.

## About pipenv-docker

It is a real pain to rebuild the docker image after installing a new package with pipenv during development!
This template solves this problem by installing python packages after `docker-compose build` and caching them in the Docker Volume.

Since pipenv is tightly integrated with virtualenv, you need to enable virutalenv before running python scripts.
This template solves this problem by using `ENTRYPOINT` and `.bashrc` in the Dockerfile to hook the login and automatically log in to the pipenv environment just like a `pipenv shell`.

## For fast development

Dockerfile.dev`.

pipenv virtualenv to install pip packages and cache the environment in a Docker Volume.
Use [Entrykit](https://github.com/progrium/entrykit) to install the python environment in ENTRYPOINT.
The following will automatically log you into the pipenv environment

- docker-compose run --rm app bash`.
- docker-compose up -d && docker-compose exec app bash`.
- `command` in docker-compose.yml.

## For production

Dockerfile`.
pipenv install --system --deploy` to install pip packages to the global (system) environment.
Avoid redundant virtualization (Docker, virtualenv in pipenv).

## Launch Docker with the following command

docker compose up -d --build
docker container ls
docker-compose run --rm app bash

## LICENSE

MIT 

