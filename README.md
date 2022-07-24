**Option 1 (preferred): Docker**

- Install Docker Engine (https://docs.docker.com/engine/install/#server) and Docker Compose (https://docs.docker.com/compose/install/). Be sure to be able to manage it as a non-root user (https://docs.docker.com/engine/install/linux-postinstall/).
- On the project folder, run in your terminal `docker-compose up`
- You should see something like:
```
Serving notebooks from local directory: /tesis
Jupyter Notebook 6.4.3 is running at: http://127.0.0.1:8888/?token=xxxxxxx
```

Ctrl+click on that last link should open up a browser tab with the project tree. Notebooks are in clasificacion-humedales/notebooks.


**Option 2: virtualenv**

- In the project folder, create a virtualenv, running in your terminal `python3 -m venv .venv`
- Activate it: `source .venv/bin/activate`
- Run `pip3 install -r requirements.txt` to install the requirements
- Run `pip3 install .` to install the project's packages and modules
- Run `jupyter notebook` to run the notebooks. - You should see something like:
```
Jupyter Notebook 6.4.3 is running at: http://127.0.0.1:8888/?token=xxxxxxx
```

Ctrl+click on that last link should open up a browser tab with the project tree. Notebooks are in clasificacion-humedales/notebooks.
