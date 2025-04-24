
Getting Started
===============

In this guide, we provide the instructions to setup and install the FRS codebase on your machine.
The goal is to set everything up in order to be able to run the example, based on the IEEE 13 network model, provided in the repository.

Requirements
------------

A set of dependencies must be installed by hand, please follow the instructions in their respective subsection

* :ref:`gridapps-d-install`
* :ref:`python38-install`
* :ref:`ipopt-install`
* :ref:`pipenv-install`

.. _gridapps-d-install:

GridAPPS-D:
^^^^^^^^^^^
The FRS relies on `GridAPPS-D <https://gridapps-d.org/>`_ for its simulation engine, and communication needs. 
You should start by installing GridAPPS-D on your system.

We recommend using the GridAPPS-D docker script available `online <https://github.com/GRIDAPPSD/gridappsd-docker>`_.
Provided that you have installed all the requirements you can start GridAPPS-D with the following commands:

.. code-block:: shell

    git clone https://github.com/GRIDAPPSD/gridappsd-docker
    cd gridappsd-docker
    ./run.sh

Now we are inside the executing container:

.. code-block:: shell

    root@737c30c82df7:/gridappsd# ./run-gridappsd.sh

GridAPPS-D is now running and you can access its interface at http://localhost:8080/

.. _python38-install:

Python 3.8:
^^^^^^^^^^^
The FRS code was developed in using Python 3.8.
We recommend you use this version and install it using `Anaconda <https://www.anaconda.com/download/>`_.

.. code-block:: shell

    conda create -n py38 python=3.8

.. _ipopt-install:

IPOPT:
^^^^^^
The FRS was developped using Pyomo with `IPOPT <https://github.com/coin-or/Ipopt>`_ as its main solver. 
We recommend installing IPOPT using Anaconda:

.. code-block:: shell

    conda install -c conda-forge ipopt

.. _pipenv-install:

Pipenv:
^^^^^^^
The remainder of the requirements for the FRS code to run can be installed through `pip <https://pypi.org/project/pip/>`_.
We used `Pipenv <https://pipenv.pypa.io/en/latest/>`_ to managed the virtual environment setup for the FRS. 

.. code-block:: shell

    pip install pipenv --user

Installation
------------

Source Code
^^^^^^^^^^^

The FRS source code is available at |frs_repo|. 

.. code-block:: shell
    :substitutions:

    git clone |frs_repo|
    cd frs-fastderms

The first step is to setup the virtual environment for running the project code:

.. code-block:: shell

    pipenv install

You can now use the newly created virtual environment as kernel in the Jupyter Notebooks provided in this repository.
Python scripts should be run with the following syntax to use the virtual environment:

.. code-block:: shell

    pipenv run my_script.py 

Data Folder
^^^^^^^^^^^

The repository is structured to store static data file in the ``/data`` folder located at the root of the repository.
While a few files are included in this repository, the majority is not as it would take too much space in the repository.
It is instead made available as an archive (|frs_shared_data_folder|) that we invite you to download and unzip at the following location.

.. code-block:: text

    frs-fastderms/
    ├- data/
    │ ├- CIM_files/
    │ └- Shared_Data_notinGIT/ # <-- Unzip content here
    │   └-... 
    └─...

Feeder Models
^^^^^^^^^^^^^

In the base implementation of the FRS, we use the grid simulator built into GridAPPS-D that uses an internal database of grid models (Blazegraph).
This means that you need to load your custom grid models into the database prior to running the FRS.
Please refer to the detailed guide (below) to load a new model into the GridAPPS-D grid model database.

.. dropdown:: Load Custom Model in GridAPPS-D

    This is a 2-step process:
    
    #. Load the XML file of the feeder into Blazegraph:
    
        .. code-block:: shell

            curl -s -D- -H 'Content-Type: application/xml' --upload-file {path_to_feeder_nodel.xml} -X POST 'http://localhost:8889/bigdata/namespace/kb/sparql'

    #. Add the measurement IDs into Blazegraph:

        .. code-block:: shell

            git clone https://github.com/GRIDAPPSD/CIMHub.git
            pip install SPARQLWrapper numpy pandas
            cd CIMHub
            python3 ../CIMHub/utils/ListFeeders.py #View the list of feeder models currently in the Blazegraph Database
            python3 ../CIMHub/utils/ListMeasureables.py cimhubconfig.json {name_of_feeder_model} {mRID associated with feedder_model} #Create the set of txt files containing the measurable objects in your new model using the ListMeasurables script:
            for f in `ls -1 *txt`; do python3 ../CIMHub/utils/InsertMeasurements.py cimhubconfig.json $f uuidfile.json; done #Insert the measurements into Blazegraph using the InsertMeasurements script. The measurement MRIDs will be saved into the file uuidfile.json

To simplify this step and get you started faster, we provide a Blazegraph image with a modified IEEE 13 feeder model specific to our example. 
After setting up the data folder in the previous Section, you can find the image in the provided demo data.

.. code-block:: text

    frs-fastderms/
    ├- data/
    │ ├- CIM_files/
    │ └- Shared_Data_notinGIT/ 
    │   ├- IEEE13_demo/
    │   │ ├- blazegraph_image/
    │   │ │ └- ieee13_demo_blazegraph.tar.gz   # <<-- This file
    │   │ └-... 
    │   └-... 
    └─...

In order to use this file, proceed to do the following:

#. Load the image into Docker:

    .. code-block:: shell
        
        cd frs-fastderms/data/SharedDatanotinGIT/IEEE13_demo/blazegraph_image/
        docker load < ./ieee13_demo_blazegraph.tar.gz

#. Amend the GridAPPS-D Docker file:

    .. code-block:: shell

        cd gridappsd-docker

    Open ``docker-compose.yml`` and edit the following lines:

    .. code-block:: yaml

        blazegraph:
            # Comment out the original line
            #image: gridappsd/blazegraph${GRIDAPPSD_TAG}
            # Replace with this new line
            image: gridappsd/blazegraph:ieee13_demo

Run an Example
--------------

You are now ready to run the :doc:`IEEE 13 example that is provided in the repository </examples/example_IEEE13>`. 

