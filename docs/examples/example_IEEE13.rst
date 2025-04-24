=======
IEEE 13
=======

Simulation Case
===============

We prepared a demonstration of the Flexible Resource Scheduler (FRS) based on a modified `IEEE 13 bus feeder <https://cmte.ieee.org/pes-testfeeders/resources/>`_. 
The simulation aims to demonstrate how the FRS interacts with a feeder that is comprised by regular loads, utility scale PV, utility scale Batteries, and aggregated behind the meter Batteries.

The simulation is set to take place on April 1st, 2022, between 14:00 and 18:00.
Load data are sourced from AMI data, and associated load forecast data were generated following the process outlined in [#this_paper]_.
Price data were sourced from an ISO market that provides both day-ahead and day-of prices.

.. image:: /images/IEEE13_example.*
    :alt: Modified IEEE 13 bus network model

Modified IEEE 13 bus network model for FRS demonstration.

Additional Resources:
---------------------

The following were added to the feeder: 

* Utility Scale PV:
    * PV1 at bus 646 with 225 kW
    * PV2 at bus 611 with 250 kW
    * PV3 at bus 671 with 375 kW
* Utility Scale Battery:
    * BAT1 at bus 692 with 700 kW (in green on Figure)
    * BAT2 at bus 634 with 150 kW (in blue on Figure)
    * BAT3 at bus 646 with 140 kW (in black on Figure)
* Aggregated behind the meter Batteries:
    * BAT4 at buses 611, 645, and 675 with 240 kW (in grey on Figure)
    * BAT5 at buses 611, 646, 652, 675, and 692 with 170 kW (in light brown on Figure)

.. [#this_paper] `10.1109/NAPS61145.2024.10741734 <https://doi.org/10.1109/NAPS61145.2024.10741734>`_

Prerequisites
=============
Prior to starting the FRS demonstration you should make sure that you have followed all the steps outlined on the :doc:`Getting Started page </getting_started>`.
The data that you downloaded and unzipped should reflect the following structure:

.. code-block:: text

    frs-fastderms/
    ├- data/
    │ ├- CIM_files/
    │ └- Shared_Data_notinGIT/ 
    │   ├- IEEE13_demo/
    │   │ ├- blazegraph_image/
    │   │ │ └- ieee13_demo_blazegraph.tar.gz
    │   │ ├- forecasts/
    │   │ │ ├- load/
    │   │ │ │ ├- mu_39node.csv
    │   │ │ │ ├- node_sequence.csv
    │   │ │ │ └- sigma_39node.csv
    │   │ │ └- pv/
    │   │ │   ├- mu_of_30_PVs.csv
    │   │ │   └- sigma_of_30_PVs.csv
    │   │ ├- ieeezipload.player
    │   │ └- weather_data.csv
    │   └-... 
    └─...

Folder Structure
================

All the files necessary to run the demonstrations are gathered in the ``/demos`` folder, as follows:

.. code-block:: text

 frs-fastderms/
    ├- demos/
    │ ├- FULL_DEMO.py
    │ └- IEEE13/
    │   ├- archive/
    │   ├- inputs/
    │   │ ├- MS_IEEE13_demo.xlsx
    │   │ ├- sim_config_IEEE13.json
    │   │ └- sim_config_IEEE13_ochre.json
    │   ├- logs/
    │   ├- DEMO.py
    │   ├- Results_Analysis.ipynb
    │   └- SETTINGS_Config.ipynb
    └─...

Each files / folder has the following purpose:

* ``FULL_DEMO.py``: Script that orchestrate the entire demonstration. It allows the user to select the options available for running the demonstration. More details about this file are :ref:`available here <full_demo.py>`
* ``DEMO.py``: Script that tailors ``FULL_DEMO.py`` to our IEEE 13 case. It will be the entrypoint for running the demonstration.
* ``SETTINGS_Config.ipynb``: Notebook used to generate all the input data and files to run the demonstration. 
* ``Results_Analysis.ipynb``: Notebook used to process the simulation output and prepare a set of graphs for analysing the results.
* ``MS_IEEE13_demo.xlsx``: OCHRE configuration file (also referred to as `Master Spreadsheet`)
* ``sim_config_IEEE13{_ochre}.json``: GridAPPS-D simulation configuration file (w/wo OCHRE)
* ``./logs``: folder containing all the temporary files of the simulation, including all the logs files
* ``./archive``: export folder for the experiment data to preserve and carry out result analysis


Note that the folders ``./archive`` and ``./logs`` are initially not present upon cloning the repository, but will be automatically created when running the ``DEMO.py`` script.

Running the demonstration
=========================

Follow these steps to start the simulation.

GridAPPS-D:
-----------

#. Spin up the containers:
    The FRS was developed using GridAPPS-D ``v2023.09.0``.

    .. code-block:: shell

        cd gridappsd-docker
        ./run.sh -t v2023.09.0
#. Start the platform from within the gridappsd container:
    
    .. code-block:: shell

        root@737c30c82df7:/gridappsd# ./run-gridappsd.sh

Simulation settings and input data files
----------------------------------------

A set of input files need to be generated prior to the first simulation run.
These files include all the static and forecast data for the chosen case, and also the configuration file for the FRS. 

* Open ``SETTINGS_Config.ipynb`` and run all the cells. 

Note that you will only need to do this step once, unless you intend to change the data in the ``SETTINGS_Config.ipynb`` notebook.

Simulation
----------

In a new terminal window, run the ``DEMO.py`` script:

.. code-block:: shell

    cd frs-fastderms
    pipenv run python ./demos/IEEE13/DEMO.py

The ``DEMO.py`` uses the same input arguments as defined in ``FULL_DEMO.py``: 

* ``-da`` / ``-nda``: Run the day ahead simulation / Do NOT run the day ahead simulation
* ``-s`` / ``-ns``: Run the day-of simulation / DO NOT run the day-of simulation
* ``-b`` / ``-nb``: Run the baselines / DO NOT run the baselines:
    This command triggers two different runs where:
    
    * All DERs are set to output no power, which enables the recording of the background load
    * All PVs are not curtailled, other DERs are set to output no power, which enables the recording of the full PV potential.
* ``-c [follow by case number (int)]``: Provide the case number to run. In this example we have the following cases:
    * case 1 is the regular case with Batteries and PV
    * case 2 includes the TMM
    * case 3 includes the ADMS message parsing
    * case 4 includes *both* TMM and ADMS message parsing

.. dropdown:: Running the demo as a background process

    The demo script can be run as a background process, which is especially important when running the full simulation (day ahead, simulation, and baselines), since the process will take nearly 15h to complete. 
    To achieve this, you can use the ``screen`` command:

    .. code-block:: shell

        screen -d -m -S myieee13_demo pipenv run python ./demos/IEEE/DEMO.py -da -s -b

When the simulation has completed, you can run the ``Results_Analysis.ipynb`` notebook to open the results and proceed to a quick analysis through a couple plots.