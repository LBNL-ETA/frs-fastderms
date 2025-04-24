
Code Structure Overview
***********************

**class FAST_DERMS_CONTROL.orchestrator.Orchestrator(model_id,
IOmodule: IOmodule, **kw_args)**

   Main orchestrator class for the FAST-DERMS control system.

   This class coordinates the Day-Ahead (DA) and Model Predictive
   Control (MPC) operations for distributed energy resources. It
   handles initialization, model setup, DA scheduling, deterministic
   runs, and real-time MPC control.

   The orchestrator responds to various commands received through
   GridAPPS-D messaging system: - init: Initialize and setup the
   network model - init_red: Initialize and reduce the network model -
   DA: Run Day-Ahead scheduling - DET: Run deterministic analysis -
   export_models: Export DA and DET models - export_DA: Export DA
   process data - DEMO: Run complete demonstration sequence -
   hotstart: Import previously saved DA process - MPC: Initialize and
   run MPC

   Inherits from:
      fastderms_app: Base application class providing common
      functionality

   Attributes:
      IOs (IOmodule): Input/Output module for data handling handler
      (modelHandler): Network model handler DA (DayAhead): Day-ahead
      scheduling module builder (ScenarioBuilder): Scenario generation
      module

   **error()**

      Get the current error code of the orchestrator.

      Returns:
         int or bool: The error code value. Possible values:
            *  0 or False: No error

            *  1 or True: General error

            *  2: Controlled termination

   **export_DayAhead_Models(**kw_args)**

      Export Day-Ahead optimization models and their associated
      statistics.

      This method exports both the scenario-based (DA) and
      deterministic (DET) optimization models along with their solver
      statistics. The exported data includes the full Pyomo model data
      and solver performance metrics.

      Args:
         **kw_args: Keyword arguments including:
            *  filename (str): Name of the export file (default:
               ‘DayAhead_Models.pkl’)

            *  quit_after (bool): Flag to terminate orchestrator after
               export (default: True)

      The exported data dictionary contains:
         *  DA_model: Full Pyomo data for the scenario-based model (if
            available)

         *  DA_stats: Solver statistics for the DA model including:
               *  solver_status: Termination condition

               *  solver_time: Solution time

         *  DET_model: Full Pyomo data for the deterministic model (if
            available)

         *  DET_stats: Solver statistics for the DET model including:
               *  solver_status: Termination condition

               *  solver_time: Solution time

      Note:
         *  The method will only export models and statistics that are
            available

         *  Results are archived with a timestamp based on the case
            settings

         *  If quit_after is True, sets error_code to 2 for controlled
            termination

   **export_DayAhead_Process(**kw_args)**

      Export the complete Day-Ahead process data including handler,
      builder, and IO results.

      This method exports the current state of the Day-Ahead process,
      including: - Network model handler state - Scenario builder
      configuration - IO module results

      The exported data is saved to a pickle file with an optional
      custom filename and automatically archived with a timestamp.

      Args:
         **kw_args: Keyword arguments including:
            *  filename (str): Name of the export file (default:
               ‘DayAhead_Process.pkl’)

            *  quit_after (bool): Flag to terminate orchestrator after
               export (default: True)

      Note:
         *  The export includes the complete state of the handler,
            builder, and IO results

         *  The archive filename includes the case timestamp in
            YYYYMMDD format

         *  If quit_after is True, sets error_code to 2 for controlled
            termination

   **import_DayAhead_Process(**kw_args)**

      Import previously saved Day-Ahead process data from a file.

      This method loads and restores the state of the Day-Ahead
      process components, including the IO module results, model
      handler, and scenario builder from a previously exported file.

      Args:
         filename (str): Path to the file containing the saved
         Day-Ahead process data **kw_args: Additional keyword
         arguments (currently unused but maintained for future
         extensibility)

      Note:
         *  The method attempts to restore three main components:
               1. IO module results (self.IOs.results)

               2. Model handler (self.handler)

               3. Scenario builder (self.builder)

         *  For the model handler import, the mRID and Sbase values
            from the saved data are ignored in favor of the current
            instance’s values

         *  If any component is not found in the imported data, that
            component remains unchanged

   **init_MPC(**kw_args)**

      Initialize the Model Predictive Control (MPC) module with
      specified parameters.

      This method sets up the MPC module by: 1. Configuring
      time-related parameters (skip intervals, timesteps, periods) 2.
      Setting up the initial time and case settings 3. Initializing
      the MPC case in the model handler 4. Creating a new ModelPredict
      instance with specified parameters

      Args:
         **kw_args: Keyword arguments including:
            *  n_skip (int): Number of timesteps to skip between MPC
               runs (default: 1)

            *  n_timestep (int): Number of timesteps in MPC horizon
               (default: 24)

            *  timestep_period (int): Duration of each timestep in
               minutes (default: 15)

            *  t0 (datetime): Initial time for MPC (default: current
               case t0)

            *  n_thermal (int): Number of thermal constraints to
               consider (default: 20)

            *  E_BAT_final_flag (bool): Flag for battery final state
               constraint (default: False)

            *  E_EV_final_flag (bool): Flag for EV final state
               constraint (default: False)

            *  E_FL_final_flag (bool): Flag for flexible load final
               state constraint (default: False)

      Note:
         *  The MPC period is calculated as n_skip * timestep_period

         *  The MPC horizon is calculated as n_timestep *
            timestep_period

         *  All time-related parameters are logged at debug or info
            level

         *  The method creates a new MPCc.ModelPredict instance stored
            in self.MPC

   **on_message(headers, message)**

      Handle incoming messages from various topics in the GridAPPS-D
      platform.

      This method processes messages from three main topics: 1.
      Orchestrator topic: Handles control commands for model setup, DA
      scheduling, and MPC operations 2. Simulation output topic:
      Processes simulation timestamps and triggers MPC runs 3. ADMS
      topic: Processes network management system constraints and
      configurations

      The method supports the following commands on the orchestrator
      topic: - init: Initialize and setup the network model -
      init_red: Initialize and reduce the network model - DA: Run
      Day-Ahead scheduling - DET: Run deterministic analysis -
      export_models: Export DA and DET models - export_DA: Export DA
      process data - DEMO: Run complete demonstration sequence -
      hotstart: Import previously saved DA process - MPC: Initialize
      and run MPC


      Parameters
      ==========

      headers : dict
         Message headers containing metadata such as: - destination:
         Topic where the message was published - timestamp: Message
         timestamp in milliseconds

      message : dict
         Message payload containing command and parameter data.
         Structure varies by message type: - For orchestrator
         messages: Contains ‘command’ and command-specific parameters
         - For simulation output: Contains simulation timestamp - For
         ADMS messages: Contains network constraints and
         configurations


      Raises
      ======

      Exception
         Any unhandled exception during message processing will set
         the error code and re-raise the exception


      Notes
      =====

      *  MPC operations are triggered based on simulation timestamps
         and configured periods

      *  ADMS messages are only processed when MPC is running

      *  Error handling includes logging of line numbers and error
         details

   **process_ADMS_message(message)**

      Process messages received from the Advanced Distribution
      Management System (ADMS).

      This method handles two types of ADMS messages: 1. NMS
      constraints: Network Management System constraints for power
      limits 2. NMS switch: Network switch configuration updates

      For NMS constraints, the method: - Validates the feeder ID
      matches the current handler - Converts timestamps to MPC
      timesteps - Standardizes power units to Watts - Creates and
      publishes forecasts with the new constraints

      Args:
         message (dict): Message payload containing:
            *  datatype (str): Type of ADMS message (‘NMS constraints’
               or ‘NMS switch’)

            *  message (list): For NMS constraints, list of constraint
               dictionaries with:
                  *  Substation (str): Substation name

                  *  Circuit (str): Circuit identifier

                  *  Feeder ID (str): Feeder identifier

                  *  Upper Limit (str/int): Power upper limit

                  *  Lower Limit (str/int): Power lower limit

                  *  Units (str): Power units (‘W’, ‘kW’, or ‘MW’)

                  *  Start Time (str): Constraint start time

                  *  End Time (str): Constraint end time

      Note:
         *  Times are rounded to nearest MPC timestep (floor for
            start, ceil for end)

         *  All power values are converted to Watts internally

         *  Forecasts are published to MPC topic for the constraint
            duration

   **reduce_Model(**kw_args)**

      Reduce the complexity of the power system model by applying
      specified reduction rules.

      This method applies model reduction techniques to simplify the
      network model while preserving essential characteristics. It
      dynamically loads and applies reduction rules from the
      model_reduction module.

      Args:
         **kw_args: Keyword arguments including:
            *  rules (list[str]): List of reduction rule names to
               apply. Default is [‘remove_leaves’]. Each rule should
               correspond to a class name in the model_reduction
               module.

      Note:
         *  The method will skip any rules that cannot be found in the
            model_reduction module

         *  Rules are applied in the order they are specified in the
            rules list

         *  The model handler must be properly initialized before
            calling this method

      Raises:
         Any exceptions from the model reduction process are logged
         but not raised

   **run_DA(**kw_args)**

      Run the Day-Ahead (DA) scheduling optimization process.

      This method performs the following operations: 1. Updates
      network model and initializes case with specified parameters 2.
      Updates forecasts and generates scenarios 3. Builds and solves
      the DA optimization problem 4. Exports results and self-schedule
      bids

      Args:
         **kw_args: Keyword arguments including:
            *  n_timestep (int): Number of timesteps in the scheduling
               horizon (default: 24)

            *  timestep_period (int): Duration of each timestep in
               minutes (default: 60)

            *  n_scenario (int): Number of scenarios to generate
               (default: 10)

            *  t0 (float): Start time in epoch seconds (default:
               current time)

            *  remove_reserves (bool): Flag to exclude reserve
               requirements (default: False)

            *  n_init (int): Number of initial scenarios for selection
               (default: 1000)

            *  n_thermal (int): Number of thermal constraints to
               consider (default: 20)

      Note:
         The method temporarily deactivates thermal line limits during
         optimization. Results are exported if logging level is INFO
         or lower.

      Raises:
         Pyomo_Exception: If the scenario-based optimization problem
         fails to solve Exception: For other unexpected errors during
         execution

   **run_DET(**kw_args)**

      Run the Deterministic (DET) optimization process for Day-Ahead
      scheduling.

      This method performs a deterministic optimization run using
      expected values instead of scenarios. The process includes:
         1. Updating deterministic forecast data

         2. Building and solving the deterministic optimization model

         3. Exporting results and handling any solver failures

      Args:
         **kw_args: Keyword arguments including:
            *  n_thermal (int): Number of thermal constraints to
               consider (default: 20)

      Raises:
         Pyomo_Exception: If the deterministic optimization problem
         fails to solve Exception: For other unexpected errors during
         execution

      Note:
         *  If the solver fails, the method will fall back to using
            backup data

         *  Results are exported if logging level is INFO or lower

         *  Solver status and execution time are logged at WARNING
            level

   **run_MPC(t0, **kw_args)**

      Execute a Model Predictive Control (MPC) optimization step.

      This method performs a complete MPC optimization cycle
      including: 1. Updating the case time and initial energy states
      2. Applying reserve requirements and updating the MPC case 3.
      Building scenarios using expected values 4. Solving the MPC
      optimization problem 5. Exporting and publishing results

      Args:
         **kw_args: Keyword arguments including:
            *  t0 (datetime): Start time for this MPC step (default:
               previous t0 + skip period)

            *  opt_R_dist (dict): Reserve distribution parameters
               (default: default_opt_R_dist)
                  *  beta_load (float): Load reserve factor

                  *  beta_DER (float): DER reserve factor

                  *  R_sigma_up (float): Upward reserve sigma

                  *  R_sigma_dn (float): Downward reserve sigma

            *  sigma_multiplier (float): Multiplier for scenario
               standard deviation (default: 1)

      Raises:
         Pyomo_Exception: If the MPC optimization problem fails to
         solve Exception: For other unexpected errors during execution

      Note:
         *  Results are published to the MPC topic if successful

         *  Detailed results are archived if logging level is INFO or
            lower

         *  Initial energy states are exported regardless of
            success/failure

         *  Execution time is logged at WARNING level

   **running()**

      Check if the orchestrator is running without errors.

      Returns:
         bool: True if the orchestrator is running without errors,
         False otherwise. The status is determined by checking if
         there’s no error code set.

   **setup_Model(**kw_args)**

      Set up and initialize the power system model with specified
      parameters.

      This method performs the following operations: 1. Initializes
      the network model 2. Updates voltage limits 3. Converts network
      values to per-unit system

      Args:
         **kw_args: Keyword arguments including:
            *  Vmin (float): Minimum voltage limit in per unit
               (default: 0.9)

            *  Vmax (float): Maximum voltage limit in per unit
               (default: 1.1)

            *  force_static (bool): Force using static data (passed to
               initialize_model)

            *  Any other arguments accepted by
               handler.initialize_model()

      Note:
         The method uses the model handler (self.handler) to perform
         the actual model setup operations. The handler must be
         properly initialized before calling this method.
