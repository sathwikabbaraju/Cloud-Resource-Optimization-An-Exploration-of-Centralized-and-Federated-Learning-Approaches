Last login: Sun Jun  1 13:40:36 on ttys000
(base) sathwik@SATHWIKs-MacBook-Air ~ % ls       
anaconda_projects		dockersf			MemGPT				opt				R
AndroidStudioProjects		Documents			MIMIC				package-lock.json		VISUAL STUDIO CODE
Applications			Downloads			Movies				package.json			VMshare
C.P.T-CN			flan-t5-large			Music				Pictures			XCODE
Cisco Packet Tracer 8.1.1	go				MY DOC				Postman
Cisco Packet Tracer 8.2.2	JNotebook			node_modules			Public
Desktop				Library				OneDrive			PycharmProjects
(base) sathwik@SATHWIKs-MacBook-Air ~ % cd VISUAL\ STUDIO\ CODE 
(base) sathwik@SATHWIKs-MacBook-Air VISUAL STUDIO CODE % cd Cloud:Fog 
(base) sathwik@SATHWIKs-MacBook-Air Cloud:Fog % cd aws\ data 
(base) sathwik@SATHWIKs-MacBook-Air aws data % ls
__pycache__			data_explorer.py		feature_engineered_client_data	model_data			processed_client_data
checkpoints			Delta01.ipynb			fl_model_utils.py		prepare_data_for_fl.py
cpu				feature_engineer.py		main_federated_learning.py	process_client_data.py
(base) sathwik@SATHWIKs-MacBook-Air aws data % conda activate pythonProject
(pythonProject) sathwik@SATHWIKs-MacBook-Air aws data % python run main_federated_learning.py
python: can't open file '/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/run': [Errno 2] No such file or directory
(pythonProject) sathwik@SATHWIKs-MacBook-Air aws data % python main_federated_learning.py
2025-06-01 13:43:50,708	INFO worker.py:1771 -- Started a local Ray instance.
Starting Flower simulation...
WARNING :   DEPRECATED FEATURE: flwr.simulation.start_simulation() is deprecated.
	Instead, use the `flwr run` CLI command to start a local simulation in your Flower app, as shown for example below:

		$ flwr new  # Create a new Flower app from a template

		$ flwr run  # Run the Flower app in Simulation Mode

	Using `start_simulation()` is deprecated.

            This is a deprecated feature. It will be removed
            entirely in future versions of Flower.
        
INFO :      Starting Flower simulation, config: num_rounds=3, no round_timeout
2025-06-01 13:43:53,892	INFO worker.py:1771 -- Started a local Ray instance.
INFO :      Flower VCE: Ray initialized with resources: {'CPU': 8.0, 'object_store_memory': 2147483648.0, 'node:127.0.0.1': 1.0, 'node:__internal_head__': 1.0, 'memory': 7598561690.0}
INFO :      Optimize your simulation with Flower VCE: https://flower.ai/docs/framework/how-to-run-simulations.html
INFO :      Flower VCE: Resources for each Virtual Client: {'num_cpus': 1, 'num_gpus': 0.0}
INFO :      Flower VCE: Creating VirtualClientEngineActorPool with 8 actors
INFO :      [INIT]
INFO :      Requesting initial parameters from one random client
INFO :      Received initial parameters from one random client
INFO :      Starting evaluation of initial global parameters
Server: Starting evaluation for round 0 on 12 client test sets.
(ClientAppActor pid=39482) WARNING :   DEPRECATED FEATURE: `client_fn` now expects a signature `def client_fn(context: Context)`.The provided `client_fn` has signature: {'cid': <Parameter "cid: str">}. You can import the `Context` like this: `from flwr.common import Context`
(ClientAppActor pid=39482) 
(ClientAppActor pid=39482)             This is a deprecated feature. It will be removed
(ClientAppActor pid=39482)             entirely in future versions of Flower.
(ClientAppActor pid=39482)         
(ClientAppActor pid=39482) Client 0011: Initializing FlowerClient object.
(ClientAppActor pid=39482) Client 0011: Using MPS (Apple Silicon GPU)
(ClientAppActor pid=39482) Client 0011: Model initialized on mps.
Server Evaluation: Loaded test data from client_0000_prepared_data.pkl (1133736 samples).
Server Evaluation: Finished with client_0000_prepared_data.pkl. Cache cleared.
Server Evaluation: Loaded test data from client_0001_prepared_data.pkl (1117707 samples).
Server Evaluation: Finished with client_0001_prepared_data.pkl. Cache cleared.
Server Evaluation: Loaded test data from client_0002_prepared_data.pkl (1048152 samples).
Server Evaluation: Finished with client_0002_prepared_data.pkl. Cache cleared.
Server Evaluation: Loaded test data from client_0003_prepared_data.pkl (1086346 samples).
Server Evaluation: Finished with client_0003_prepared_data.pkl. Cache cleared.
Server Evaluation: Loaded test data from client_0004_prepared_data.pkl (1070576 samples).
Server Evaluation: Finished with client_0004_prepared_data.pkl. Cache cleared.
Server Evaluation: Loaded test data from client_0005_prepared_data.pkl (1132349 samples).
Server Evaluation: Finished with client_0005_prepared_data.pkl. Cache cleared.
Server Evaluation: Loaded test data from client_0006_prepared_data.pkl (1156553 samples).
Server Evaluation: Finished with client_0006_prepared_data.pkl. Cache cleared.
Server Evaluation: Loaded test data from client_0007_prepared_data.pkl (1103882 samples).
Server Evaluation: Finished with client_0007_prepared_data.pkl. Cache cleared.
Server Evaluation: Loaded test data from client_0008_prepared_data.pkl (1118787 samples).
Server Evaluation: Finished with client_0008_prepared_data.pkl. Cache cleared.
Server Evaluation: Loaded test data from client_0009_prepared_data.pkl (1034186 samples).
Server Evaluation: Finished with client_0009_prepared_data.pkl. Cache cleared.
Server Evaluation: Loaded test data from client_0010_prepared_data.pkl (1136307 samples).
Server Evaluation: Finished with client_0010_prepared_data.pkl. Cache cleared.
Server Evaluation: Loaded test data from client_0011_prepared_data.pkl (608054 samples).
Server Evaluation: Finished with client_0011_prepared_data.pkl. Cache cleared.
Server: Round 0 validation loss: 0.0931
INFO :      initial parameters (loss, other metrics): 0.09312242385947163, {'server_loss': 0.09312242385947163}
INFO :      
INFO :      [ROUND 1]
INFO :      configure_fit: strategy sampled 1 clients (out of 12)
(ClientAppActor pid=39482) WARNING :   DEPRECATED FEATURE: `client_fn` now expects a signature `def client_fn(context: Context)`.The provided `client_fn` has signature: {'cid': <Parameter "cid: str">}. You can import the `Context` like this: `from flwr.common import Context`
(ClientAppActor pid=39482) 
(ClientAppActor pid=39482)             This is a deprecated feature. It will be removed
(ClientAppActor pid=39482)             entirely in future versions of Flower.
(ClientAppActor pid=39482)         
(ClientAppActor pid=39482) Client 0001: Initializing FlowerClient object.
(ClientAppActor pid=39482) Client 0001: Using MPS (Apple Silicon GPU)
(ClientAppActor pid=39482) Client 0001: Model initialized on mps.
(ClientAppActor pid=39482) Client 0001: Loading data for the first time...
(ClientAppActor pid=39482) Client 0001: Data loaded (Train: 5215963 samples, Val: 1117707 samples).
(ClientAppActor pid=39482) Client 0001: Starting local training for 1 epochs.