(base) sathwik@SATHWIKs-MacBook-Air aws data % conda activate pythonProject
(pythonProject) sathwik@SATHWIKs-MacBook-Air aws data % python main_federated_learning.py
Starting Flower simulation...
WARNING :   DEPRECATED FEATURE: flwr.simulation.start_simulation() is deprecated.
        Instead, use the `flwr run` CLI command to start a local simulation in your Flower app, as shown for example below:

                $ flwr new  # Create a new Flower app from a template

                $ flwr run  # Run the Flower app in Simulation Mode

        Using `start_simulation()` is deprecated.

            This is a deprecated feature. It will be removed
            entirely in future versions of Flower.
        
INFO :      Starting Flower simulation, config: num_rounds=3, no round_timeout
2025-06-01 09:49:52,958 INFO worker.py:1771 -- Started a local Ray instance.
INFO :      Flower VCE: Ray initialized with resources: {'CPU': 8.0, 'memory': 5330511463.0, 'node:127.0.0.1': 1.0, 'node:__internal_head__': 1.0, 'object_store_memory': 2147483648.0}
INFO :      Optimize your simulation with Flower VCE: https://flower.ai/docs/framework/how-to-run-simulations.html
INFO :      Flower VCE: Resources for each Virtual Client: {'num_cpus': 1, 'num_gpus': 0.0}
INFO :      Flower VCE: Creating VirtualClientEngineActorPool with 8 actors
INFO :      [INIT]
INFO :      Requesting initial parameters from one random client
INFO :      Received initial parameters from one random client
INFO :      Starting evaluation of initial global parameters
Server: Starting evaluation for round 0 on 12 client test sets.
(ClientAppActor pid=2421) Client 0006: Initializing FlowerClient object.
(ClientAppActor pid=2421) Client 0006: Using MPS (Apple Silicon GPU)
(ClientAppActor pid=2421) Client 0006: Model initialized on mps.
(ClientAppActor pid=2421) WARNING :   DEPRECATED FEATURE: `client_fn` now expects a signature `def client_fn(context: Context)`.The provided `client_fn` has signature: {'cid': <Parameter "cid: str">}. You can import the `Context` like this: `from flwr.common import Context`
(ClientAppActor pid=2421) 
(ClientAppActor pid=2421)             This is a deprecated feature. It will be removed
(ClientAppActor pid=2421)             entirely in future versions of Flower.
(ClientAppActor pid=2421)         
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
Server: Round 0 validation loss: 0.0372
INFO :      initial parameters (loss, other metrics): 0.03717757017720973, {'server_loss': 0.03717757017720973}
INFO :      
INFO :      [ROUND 1]
INFO :      configure_fit: strategy sampled 1 clients (out of 12)
(ClientAppActor pid=2421) Client 0000: Initializing FlowerClient object.
(ClientAppActor pid=2421) Client 0000: Using MPS (Apple Silicon GPU)
(ClientAppActor pid=2421) Client 0000: Model initialized on mps.
(ClientAppActor pid=2421) Client 0000: Loading data for the first time...
(ClientAppActor pid=2421) WARNING :   DEPRECATED FEATURE: `client_fn` now expects a signature `def client_fn(context: Context)`.The provided `client_fn` has signature: {'cid': <Parameter "cid: str">}. You can import the `Context` like this: `from flwr.common import Context`
(ClientAppActor pid=2421) 
(ClientAppActor pid=2421)             This is a deprecated feature. It will be removed
(ClientAppActor pid=2421)             entirely in future versions of Flower.
(ClientAppActor pid=2421)         
(ClientAppActor pid=2421) Client 0000: Data loaded (Train: 5290768 samples, Val: 1133736 samples).
(ClientAppActor pid=2421) Client 0000: Starting local training for 1 epochs.
ERROR :     Traceback (most recent call last):
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_client_proxy.py", line 95, in _submit_job
    out_mssg, updated_context = self.actor_pool.get_client_result(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 401, in get_client_result
    return self._fetch_future_result(cid)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 282, in _fetch_future_result
    res_cid, out_mssg, updated_context = ray.get(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/worker.py", line 2639, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/worker.py", line 864, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ClientAppException): ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 144, in __call__
    return self._call(message, context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 128, in ffn
    out_message = handle_legacy_message_from_msgtype(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/message_handler/message_handler.py", line 128, in handle_legacy_message_from_msgtype
    fit_res = maybe_call_fit(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client.py", line 224, in maybe_call_fit
    return client.fit(fit_ins)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/numpy_client.py", line 227, in _fit
    results = self.numpy_client.fit(parameters, ins.config)  # type: ignore
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 122, in fit
    loss.backward()
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/_tensor.py", line 626, in backward
    torch.autograd.backward(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/autograd/__init__.py", line 347, in backward
    _engine_run_backward(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/autograd/graph.py", line 823, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: MPS backend out of memory (MPS allocated: 40.00 MB, other allocations: 18.10 GB, max allowed: 18.13 GB). Tried to allocate 512 bytes on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 40.00 MB, other allocations: 18.10 GB, max allowed: 18.13 GB). Tried to allocate 512 bytes on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

ERROR :     ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 144, in __call__
    return self._call(message, context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 128, in ffn
    out_message = handle_legacy_message_from_msgtype(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/message_handler/message_handler.py", line 128, in handle_legacy_message_from_msgtype
    fit_res = maybe_call_fit(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client.py", line 224, in maybe_call_fit
    return client.fit(fit_ins)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/numpy_client.py", line 227, in _fit
    results = self.numpy_client.fit(parameters, ins.config)  # type: ignore
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 122, in fit
    loss.backward()
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/_tensor.py", line 626, in backward
    torch.autograd.backward(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/autograd/__init__.py", line 347, in backward
    _engine_run_backward(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/autograd/graph.py", line 823, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: MPS backend out of memory (MPS allocated: 40.00 MB, other allocations: 18.10 GB, max allowed: 18.13 GB). Tried to allocate 512 bytes on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 40.00 MB, other allocations: 18.10 GB, max allowed: 18.13 GB). Tried to allocate 512 bytes on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).
INFO :      aggregate_fit: received 0 results and 1 failures
Server: Aggregating fit results for round 1. 0 clients returned results.
Server: Aggregation and checkpointing for round 1 completed.
Server: Starting evaluation for round 1 on 12 client test sets.
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
Server: Round 1 validation loss: 0.0372
INFO :      fit progress: (1, 0.03717757017720973, {'server_loss': 0.03717757017720973}, 2904.9006310000004)
INFO :      configure_evaluate: strategy sampled 1 clients (out of 12)
(ClientAppActor pid=2421) Client 0004: Initializing FlowerClient object.
(ClientAppActor pid=2421) Client 0004: Using MPS (Apple Silicon GPU)
(ClientAppActor pid=2421) Client 0004: Model initialized on mps.
(ClientAppActor pid=2421) Client 0004: Loading data for the first time...
(ClientAppActor pid=2421) WARNING :   DEPRECATED FEATURE: `client_fn` now expects a signature `def client_fn(context: Context)`.The provided `client_fn` has signature: {'cid': <Parameter "cid: str">}. You can import the `Context` like this: `from flwr.common import Context`
(ClientAppActor pid=2421) 
(ClientAppActor pid=2421)             This is a deprecated feature. It will be removed
(ClientAppActor pid=2421)             entirely in future versions of Flower.
(ClientAppActor pid=2421)         
(ClientAppActor pid=2421) Client 0004: Data loaded (Train: 4996017 samples, Val: 1070575 samples).
(ClientAppActor pid=2421) Client 0004: Starting local evaluation.
ERROR :     Traceback (most recent call last):
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_client_proxy.py", line 95, in _submit_job
    out_mssg, updated_context = self.actor_pool.get_client_result(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 401, in get_client_result
    return self._fetch_future_result(cid)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 282, in _fetch_future_result
    res_cid, out_mssg, updated_context = ray.get(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/worker.py", line 2639, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/worker.py", line 864, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ClientAppException): ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 144, in __call__
    return self._call(message, context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 128, in ffn
    out_message = handle_legacy_message_from_msgtype(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/message_handler/message_handler.py", line 135, in handle_legacy_message_from_msgtype
    evaluate_res = maybe_call_evaluate(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client.py", line 244, in maybe_call_evaluate
    return client.evaluate(evaluate_ins)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/numpy_client.py", line 251, in _evaluate
    results = self.numpy_client.evaluate(parameters, ins.config)  # type: ignore
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 157, in evaluate
    outputs = self.model(X_batch)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/fl_model_utils.py", line 42, in forward
    out, (hn, cn) = self.lstm(x, (h0, c0))
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 1124, in forward
    result = _VF.lstm(
RuntimeError: MPS backend out of memory (MPS allocated: 16.00 MB, other allocations: 18.12 GB, max allowed: 18.13 GB). Tried to allocate 256 bytes on shared pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 16.00 MB, other allocations: 18.12 GB, max allowed: 18.13 GB). Tried to allocate 256 bytes on shared pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

ERROR :     ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 144, in __call__
    return self._call(message, context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 128, in ffn
    out_message = handle_legacy_message_from_msgtype(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/message_handler/message_handler.py", line 135, in handle_legacy_message_from_msgtype
    evaluate_res = maybe_call_evaluate(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client.py", line 244, in maybe_call_evaluate
    return client.evaluate(evaluate_ins)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/numpy_client.py", line 251, in _evaluate
    results = self.numpy_client.evaluate(parameters, ins.config)  # type: ignore
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 157, in evaluate
    outputs = self.model(X_batch)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/fl_model_utils.py", line 42, in forward
    out, (hn, cn) = self.lstm(x, (h0, c0))
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 1124, in forward
    result = _VF.lstm(
RuntimeError: MPS backend out of memory (MPS allocated: 16.00 MB, other allocations: 18.12 GB, max allowed: 18.13 GB). Tried to allocate 256 bytes on shared pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 16.00 MB, other allocations: 18.12 GB, max allowed: 18.13 GB). Tried to allocate 256 bytes on shared pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).
INFO :      aggregate_evaluate: received 0 results and 1 failures
INFO :      
INFO :      [ROUND 2]
INFO :      configure_fit: strategy sampled 1 clients (out of 12)
ERROR :     Traceback (most recent call last):
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_client_proxy.py", line 95, in _submit_job
    out_mssg, updated_context = self.actor_pool.get_client_result(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 401, in get_client_result
    return self._fetch_future_result(cid)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 282, in _fetch_future_result
    res_cid, out_mssg, updated_context = ray.get(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/worker.py", line 2639, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/worker.py", line 864, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ClientAppException): ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 144, in __call__
    return self._call(message, context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 128, in ffn
    out_message = handle_legacy_message_from_msgtype(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/message_handler/message_handler.py", line 96, in handle_legacy_message_from_msgtype
    client = client_fn(context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 72, in adaptor_fn
    return client_fn(str(cid))  # type: ignore
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 184, in client_fn
    return FlowerClient(formatted_cid).to_client()
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 63, in __init__
    self.model.to(self.device)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1343, in to
    return self._apply(convert)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 285, in _apply
    ret = super()._apply(fn, recurse)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 930, in _apply
    param_applied = fn(param)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1329, in convert
    return t.to(
RuntimeError: MPS backend out of memory (MPS allocated: 8.00 MB, other allocations: 18.13 GB, max allowed: 18.13 GB). Tried to allocate 16.00 KB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 8.00 MB, other allocations: 18.13 GB, max allowed: 18.13 GB). Tried to allocate 16.00 KB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

ERROR :     ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 144, in __call__
    return self._call(message, context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 128, in ffn
    out_message = handle_legacy_message_from_msgtype(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/message_handler/message_handler.py", line 96, in handle_legacy_message_from_msgtype
    client = client_fn(context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 72, in adaptor_fn
    return client_fn(str(cid))  # type: ignore
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 184, in client_fn
    return FlowerClient(formatted_cid).to_client()
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 63, in __init__
    self.model.to(self.device)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1343, in to
    return self._apply(convert)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 285, in _apply
    ret = super()._apply(fn, recurse)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 930, in _apply
    param_applied = fn(param)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1329, in convert
    return t.to(
RuntimeError: MPS backend out of memory (MPS allocated: 8.00 MB, other allocations: 18.13 GB, max allowed: 18.13 GB). Tried to allocate 16.00 KB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 8.00 MB, other allocations: 18.13 GB, max allowed: 18.13 GB). Tried to allocate 16.00 KB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).
INFO :      aggregate_fit: received 0 results and 1 failures
Server: Aggregating fit results for round 2. 0 clients returned results.
Server: Aggregation and checkpointing for round 2 completed.
Server: Starting evaluation for round 2 on 12 client test sets.
(ClientAppActor pid=2421) Client 0001: Initializing FlowerClient object.
(ClientAppActor pid=2421) Client 0001: Using MPS (Apple Silicon GPU)
(ClientAppActor pid=2421) WARNING :   DEPRECATED FEATURE: `client_fn` now expects a signature `def client_fn(context: Context)`.The provided `client_fn` has signature: {'cid': <Parameter "cid: str">}. You can import the `Context` like this: `from flwr.common import Context`
(ClientAppActor pid=2421) 
(ClientAppActor pid=2421)             This is a deprecated feature. It will be removed
(ClientAppActor pid=2421)             entirely in future versions of Flower.
(ClientAppActor pid=2421)         
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
Server: Round 2 validation loss: 0.0372
INFO :      fit progress: (2, 0.03717757017720973, {'server_loss': 0.03717757017720973}, 4207.9329382079995)
INFO :      configure_evaluate: strategy sampled 1 clients (out of 12)
ERROR :     Traceback (most recent call last):
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_client_proxy.py", line 95, in _submit_job
    out_mssg, updated_context = self.actor_pool.get_client_result(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 401, in get_client_result
    return self._fetch_future_result(cid)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 282, in _fetch_future_result
    res_cid, out_mssg, updated_context = ray.get(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/worker.py", line 2639, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/worker.py", line 864, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ClientAppException): ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 144, in __call__
    return self._call(message, context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 128, in ffn
    out_message = handle_legacy_message_from_msgtype(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/message_handler/message_handler.py", line 96, in handle_legacy_message_from_msgtype
    client = client_fn(context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 72, in adaptor_fn
    return client_fn(str(cid))  # type: ignore
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 184, in client_fn
    return FlowerClient(formatted_cid).to_client()
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 63, in __init__
    self.model.to(self.device)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1343, in to
    return self._apply(convert)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 285, in _apply
    ret = super()._apply(fn, recurse)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 930, in _apply
    param_applied = fn(param)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1329, in convert
    return t.to(
RuntimeError: MPS backend out of memory (MPS allocated: 8.00 MB, other allocations: 18.13 GB, max allowed: 18.13 GB). Tried to allocate 16.00 KB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 8.00 MB, other allocations: 18.13 GB, max allowed: 18.13 GB). Tried to allocate 16.00 KB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

ERROR :     ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 144, in __call__
    return self._call(message, context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 128, in ffn
    out_message = handle_legacy_message_from_msgtype(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/message_handler/message_handler.py", line 96, in handle_legacy_message_from_msgtype
    client = client_fn(context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 72, in adaptor_fn
    return client_fn(str(cid))  # type: ignore
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 184, in client_fn
    return FlowerClient(formatted_cid).to_client()
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 63, in __init__
    self.model.to(self.device)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1343, in to
    return self._apply(convert)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 285, in _apply
    ret = super()._apply(fn, recurse)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 930, in _apply
    param_applied = fn(param)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1329, in convert
    return t.to(
RuntimeError: MPS backend out of memory (MPS allocated: 8.00 MB, other allocations: 18.13 GB, max allowed: 18.13 GB). Tried to allocate 16.00 KB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 8.00 MB, other allocations: 18.13 GB, max allowed: 18.13 GB). Tried to allocate 16.00 KB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).
INFO :      aggregate_evaluate: received 0 results and 1 failures
INFO :      
INFO :      [ROUND 3]
INFO :      configure_fit: strategy sampled 1 clients (out of 12)
ERROR :     Traceback (most recent call last):
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_client_proxy.py", line 95, in _submit_job
    out_mssg, updated_context = self.actor_pool.get_client_result(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 401, in get_client_result
    return self._fetch_future_result(cid)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 282, in _fetch_future_result
    res_cid, out_mssg, updated_context = ray.get(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/worker.py", line 2639, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/worker.py", line 864, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ClientAppException): ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 144, in __call__
    return self._call(message, context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 128, in ffn
    out_message = handle_legacy_message_from_msgtype(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/message_handler/message_handler.py", line 96, in handle_legacy_message_from_msgtype
    client = client_fn(context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 72, in adaptor_fn
    return client_fn(str(cid))  # type: ignore
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 184, in client_fn
    return FlowerClient(formatted_cid).to_client()
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 63, in __init__
    self.model.to(self.device)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1343, in to
    return self._apply(convert)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 285, in _apply
    ret = super()._apply(fn, recurse)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 930, in _apply
    param_applied = fn(param)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1329, in convert
    return t.to(
RuntimeError: MPS backend out of memory (MPS allocated: 8.00 MB, other allocations: 18.13 GB, max allowed: 18.13 GB). Tried to allocate 16.00 KB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 8.00 MB, other allocations: 18.13 GB, max allowed: 18.13 GB). Tried to allocate 16.00 KB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

ERROR :     ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 144, in __call__
    return self._call(message, context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 128, in ffn
    out_message = handle_legacy_message_from_msgtype(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/message_handler/message_handler.py", line 96, in handle_legacy_message_from_msgtype
    client = client_fn(context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 72, in adaptor_fn
    return client_fn(str(cid))  # type: ignore
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 184, in client_fn
    return FlowerClient(formatted_cid).to_client()
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 63, in __init__
    self.model.to(self.device)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1343, in to
    return self._apply(convert)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 285, in _apply
    ret = super()._apply(fn, recurse)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 930, in _apply
    param_applied = fn(param)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1329, in convert
    return t.to(
RuntimeError: MPS backend out of memory (MPS allocated: 8.00 MB, other allocations: 18.13 GB, max allowed: 18.13 GB). Tried to allocate 16.00 KB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 8.00 MB, other allocations: 18.13 GB, max allowed: 18.13 GB). Tried to allocate 16.00 KB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).
INFO :      aggregate_fit: received 0 results and 1 failures
Server: Aggregating fit results for round 3. 0 clients returned results.
Server: Aggregation and checkpointing for round 3 completed.
(ClientAppActor pid=2421) Client 0005: Initializing FlowerClient object.
(ClientAppActor pid=2421) Client 0005: Using MPS (Apple Silicon GPU)
(ClientAppActor pid=2421) Client 0003: Initializing FlowerClient object.
(ClientAppActor pid=2421) Client 0003: Using MPS (Apple Silicon GPU)
Server: Starting evaluation for round 3 on 12 client test sets.
(ClientAppActor pid=2421) WARNING :   DEPRECATED FEATURE: `client_fn` now expects a signature `def client_fn(context: Context)`.The provided `client_fn` has signature: {'cid': <Parameter "cid: str">}. You can import the `Context` like this: `from flwr.common import Context`
(ClientAppActor pid=2421) 
(ClientAppActor pid=2421)             This is a deprecated feature. It will be removed
(ClientAppActor pid=2421)             entirely in future versions of Flower.
(ClientAppActor pid=2421)         
(ClientAppActor pid=2421) WARNING :   DEPRECATED FEATURE: `client_fn` now expects a signature `def client_fn(context: Context)`.The provided `client_fn` has signature: {'cid': <Parameter "cid: str">}. You can import the `Context` like this: `from flwr.common import Context`
(ClientAppActor pid=2421) 
(ClientAppActor pid=2421)             This is a deprecated feature. It will be removed
(ClientAppActor pid=2421)             entirely in future versions of Flower.
(ClientAppActor pid=2421)         
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
Server: Round 3 validation loss: 0.0372
INFO :      fit progress: (3, 0.03717757017720973, {'server_loss': 0.03717757017720973}, 5514.363342958)
INFO :      configure_evaluate: strategy sampled 1 clients (out of 12)
ERROR :     Traceback (most recent call last):
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_client_proxy.py", line 95, in _submit_job
    out_mssg, updated_context = self.actor_pool.get_client_result(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 401, in get_client_result
    return self._fetch_future_result(cid)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 282, in _fetch_future_result
    res_cid, out_mssg, updated_context = ray.get(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/worker.py", line 2639, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/ray/_private/worker.py", line 864, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ClientAppException): ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 144, in __call__
    return self._call(message, context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 128, in ffn
    out_message = handle_legacy_message_from_msgtype(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/message_handler/message_handler.py", line 96, in handle_legacy_message_from_msgtype
    client = client_fn(context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 72, in adaptor_fn
    return client_fn(str(cid))  # type: ignore
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 184, in client_fn
    return FlowerClient(formatted_cid).to_client()
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 63, in __init__
    self.model.to(self.device)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1343, in to
    return self._apply(convert)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 285, in _apply
    ret = super()._apply(fn, recurse)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 930, in _apply
    param_applied = fn(param)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1329, in convert
    return t.to(
RuntimeError: MPS backend out of memory (MPS allocated: 8.00 MB, other allocations: 18.13 GB, max allowed: 18.13 GB). Tried to allocate 16.00 KB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 8.00 MB, other allocations: 18.13 GB, max allowed: 18.13 GB). Tried to allocate 16.00 KB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

ERROR :     ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 144, in __call__
    return self._call(message, context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 128, in ffn
    out_message = handle_legacy_message_from_msgtype(
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/message_handler/message_handler.py", line 96, in handle_legacy_message_from_msgtype
    client = client_fn(context)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/client/client_app.py", line 72, in adaptor_fn
    return client_fn(str(cid))  # type: ignore
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 184, in client_fn
    return FlowerClient(formatted_cid).to_client()
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 63, in __init__
    self.model.to(self.device)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1343, in to
    return self._apply(convert)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 903, in _apply
    module._apply(fn)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 285, in _apply
    ret = super()._apply(fn, recurse)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 930, in _apply
    param_applied = fn(param)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1329, in convert
    return t.to(
RuntimeError: MPS backend out of memory (MPS allocated: 8.00 MB, other allocations: 18.13 GB, max allowed: 18.13 GB). Tried to allocate 16.00 KB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=2421, ip=127.0.0.1, actor_id=657e9117391d337e6a5c9bac01000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x1064031c0>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 8.00 MB, other allocations: 18.13 GB, max allowed: 18.13 GB). Tried to allocate 16.00 KB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).
INFO :      aggregate_evaluate: received 0 results and 1 failures
INFO :      
INFO :      [SUMMARY]
INFO :      Run finished 3 round(s) in 5514.38s
INFO :          History (loss, centralized):
INFO :                  round 0: 0.03717757017720973
INFO :                  round 1: 0.03717757017720973
INFO :                  round 2: 0.03717757017720973
INFO :                  round 3: 0.03717757017720973
INFO :          History (metrics, centralized):
INFO :          {'server_loss': [(0, 0.03717757017720973),
INFO :                           (1, 0.03717757017720973),
INFO :                           (2, 0.03717757017720973),
INFO :                           (3, 0.03717757017720973)]}
INFO :      

Flower simulation finished.
(ClientAppActor pid=2421) Client 0007: Initializing FlowerClient object.
(ClientAppActor pid=2421) Client 0007: Using MPS (Apple Silicon GPU)
(ClientAppActor pid=2421) WARNING :   DEPRECATED FEATURE: `client_fn` now expects a signature `def client_fn(context: Context)`.The provided `client_fn` has signature: {'cid': <Parameter "cid: str">}. You can import the `Context` like this: `from flwr.common import Context`
(ClientAppActor pid=2421) 
(ClientAppActor pid=2421)             This is a deprecated feature. It will be removed
(ClientAppActor pid=2421)             entirely in future versions of Flower.
(ClientAppActor pid=2421)         
(pythonProject) sathwik@SATHWIKs-MacBook-Air aws data % 
