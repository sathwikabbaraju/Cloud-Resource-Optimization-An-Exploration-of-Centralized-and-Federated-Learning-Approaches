(pythonProject) sathwik@SATHWIKs-MacBook-Air aws data % python main_federated_learning.py
2025-06-01 19:23:09,906 INFO worker.py:1771 -- Started a local Ray instance.
Starting Flower simulation...
WARNING :   DEPRECATED FEATURE: flwr.simulation.start_simulation() is deprecated.
        Instead, use the `flwr run` CLI command to start a local simulation in your Flower app, as shown for example below:

                $ flwr new  # Create a new Flower app from a template

                $ flwr run  # Run the Flower app in Simulation Mode

        Using `start_simulation()` is deprecated.

            This is a deprecated feature. It will be removed
            entirely in future versions of Flower.
        
INFO :      Starting Flower simulation, config: num_rounds=3, no round_timeout
2025-06-01 19:23:13,394 INFO worker.py:1771 -- Started a local Ray instance.
INFO :      Flower VCE: Ray initialized with resources: {'CPU': 8.0, 'memory': 7606008218.0, 'node:127.0.0.1': 1.0, 'object_store_memory': 2147483648.0, 'node:__internal_head__': 1.0}
INFO :      Optimize your simulation with Flower VCE: https://flower.ai/docs/framework/how-to-run-simulations.html
INFO :      Flower VCE: Resources for each Virtual Client: {'num_cpus': 1, 'num_gpus': 0.0}
INFO :      Flower VCE: Creating VirtualClientEngineActorPool with 8 actors
INFO :      [INIT]
INFO :      Requesting initial parameters from one random client
INFO :      Received initial parameters from one random client
INFO :      Starting evaluation of initial global parameters
Server: Starting evaluation for round 0 on 12 client test sets.
Server Evaluation: Loaded test data from client_0000 (1133736 samples).
(ClientAppActor pid=68216) WARNING :   DEPRECATED FEATURE: `client_fn` now expects a signature `def client_fn(context: Context)`.The provided `client_fn` has signature: {'cid': <Parameter "cid: str">}. You can import the `Context` like this: `from flwr.common import Context`
(ClientAppActor pid=68216) 
(ClientAppActor pid=68216)             This is a deprecated feature. It will be removed
(ClientAppActor pid=68216)             entirely in future versions of Flower.
(ClientAppActor pid=68216)         
(ClientAppActor pid=68216) Client 0009: Initializing FlowerClient object.
(ClientAppActor pid=68216) Client 0009: Using MPS (Apple Silicon GPU)
(ClientAppActor pid=68216) Client 0009: Model initialized on mps.
Server Evaluation: Finished with client_0000. Cache cleared.
Server Evaluation: Loaded test data from client_0001 (1117707 samples).
Server Evaluation: Finished with client_0001. Cache cleared.
Server Evaluation: Loaded test data from client_0002 (1048152 samples).
Server Evaluation: Finished with client_0002. Cache cleared.
Server Evaluation: Loaded test data from client_0003 (1086346 samples).
Server Evaluation: Finished with client_0003. Cache cleared.
Server Evaluation: Loaded test data from client_0004 (1070576 samples).
Server Evaluation: Finished with client_0004. Cache cleared.
Server Evaluation: Loaded test data from client_0005 (1132349 samples).
Server Evaluation: Finished with client_0005. Cache cleared.
Server Evaluation: Loaded test data from client_0006 (1156553 samples).
Server Evaluation: Finished with client_0006. Cache cleared.
Server Evaluation: Loaded test data from client_0007 (1103882 samples).
Server Evaluation: Finished with client_0007. Cache cleared.
Server Evaluation: Loaded test data from client_0008 (1118787 samples).
Server Evaluation: Finished with client_0008. Cache cleared.
Server Evaluation: Loaded test data from client_0009 (1034186 samples).
Server Evaluation: Finished with client_0009. Cache cleared.
Server Evaluation: Loaded test data from client_0010 (1136307 samples).
Server Evaluation: Finished with client_0010. Cache cleared.
Server Evaluation: Loaded test data from client_0011 (608054 samples).
Server Evaluation: Finished with client_0011. Cache cleared.
Server: Round 0 validation loss: 0.0800
INFO :      initial parameters (loss, other metrics): 0.07997350997859781, {'server_loss': 0.07997350997859781}
INFO :      
INFO :      [ROUND 1]
INFO :      configure_fit: strategy sampled 1 clients (out of 12)
(ClientAppActor pid=68216) WARNING :   DEPRECATED FEATURE: `client_fn` now expects a signature `def client_fn(context: Context)`.The provided `client_fn` has signature: {'cid': <Parameter "cid: str">}. You can import the `Context` like this: `from flwr.common import Context`
(ClientAppActor pid=68216) 
(ClientAppActor pid=68216)             This is a deprecated feature. It will be removed
(ClientAppActor pid=68216)             entirely in future versions of Flower.
(ClientAppActor pid=68216)         
(ClientAppActor pid=68216) Client 0006: Initializing FlowerClient object.
(ClientAppActor pid=68216) Client 0006: Using MPS (Apple Silicon GPU)
(ClientAppActor pid=68216) Client 0006: Model initialized on mps.
(ClientAppActor pid=68216) Client 0006: Creating data loaders from memmap.
(ClientAppActor pid=68216) Client 0006: Data loaders created (Train: 5397243 samples, Val: 1156552 samples).
(ClientAppActor pid=68216) Client 0006: Starting local training for 1 epochs.
  INFO :      aggregate_fit: received 1 results and 0 failures
Server: Aggregating fit results for round 1. 1 clients returned results.
(ClientAppActor pid=68216) Client 0006: Epoch 1/1, Loss: 0.0001
(ClientAppActor pid=68216) Client 0006: Local training finished. Cache cleared.
WARNING :   No fit_metrics_aggregation_fn provided
Server: Saved global model checkpoint to checkpoints/global_model_round_1.pth
Server: Aggregation and checkpointing for round 1 completed.
Server: Starting evaluation for round 1 on 12 client test sets.
Server Evaluation: Loaded test data from client_0000 (1133736 samples).
Server Evaluation: Finished with client_0000. Cache cleared.
Server Evaluation: Loaded test data from client_0001 (1117707 samples).
Server Evaluation: Finished with client_0001. Cache cleared.
Server Evaluation: Loaded test data from client_0002 (1048152 samples).
Server Evaluation: Finished with client_0002. Cache cleared.
Server Evaluation: Loaded test data from client_0003 (1086346 samples).
Server Evaluation: Finished with client_0003. Cache cleared.
Server Evaluation: Loaded test data from client_0004 (1070576 samples).
Server Evaluation: Finished with client_0004. Cache cleared.
Server Evaluation: Loaded test data from client_0005 (1132349 samples).
Server Evaluation: Finished with client_0005. Cache cleared.
Server Evaluation: Loaded test data from client_0006 (1156553 samples).
Server Evaluation: Finished with client_0006. Cache cleared.
Server Evaluation: Loaded test data from client_0007 (1103882 samples).
Server Evaluation: Finished with client_0007. Cache cleared.
Server Evaluation: Loaded test data from client_0008 (1118787 samples).
Server Evaluation: Finished with client_0008. Cache cleared.
Server Evaluation: Loaded test data from client_0009 (1034186 samples).
Server Evaluation: Finished with client_0009. Cache cleared.
Server Evaluation: Loaded test data from client_0010 (1136307 samples).
Server Evaluation: Finished with client_0010. Cache cleared.
Server Evaluation: Loaded test data from client_0011 (608054 samples).
Server Evaluation: Finished with client_0011. Cache cleared.
Server: Round 1 validation loss: 0.0000
INFO :      fit progress: (1, 2.2486159447038714e-05, {'server_loss': 2.2486159447038714e-05}, 25363.272252042)
INFO :      configure_evaluate: strategy sampled 1 clients (out of 12)
(ClientAppActor pid=68216) WARNING :   DEPRECATED FEATURE: `client_fn` now expects a signature `def client_fn(context: Context)`.The provided `client_fn` has signature: {'cid': <Parameter "cid: str">}. You can import the `Context` like this: `from flwr.common import Context`
(ClientAppActor pid=68216) 
(ClientAppActor pid=68216)             This is a deprecated feature. It will be removed
(ClientAppActor pid=68216)             entirely in future versions of Flower.
(ClientAppActor pid=68216)         
(ClientAppActor pid=68216) Client 0009: Initializing FlowerClient object.
(ClientAppActor pid=68216) Client 0009: Using MPS (Apple Silicon GPU)
(ClientAppActor pid=68216) Client 0009: Model initialized on mps.
(ClientAppActor pid=68216) Client 0009: Creating data loaders from memmap.
(ClientAppActor pid=68216) Client 0009: Data loaders created (Train: 4826199 samples, Val: 1034186 samples).
(ClientAppActor pid=68216) Client 0009: Starting local evaluation.
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
ray.exceptions.RayTaskError(ClientAppException): ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
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
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/fl_model_utils.py", line 30, in forward
    out, (hn, cn) = self.lstm(x, (h0, c0))
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 1124, in forward
    result = _VF.lstm(
RuntimeError: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.

ERROR :     ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
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
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/fl_model_utils.py", line 30, in forward
    out, (hn, cn) = self.lstm(x, (h0, c0))
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 1124, in forward
    result = _VF.lstm(
RuntimeError: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.
INFO :      aggregate_evaluate: received 0 results and 1 failures
INFO :      
INFO :      [ROUND 2]
INFO :      configure_fit: strategy sampled 1 clients (out of 12)
(ClientAppActor pid=68216) WARNING :   DEPRECATED FEATURE: `client_fn` now expects a signature `def client_fn(context: Context)`.The provided `client_fn` has signature: {'cid': <Parameter "cid: str">}. You can import the `Context` like this: `from flwr.common import Context`
(ClientAppActor pid=68216) 
(ClientAppActor pid=68216)             This is a deprecated feature. It will be removed
(ClientAppActor pid=68216)             entirely in future versions of Flower.
(ClientAppActor pid=68216)         
(ClientAppActor pid=68216) Client 0005: Initializing FlowerClient object.
(ClientAppActor pid=68216) Client 0005: Using MPS (Apple Silicon GPU)
(ClientAppActor pid=68216) Client 0005: Model initialized on mps.
(ClientAppActor pid=68216) Client 0005: Creating data loaders from memmap.
(ClientAppActor pid=68216) Client 0005: Data loaders created (Train: 5284293 samples, Val: 1132348 samples).
(ClientAppActor pid=68216) Client 0005: Starting local training for 1 epochs.
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
ray.exceptions.RayTaskError(ClientAppException): ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
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
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 124, in fit
    outputs = self.model(X_batch)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/fl_model_utils.py", line 30, in forward
    out, (hn, cn) = self.lstm(x, (h0, c0))
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 1124, in forward
    result = _VF.lstm(
RuntimeError: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.

ERROR :     ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
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
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 124, in fit
    outputs = self.model(X_batch)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/fl_model_utils.py", line 30, in forward
    out, (hn, cn) = self.lstm(x, (h0, c0))
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 1124, in forward
    result = _VF.lstm(
RuntimeError: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.
INFO :      aggregate_fit: received 0 results and 1 failures
Server: Aggregating fit results for round 2. 0 clients returned results.
Server: Aggregation and checkpointing for round 2 completed.
Server: Starting evaluation for round 2 on 12 client test sets.
Server Evaluation: Loaded test data from client_0000 (1133736 samples).
Server Evaluation: Finished with client_0000. Cache cleared.
Server Evaluation: Loaded test data from client_0001 (1117707 samples).
Server Evaluation: Finished with client_0001. Cache cleared.
Server Evaluation: Loaded test data from client_0002 (1048152 samples).
Server Evaluation: Finished with client_0002. Cache cleared.
Server Evaluation: Loaded test data from client_0003 (1086346 samples).
Server Evaluation: Finished with client_0003. Cache cleared.
Server Evaluation: Loaded test data from client_0004 (1070576 samples).
Server Evaluation: Finished with client_0004. Cache cleared.
Server Evaluation: Loaded test data from client_0005 (1132349 samples).
Server Evaluation: Finished with client_0005. Cache cleared.
Server Evaluation: Loaded test data from client_0006 (1156553 samples).
Server Evaluation: Finished with client_0006. Cache cleared.
Server Evaluation: Loaded test data from client_0007 (1103882 samples).
Server Evaluation: Finished with client_0007. Cache cleared.
Server Evaluation: Loaded test data from client_0008 (1118787 samples).
Server Evaluation: Finished with client_0008. Cache cleared.
Server Evaluation: Loaded test data from client_0009 (1034186 samples).
Server Evaluation: Finished with client_0009. Cache cleared.
Server Evaluation: Loaded test data from client_0010 (1136307 samples).
Server Evaluation: Finished with client_0010. Cache cleared.
Server Evaluation: Loaded test data from client_0011 (608054 samples).
Server Evaluation: Finished with client_0011. Cache cleared.
Server: Round 2 validation loss: 0.0000
INFO :      fit progress: (2, 2.2486159447038714e-05, {'server_loss': 2.2486159447038714e-05}, 26739.026319875004)
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
ray.exceptions.RayTaskError(ClientAppException): ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
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
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/fl_model_utils.py", line 30, in forward
    out, (hn, cn) = self.lstm(x, (h0, c0))
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 1124, in forward
    result = _VF.lstm(
RuntimeError: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.

ERROR :     ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
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
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/fl_model_utils.py", line 30, in forward
    out, (hn, cn) = self.lstm(x, (h0, c0))
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 1124, in forward
    result = _VF.lstm(
RuntimeError: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.
INFO :      aggregate_evaluate: received 0 results and 1 failures
INFO :      
INFO :      [ROUND 3]
INFO :      configure_fit: strategy sampled 1 clients (out of 12)
(ClientAppActor pid=68216) WARNING :   DEPRECATED FEATURE: `client_fn` now expects a signature `def client_fn(context: Context)`.The provided `client_fn` has signature: {'cid': <Parameter "cid: str">}. You can import the `Context` like this: `from flwr.common import Context`
(ClientAppActor pid=68216) 
(ClientAppActor pid=68216)             This is a deprecated feature. It will be removed
(ClientAppActor pid=68216)             entirely in future versions of Flower.
(ClientAppActor pid=68216)         
(ClientAppActor pid=68216) Client 0007: Initializing FlowerClient object.
(ClientAppActor pid=68216) Client 0007: Using MPS (Apple Silicon GPU)
(ClientAppActor pid=68216) Client 0007: Model initialized on mps.
(ClientAppActor pid=68216) Client 0007: Creating data loaders from memmap.
(ClientAppActor pid=68216) Client 0007: Data loaders created (Train: 5151448 samples, Val: 1103882 samples).
(ClientAppActor pid=68216) Client 0007: Starting local evaluation.
(ClientAppActor pid=68216) WARNING :   DEPRECATED FEATURE: `client_fn` now expects a signature `def client_fn(context: Context)`.The provided `client_fn` has signature: {'cid': <Parameter "cid: str">}. You can import the `Context` like this: `from flwr.common import Context`
(ClientAppActor pid=68216) 
(ClientAppActor pid=68216)             This is a deprecated feature. It will be removed
(ClientAppActor pid=68216)             entirely in future versions of Flower.
(ClientAppActor pid=68216)         
(ClientAppActor pid=68216) Client 0006: Initializing FlowerClient object.
(ClientAppActor pid=68216) Client 0006: Using MPS (Apple Silicon GPU)
(ClientAppActor pid=68216) Client 0006: Model initialized on mps.
(ClientAppActor pid=68216) Client 0006: Creating data loaders from memmap.
(ClientAppActor pid=68216) Client 0006: Data loaders created (Train: 5397243 samples, Val: 1156552 samples).
(ClientAppActor pid=68216) Client 0006: Starting local training for 1 epochs.
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
ray.exceptions.RayTaskError(ClientAppException): ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
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
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 124, in fit
    outputs = self.model(X_batch)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/fl_model_utils.py", line 30, in forward
    out, (hn, cn) = self.lstm(x, (h0, c0))
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 1124, in forward
    result = _VF.lstm(
RuntimeError: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.

ERROR :     ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
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
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/main_federated_learning.py", line 124, in fit
    outputs = self.model(X_batch)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/fl_model_utils.py", line 30, in forward
    out, (hn, cn) = self.lstm(x, (h0, c0))
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 1124, in forward
    result = _VF.lstm(
RuntimeError: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.
INFO :      aggregate_fit: received 0 results and 1 failures
Server: Aggregating fit results for round 3. 0 clients returned results.
Server: Aggregation and checkpointing for round 3 completed.
Server: Starting evaluation for round 3 on 12 client test sets.
Server Evaluation: Loaded test data from client_0000 (1133736 samples).
Server Evaluation: Finished with client_0000. Cache cleared.
Server Evaluation: Loaded test data from client_0001 (1117707 samples).
Server Evaluation: Finished with client_0001. Cache cleared.
Server Evaluation: Loaded test data from client_0002 (1048152 samples).
Server Evaluation: Finished with client_0002. Cache cleared.
Server Evaluation: Loaded test data from client_0003 (1086346 samples).
Server Evaluation: Finished with client_0003. Cache cleared.
Server Evaluation: Loaded test data from client_0004 (1070576 samples).
Server Evaluation: Finished with client_0004. Cache cleared.
Server Evaluation: Loaded test data from client_0005 (1132349 samples).
Server Evaluation: Finished with client_0005. Cache cleared.
Server Evaluation: Loaded test data from client_0006 (1156553 samples).
Server Evaluation: Finished with client_0006. Cache cleared.
Server Evaluation: Loaded test data from client_0007 (1103882 samples).
Server Evaluation: Finished with client_0007. Cache cleared.
Server Evaluation: Loaded test data from client_0008 (1118787 samples).
Server Evaluation: Finished with client_0008. Cache cleared.
Server Evaluation: Loaded test data from client_0009 (1034186 samples).
Server Evaluation: Finished with client_0009. Cache cleared.
Server Evaluation: Loaded test data from client_0010 (1136307 samples).
Server Evaluation: Finished with client_0010. Cache cleared.
Server Evaluation: Loaded test data from client_0011 (608054 samples).
Server Evaluation: Finished with client_0011. Cache cleared.
Server: Round 3 validation loss: 0.0000
INFO :      fit progress: (3, 2.2486159447038714e-05, {'server_loss': 2.2486159447038714e-05}, 28060.027217208997)
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
ray.exceptions.RayTaskError(ClientAppException): ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
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
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/fl_model_utils.py", line 30, in forward
    out, (hn, cn) = self.lstm(x, (h0, c0))
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 1124, in forward
    result = _VF.lstm(
RuntimeError: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.

ERROR :     ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
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
  File "/Users/sathwik/VISUAL STUDIO CODE/Cloud:Fog/aws data/fl_model_utils.py", line 30, in forward
    out, (hn, cn) = self.lstm(x, (h0, c0))
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/torch/nn/modules/rnn.py", line 1124, in forward
    result = _VF.lstm(
RuntimeError: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.

The above exception was the direct cause of the following exception:

ray::ClientAppActor.run() (pid=68216, ip=127.0.0.1, actor_id=2a214d0d07167094154fdf6001000000, repr=<flwr.simulation.ray_transport.ray_actor.ClientAppActor object at 0x111803280>)
  File "/Users/sathwik/.conda/envs/pythonProject/lib/python3.10/site-packages/flwr/simulation/ray_transport/ray_actor.py", line 64, in run
    raise ClientAppException(str(ex)) from ex
flwr.client.client_app.ClientAppException: 
Exception ClientAppException occurred. Message: MPS backend out of memory (MPS allocated: 24.00 MB, other allocations: 362.51 GB). Tried to allocate 256 bytes on shared pool.
INFO :      aggregate_evaluate: received 0 results and 1 failures
INFO :      
INFO :      [SUMMARY]
INFO :      Run finished 3 round(s) in 28060.05s
INFO :          History (loss, centralized):
INFO :                  round 0: 0.07997350997859781
INFO :                  round 1: 2.2486159447038714e-05
INFO :                  round 2: 2.2486159447038714e-05
INFO :                  round 3: 2.2486159447038714e-05
INFO :          History (metrics, centralized):
INFO :          {'server_loss': [(0, 0.07997350997859781),
INFO :                           (1, 2.2486159447038714e-05),
INFO :                           (2, 2.2486159447038714e-05),
INFO :                           (3, 2.2486159447038714e-05)]}
INFO :      

Flower simulation finished.