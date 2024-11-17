# Copyright (c) 2023 Jeff Tan, Carnegie Mellon University.
import multiprocessing
import os


def gpu_map(func, args, gpus=None, method="static"):
    """Map a function over GPUs

    Args:
        func (Function): Function to parallelize
        args (List(Tuple)): List of argument tuples, to split evenly over GPUs
        gpus (List(int) or None): Optional list of GPU device IDs to use
        method (str): Either "static" or "dynamic" (default "static").
            Static assignment is the fastest if workload per task is balanced;
            dynamic assignment better handles tasks with uneven workload.
    Returns:
        outs (List): List of outputs
    """
    mp = multiprocessing.get_context("spawn")  # spawn allows CUDA usage
    devices = os.getenv("CUDA_VISIBLE_DEVICES")
    outputs = None

    # Compute list of GPUs
    if gpus is None:
        if devices is None:
            num_gpus = int(os.popen("nvidia-smi -L | wc -l").read())
            gpus = list(range(num_gpus))
        else:
            gpus = [int(n) for n in devices.split(",")]

    # Map arguments over GPUs using static or dynamic assignment
    try:
        if method == "static":
            # Interleave arguments across GPUs
            args_by_rank = [[] for rank in range(len(gpus))]
            for it, arg in enumerate(args):
                args_by_rank[it % len(gpus)].append(arg)

            # Spawn processes
            spawned_procs = []
            result_queue = mp.Queue()
            for rank, gpu_id in enumerate(gpus):
                # Environment variables get copied on process creation
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                proc_args = (func, args_by_rank[rank], rank, result_queue)
                proc = mp.Process(target=gpu_map_static_helper, args=proc_args)
                proc.start()
                spawned_procs.append(proc)

            # Wait to finish
            for proc in spawned_procs:
                proc.join()

            # Construct output list
            outputs_by_rank = {}
            while True:
                try:
                    rank, out = result_queue.get(block=False)
                    outputs_by_rank[rank] = out
                except multiprocessing.queues.Empty:
                    break

            outputs = []
            for it in range(len(args)):
                rank = it % len(gpus)
                idx = it // len(gpus)
                outputs.append(outputs_by_rank[rank][idx])

        elif method == "dynamic":
            gpu_queue = mp.Queue()
            for gpu_id in gpus:
                gpu_queue.put(gpu_id)

            # Spawn processes as GPUs become available
            spawned_procs = []
            result_queue = mp.Queue()
            for it, arg in enumerate(args):
                # Take latest available gpu_id (blocking)
                gpu_id = gpu_queue.get()

                # Environment variables get copied on process creation
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                proc_args = (func, arg, it, gpu_id, result_queue, gpu_queue)
                proc = mp.Process(target=gpu_map_dynamic_helper, args=proc_args)
                proc.start()
                spawned_procs.append(proc)

            # Wait to finish
            for proc in spawned_procs:
                proc.join()

            # Construct output list
            outputs_by_it = {}
            while True:
                try:
                    it, out = result_queue.get(block=False)
                    outputs_by_it[it] = out
                except multiprocessing.queues.Empty:
                    break

            outputs = []
            for it in range(len(args)):
                outputs.append(outputs_by_it[it])

        else:
            raise NotImplementedError

    except Exception as e:
        pass

    # Restore env vars
    finally:
        if devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
        else:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        return outputs


def gpu_map_static_helper(func, args, rank, result_queue):
    out = [func(*arg) for arg in args]
    result_queue.put((rank, out))


def gpu_map_dynamic_helper(func, arg, it, gpu_id, result_queue, gpu_queue):
    out = func(*arg)
    gpu_queue.put(gpu_id)
    result_queue.put((it, out))
