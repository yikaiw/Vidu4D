# WIP by Gengshan Yang
# generate three visualizations (reference view, bev, turntable) rendering, mesh export in parallel
# python scripts/run_rendering_parallel.py logdir/dog-98-category-comp/opts.log 0-2 0,1,2
import sys
import subprocess

# Set the flagfile.
flagfile = sys.argv[1]

# Set the range of inst_ids.
start_inst_id, end_inst_id = map(int, sys.argv[2].split("-"))
id_list = list(range(start_inst_id, end_inst_id + 1))

# Set the devices id
dev_list = sys.argv[3].split(",")
dev_list = list(map(int, dev_list))
num_devices = len(dev_list)
id_per_device = len(id_list) // num_devices

print(
    "rendering videos",
    id_list,
    "on devices",
    dev_list,
)

# render proxy over rounds
logdir = flagfile.rsplit("/", 1)[0]
subprocess.Popen(
    f"python scripts/render_intermediate.py --testdir {logdir}/", shell=True
)

# Loop over each device.
for dev_id, device in enumerate(dev_list):
    # Initialize an empty command list for this device.
    command_for_device = []

    # Loop over the inst_ids assigned to this device.
    if dev_id == num_devices - 1:
        assigned_ids = id_list[dev_id * id_per_device :]
    else:
        assigned_ids = id_list[dev_id * id_per_device : (dev_id + 1) * id_per_device]
    for inst_id in assigned_ids:
        # Add the command for this inst_id to the device's command list.
        command_for_device.append(
            f"CUDA_VISIBLE_DEVICES={device} python lab4d/render.py --flagfile={flagfile} --load_suffix latest --inst_id {inst_id} --render_res 256 --data_prefix full"
        )
        command_for_device.append(
            f"CUDA_VISIBLE_DEVICES={device} python lab4d/render.py --flagfile={flagfile} --load_suffix latest --inst_id {inst_id} --render_res 256 --viewpoint rot-0-360"
        )
        # command_for_device.append(
        #     f"CUDA_VISIBLE_DEVICES={device} python lab4d/render.py --flagfile={flagfile} --load_suffix latest --inst_id {inst_id} --render_res 256 --viewpoint bev-90"
        # )
        # command_for_device.append(
        #     f"CUDA_VISIBLE_DEVICES={device} python lab4d/export.py --flagfile={flagfile} --load_suffix latest --inst_id {inst_id}"
        # )

        # Add a delay between commands to avoid overloading the device.
        command_for_device.append("sleep 1")

    # Join all commands for this device into a single string.
    command_str = "; ".join(command_for_device)

    # Start a screen session for this device, executing the device's command string.
    subprocess.Popen(
        f'screen -S render-{device}-{",".join(str(i) for i in assigned_ids)} -d -m bash -c "{command_str}"',
        shell=True,
    )
