import os
import shlex
from datetime import datetime
from pathlib import Path


def main(args, dispatch_command):
    """
        Entry point for bulk_execute
        Manages the bulk execution of multiple commands in a row
        Instead of chaining commands in terminal (with ";"), this allows to dynamically change order or insert commands in a txt file
        The file is expected to contain one command per line
        Commands must be exactly like in the terminal!
    """
    queue_path = args.queue_file
    cfg_path = Path(os.path.realpath(__file__)).parent.parent / "cfg"
    cfg_queues = [qf for qf in os.listdir(cfg_path) if qf.endswith("queue.txt")]

    if queue_path is None:
        raise ValueError(f"queue_path is None!")
    elif queue_path in cfg_queues:
        queue_path = cfg_path / queue_path
    elif not os.path.exists(queue_path):
        raise ValueError(f"Could not find queue file at {queue_path}. Please provide a valid path to the queue file.")
    _bulk_execute(queue_path, dispatch_command)


def _get_next_command(queue_path):
    queue_root = os.path.dirname(queue_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(queue_path, "r") as f:
        lines = f.readlines()
    if not lines:
        return None
    next_command = lines[0].strip()
    with open(queue_path, "w") as f:
        f.writelines(lines[1:])
    with open(f"{queue_root}/executed_commands.txt", "a") as f:
        f.write(f"[{timestamp}]:\n{next_command}\n")
    return next_command

def _log_failed(command: str, queue_path: str, e: Exception | SystemExit):
    queue_root = os.path.dirname(queue_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    failed_path = f"{queue_root}/failed_commands.txt"
    failed_text = f"----------------------------------------\n"+f"[{timestamp}]:\nCommand failed: {command}\n"+f"Error: {e.code if type(e) == SystemExit else e}\n"
    print(failed_text)
    with open(failed_path, "a") as f:
        f.write(failed_text)

def _bulk_execute(queue_path, dispatch_command):
    while True:
        line = _get_next_command(queue_path)
        if line is None:
            print("Bulk execution completed. No more commands in the queue.")
            break
        argv = shlex.split(line)
        try:
            print("\nBulk executing command: ", argv)
            dispatch_command(argv[1:])
            print("Success...\n")
        except Exception as e:
            _log_failed(line, queue_path, e)
        except SystemExit as e:
            if e.code != 0:
                _log_failed(line, queue_path, e)
                print("Success...")