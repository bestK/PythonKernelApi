import atexit
import json
import os
import pathlib
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
from time import sleep

import config
import utils
from dotenv import load_dotenv
from jupyter_client import BlockingKernelClient

load_dotenv(".env")


# Set up globals
messaging = None
logger = config.get_logger()


class FlushingThread(threading.Thread):
    def __init__(self, kc, kill_sema):
        threading.Thread.__init__(self)
        self.kill_sema = kill_sema
        self.kc = kc

    def run(self):
        logger.info("Running message flusher...")
        while True:
            if self.kill_sema.acquire(blocking=False):
                logger.info("Sema was released to kill thread")
                sys.exit()

            flush_kernel_msgs(self.kc, None)
            time.sleep(1)


def cleanup_spawned_processes():
    print("Cleaning up kernels...")
    for filename in os.listdir(config.KERNEL_PID_DIR):
        fp = os.path.join(config.KERNEL_PID_DIR, filename)
        if os.path.isfile(fp):
            try:
                pid = int(filename.split(".pid")[0])
                logger.debug("Killing process with pid %s" % pid)
                os.remove(fp)
                try:
                    if os.name == "nt":
                        os.kill(pid, signal.CTRL_BREAK_EVENT)
                    else:
                        os.kill(pid, signal.SIGKILL)

                    # After successful kill, cleanup pid file
                    os.remove(fp)

                except Exception:
                    # Windows process killing is flaky
                    pass
            except Exception as e:
                logger.debug(f'cleanup_spawned_processes error:{e}')


def start_snakemq(kc):
    global messaging

    messaging, link = utils.init_snakemq(config.IDENT_KERNEL_MANAGER, "connect")

    def on_recv(conn, ident, message):
        logger.debug(f"km.on_recv:{message}")
        if ident == config.IDENT_MAIN:
            message = json.loads(message.data.decode("utf-8"))

            if message["type"] == "execute":
                logger.debug("Executing command: %s" % message["value"])
                kc.execute(message["value"])
                # Try direct flush with default wait (0.2)
                flush_kernel_msgs(kc, message["uid"])

    messaging.on_message_recv.add(on_recv)

    start_flusher(kc)

    # Send alive
    utils.send_json(messaging, {"type": "status", "value": "ready"}, config.IDENT_MAIN)
    logger.info("Python kernel ready to receive messages!")

    logger.info("Starting snakemq loop")

    try:
        link.loop()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error("Error in snakemq loop: %s" % e)
        sys.exit(1)


def start_flusher(kc):
    # Start FlushMessenger
    kill_sema = threading.Semaphore()
    kill_sema.acquire()
    t = FlushingThread(kc, kill_sema)
    t.start()

    def end_thread():
        kill_sema.release()

    atexit.register(end_thread)


def send_message(uid, message, message_type="message"):
    utils.send_json(
        messaging,
        {"uid": uid, "type": message_type, "value": message},
        config.IDENT_MAIN,
    )


def flush_kernel_msgs(kc, uid="1", tries=1, timeout=0.2):
    logger.debug(f"flush_kernel_msgs {uid}")
    try:
        hit_empty = 0

        while True:
            try:
                msg = kc.get_iopub_msg(timeout=timeout)
                if msg["msg_type"] == "execute_result":
                    if "text/plain" in msg["content"]["data"]:
                        send_message(
                            uid, msg["content"]["data"]["text/plain"], "message_raw"
                        )
                if msg["msg_type"] == "display_data":
                    if "image/png" in msg["content"]["data"]:
                        # Convert to Slack upload
                        send_message(
                            uid,
                            msg["content"]["data"]["image/png"],
                            message_type="image/png",
                        )
                    elif "text/plain" in msg["content"]["data"]:
                        send_message(uid, msg["content"]["data"]["text/plain"])

                elif msg["msg_type"] == "stream":
                    logger.debug("Received stream output %s" % msg["content"]["text"])
                    send_message(uid, msg["content"]["text"])
                elif msg["msg_type"] == "error":
                    send_message(
                        uid,
                        utils.escape_ansi("\n".join(msg["content"]["traceback"])),
                        "message_raw",
                    )
            except queue.Empty:
                hit_empty += 1
                if hit_empty == tries:
                    # Empty queue for one second, give back control
                    break
            except (ValueError, IndexError):
                # get_iopub_msg suffers from message fetch errors
                break
            except Exception as e:
                logger.debug(f"flush_kernel_msgs error:{e} [{type(e)}")
                logger.debug(traceback.format_exc())
                break
    except Exception as e:
        logger.debug(f"flush_kernel_msgs error!:{e} [{type(e)}")


def start_kernel():
    kernel_connection_file = os.path.join(os.getcwd(), "kernel_connection_file.json")

    if os.path.isfile(kernel_connection_file):
        os.remove(kernel_connection_file)
    if os.path.isdir(kernel_connection_file):
        os.rmdir(kernel_connection_file)

    launch_kernel_script_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(), "launch_kernel.py"
    )

    os.makedirs("workspace/", exist_ok=True)

    kernel_process = subprocess.Popen(
        [
            sys.executable,
            launch_kernel_script_path,
            "--IPKernelApp.connection_file",
            kernel_connection_file,
            "--matplotlib=inline",
            "--quiet",
        ],
        cwd="workspace/",
    )
    # Write PID for caller to kill
    str_kernel_pid = str(kernel_process.pid)
    os.makedirs(config.KERNEL_PID_DIR, exist_ok=True)
    with open(os.path.join(config.KERNEL_PID_DIR, str_kernel_pid + ".pid"), "w") as p:
        p.write("kernel")

    # Wait for kernel connection file to be written
    while True:
        if not os.path.isfile(kernel_connection_file):
            sleep(0.1)
        else:
            # Keep looping if JSON parsing fails, file may be partially written
            try:
                with open(kernel_connection_file, "r") as fp:
                    json.load(fp)
                break
            except json.JSONDecodeError:
                pass

    # Client
    kc = BlockingKernelClient(connection_file=kernel_connection_file)
    kc.load_connection_file()
    kc.start_channels()
    kc.wait_for_ready()
    return kc


if __name__ == "__main__":
    kc = start_kernel()
    start_snakemq(kc)
