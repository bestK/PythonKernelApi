import utils
import kernel_manager
import config
import asyncio
import json
import logging
import os
import pathlib
import subprocess
import sys
import threading
import time
from queue import Queue

from collections import deque
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS  # Import the CORS library

load_dotenv(".env")

OPENAI_API_TYPE = os.environ.get("OPENAI_API_TYPE", "openai")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION", "2023-03-15-preview")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
APP_PORT = int(os.environ.get("API_PORT", 443))

# Get global logger
logger = config.get_logger()

# Note, only one kernel_manager_process can be active
kernel_manager_process = None

# Use efficient Python queues to store messages
result_queue = Queue()
send_queue = Queue()

messaging = None

# We know this Flask app is for local use. So we can disable the verbose Werkzeug logger
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

cli = sys.modules["flask.cli"]
cli.show_server_banner = lambda *x: None

app = Flask(__name__)
CORS(app)


def start_kernel_manager():
    global kernel_manager_process

    kernel_manager_script_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(), "kernel_manager.py"
    )
    kernel_manager_process = subprocess.Popen(
        [sys.executable, kernel_manager_script_path]
    )

    # Write PID as <pid>.pid to config.KERNEL_PID_DIR
    os.makedirs(config.KERNEL_PID_DIR, exist_ok=True)
    with open(
        os.path.join(config.KERNEL_PID_DIR, "%d.pid" % kernel_manager_process.pid), "w"
    ) as p:
        p.write("kernel_manager")


def cleanup_kernel_program():
    kernel_manager.cleanup_spawned_processes()


async def start_snakemq():
    global messaging

    messaging, link = utils.init_snakemq(config.IDENT_MAIN)

    def on_recv(conn, ident, message):
        message = json.loads(message.data.decode("utf-8"))

        if message["type"] == "status":
            if message["value"] == "ready":
                logger.debug("Kernel is ready.")
                result_queue.put({"value": "Kernel is ready.", "type": "message"})

        elif message["type"] in ["message", "message_raw", "image/png", "image/jpeg"]:
            # TODO: 1:1 kernel <> channel mapping
            logger.debug("%s of type %s" % (message["value"], message["type"]))

            result_queue.put({"value": message["value"], "type": message["type"]})

    messaging.on_message_recv.add(on_recv)
    logger.info("Starting snakemq loop")

    def send_queued_messages():
        while True:
            if send_queue.qsize() > 0:
                message = send_queue.get()
                utils.send_json(
                    messaging,
                    {"type": "execute", "value": message["command"]},
                    config.IDENT_KERNEL_MANAGER,
                )
            time.sleep(0.1)

    async def async_send_queued_messages():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, send_queued_messages)

    async def async_link_loop():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, link.loop)

    # Wrap the snakemq_link.Link loop in an asyncio task
    await asyncio.gather(async_send_queued_messages(), async_link_loop())

@app.route("/", methods=["POST", "GET"])
def handle_index():
    return jsonify({"result": "success"})

@app.route("/python", methods=["POST", "GET"])
def handle_python():
    if request.method == "GET":
        # Handle GET requests by sending everything that's in the receive_queue
        results = [result_queue.get() for _ in range(result_queue.qsize())]
        return jsonify({"results": results})
    elif request.method == "POST":
        data = request.json

        send_queue.put(data)

        return jsonify({"result": "success"})


@app.route("/generate", methods=["POST"])
def generate_code():
    user_prompt = request.json.get("prompt", "")
    user_openai_key = request.json.get("openAIKey", None)
    model = request.json.get("model", None)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    code, status = loop.run_until_complete(
        get_code(user_prompt, user_openai_key, model)
    )
    loop.close()

    # Append all messages to the message buffer for later use
    message_buffer.append(user_prompt + "\n\n", user_openai_key)

    return jsonify({"code": code}), status


@app.route("/upload", methods=["POST"])
def upload_file():
    # check if the post request has the file part
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files["file"]
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], file.filename))
        return jsonify({"result": "File successfully uploaded"}), 200
    else:
        return jsonify({"error": "File type not allowed"}), 400


@app.route("/download")
def download_file():
    # Get query argument file
    file = request.args.get("file")
    if file is None:
        return jsonify({"error": "file must not be null"}), 400
    # from `workspace/` send the file
    # make sure to set required headers to make it download the file
    return send_from_directory(
        os.path.join(os.getcwd(), "workspace"), file, as_attachment=True
    )


@app.route("/restart", methods=["POST"])
def handle_restart():
    cleanup_kernel_program()
    start_kernel_manager()

    return jsonify({"result": "success"})


@app.route("/shutdown", methods=["POST"])
def handle_stop():
    cleanup_kernel_program()
    return jsonify({"result": "bye~"})


class LimitedLengthString:
    def __init__(self, maxlen=2000):
        self.data = {}
        self.len = 0
        self.maxlen = maxlen

    def append(self, string, key=None):
        if key not in self.data:
            self.data[key] = deque()
        self.data[key].append(string)
        self.len += len(string)
        while self.len > self.maxlen:
            popped = self.data[key].popleft()
            self.len -= len(popped)

    def get_string(self, key=None):
        if key is not None and key in self.data:
            result = "".join(self.data[key])
            return result[-self.maxlen :]
        else:
            result = "".join(["".join(strings) for strings in self.data.values()])
            return result[-self.maxlen :]


message_buffer = LimitedLengthString()


async def get_code(user_prompt, user_openai_key=None, model="gpt-3.5-turbo"):
    prompt = f"""First, here is a history of what I asked you to do earlier. 
    The actual prompt follows after ENDOFHISTORY. 
    History:
    {message_buffer.get_string(user_openai_key)}
    ENDOFHISTORY.
    Write Python code, in a triple backtick Markdown code block, that does the following:
    {user_prompt}
    
    Notes: 
        First, think step by step what you want to do and write it down in English.
        Then generate valid Python code in a code block 
        Make sure all code is valid - it be run in a Jupyter Python 3 kernel environment. 
        Define every variable before you use it.
        For data munging, you can use 
            'numpy', # numpy==1.24.3
            'dateparser' #dateparser==1.1.8
            'pandas', # matplotlib==1.5.3
            'geopandas' # geopandas==0.13.2
        For pdf extraction, you can use
            'PyPDF2', # PyPDF2==3.0.1
            'pdfminer', # pdfminer==20191125
            'pdfplumber', # pdfplumber==0.9.0
        For data visualization, you can use
            'matplotlib', # matplotlib==3.7.1
        Be sure to generate charts with matplotlib. If you need geographical charts, use geopandas with the geopandas.datasets module.
        If the user has just uploaded a file, focus on the file that was most recently uploaded (and optionally all previously uploaded files)
    
    Teacher mode: if the code modifies or produces a file, end your output AFTER YOUR CODE BLOCK with a link to it as <a href='/download?file=INSERT_FILENAME_HERE'>Download file</a>. Replace INSERT_FILENAME_HERE with the actual filename. So just print that HTML to stdout at the end, AFTER your code block."""
    temperature = 0.7
    message_array = [
        {
            "role": "user",
            "content": prompt,
        },
    ]

    final_openai_key = OPENAI_API_KEY
    if user_openai_key:
        final_openai_key = user_openai_key

    if OPENAI_API_TYPE == "openai":
        data = {
            "model": model,
            "messages": message_array,
            "temperature": temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {final_openai_key}",
        }

        response = requests.post(
            f"{OPENAI_BASE_URL}/v1/chat/completions",
            data=json.dumps(data),
            headers=headers,
        )
    elif OPENAI_API_TYPE == "azure":
        data = {
            "messages": message_array,
            "temperature": temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "api-key": f"{final_openai_key}",
        }

        response = requests.post(
            f"{OPENAI_BASE_URL}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={OPENAI_API_VERSION}",
            data=json.dumps(data),
            headers=headers,
        )
    else:
        return "Error: Invalid OPENAI_PROVIDER", 500

    def extract_code(text):
        # Match triple backtick blocks first
        triple_match = re.search(r"```(?:\w+\n)?(.+?)```", text, re.DOTALL)
        if triple_match:
            return triple_match.group(1).strip()
        else:
            # If no triple backtick blocks, match single backtick blocks
            single_match = re.search(r"`(.+?)`", text, re.DOTALL)
            if single_match:
                return single_match.group(1).strip()
        # If no code blocks found, return original text
        return text

    if response.status_code != 200:
        return "Error: " + response.text, 500

    return extract_code(response.json()["choices"][0]["message"]["content"]), 200


async def main():
    start_kernel_manager()

    # Run Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.start()

    # Run in background
    await start_snakemq()


def run_flask_app():
    app.run(host="0.0.0.0", port=APP_PORT)


if __name__ == "__main__":
    asyncio.run(main())
