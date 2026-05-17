import os
import requests 
import json
import numpy as np
import tensorflow as tf
import torch 
from tqdm import tqdm
import urllib.request
import ssl
import zipfile
from pathlib import Path


def download_and_load_gpt2(model_size, models_dir):
    """
    Downloads a GPT-2 model from OpenAI's public storage and loads its parameters.
    This function downloads all necessary files for a specified GPT-2 model size,
    including model checkpoints, vocabulary, encoder, and hyperparameters. After 
    downloading, it loads the model settings and parameters from the checkpoint.
    Parameters
    ----------
    model_size : str
        Size of the GPT-2 model to download. Must be one of: "124M", "355M", 
        "774M", or "1558M".
    models_dir : str
        Directory path where the model files will be stored. A subdirectory
        with the model_size name will be created.
    Returns
    -------
    tuple
        A tuple containing:
        - settings (dict): Model hyperparameters and settings
        - params (dict): Model parameters loaded from the TensorFlow checkpoint
    Raises
    ------
    ValueError
        If the specified model_size is not one of the allowed sizes.
    Notes
    -----
    This function requires the download_file and load_gpt2_params_from_tf_ckpt 
    functions to be defined elsewhere in the code.
    """

    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    ## We have reached here until now ---> we have downloaded the files on our local machine.

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params

def download_file(url, destination):
    """
    Downloads a file from a specified URL to a local destination with progress tracking.

    This function downloads a file from the provided URL to the specified local path,
    showing a progress bar during download. It checks if the file already exists and
    has the same size before downloading to avoid unnecessary transfers.

    Args:
        url (str): The URL of the file to download.
        destination (str): The local file path where the downloaded file will be saved.

    Returns:
        None

    Notes:
        - SSL verification is disabled for the download request.
        - If the file already exists with the same size, the download is skipped.
        - The progress bar shows the download progress using tqdm.

    Raises:
        requests.exceptions.RequestException: If there's an error during the download process.
    """
    
    try:
        # Send a GET request to download the file, disabling SSL verification
        response = requests.get(url, stream=True, verify=False)

        # Get the total file size from headers, defaulting to 0 if not present
        file_size = int(response.headers.get("content-length", 0))

        # Check if file exists and has the same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        # Define the block size for reading the file
        block_size = 1024  # 1 Kilobyte

        # Initialize the progress bar with total file size
        progress_bar_description = url.split("/")[-1]  # Extract filename from URL
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            # Open the destination file in binary write mode
            with open(destination, "wb") as file:
                # Iterate over the file data in chunks
                for chunk in response.iter_content(block_size):
                    progress_bar.update(len(chunk))  # Update progress bar
                    file.write(chunk)  # Write the chunk to the file

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        print(f"Please check the URL: {url}")

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """
    Load GPT-2 model parameters from a TensorFlow checkpoint file and organize them
    into a structured dictionary.
    This function extracts all variables from a TensorFlow checkpoint, processes their
    names to determine their position in the model architecture, and organizes them
    into a nested dictionary that mirrors the GPT-2 model structure with blocks (layers).
    Parameters:
    ----------
    ckpt_path : str
        Path to the TensorFlow checkpoint file containing GPT-2 model weights.
    settings : dict
        Dictionary containing model configuration settings.
        Must include 'n_layer' key specifying the number of transformer layers.
    Returns:
    -------
    dict
        A nested dictionary containing all parameters organized by layer and component.
        The top level contains a 'blocks' key with a list of dictionaries, each
        representing a transformer layer. Other top-level parameters are stored directly.
    Examples:
    --------
    >>> settings = {"n_layer": 12}
    >>> params = load_gpt2_params_from_tf_ckpt("path/to/checkpoint", settings)
    """

    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

def assign(left, right):
    """
    Create a PyTorch parameter by converting the right tensor while ensuring shape compatibility with the left tensor.
    Parameters:
    ----------
    left : torch.Tensor or compatible shape object
        The reference tensor whose shape must match the right tensor.
    right : array-like
        The values to convert to a PyTorch parameter.
    Returns:
    -------
    torch.nn.Parameter
        A PyTorch parameter containing the right tensor values.
    Raises:
    ------
    ValueError
        If the shapes of left and right tensors don't match.
    Examples:
    --------
    >>> left = torch.zeros((3, 4))
    >>> right = np.random.rand(3, 4)
    >>> param = assign(left, right)
    >>> isinstance(param, torch.nn.Parameter)
    True
    """

    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    """
    Load pre-trained weights from a parameters dictionary into a GPT model.
    This function maps weights from a pre-trained GPT model (typically from OpenAI's GPT implementation)
    into a custom GPT model implementation. It handles all components of the model architecture:
    - Token and positional embeddings
    - Transformer blocks:
        - Multi-head attention (query, key, value weights and biases)
        - Output projection layers
        - Feed-forward neural networks
        - Layer normalization parameters
    - Final layer normalization
    - Output head (using the same weights as token embeddings)
    Args:
            gpt (GPTModel): The GPT model instance to load weights into
            params (dict): Dictionary containing the pre-trained model parameters
                                        with a specific structure that matches OpenAI's GPT checkpoints
    Returns:
            GPTModel: The GPT model with weights loaded from params
    Note:
            This function assumes that the architecture of the provided GPT model matches
            the structure of the weights in the params dictionary. The `assign` function 
            (not shown) is used to properly assign the weight tensors.
    """

    # load token and positional embedding weights
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    # load the weights for each transformer block
    for b in range(len(params["blocks"])):

        # load Query, Key, Value weights
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # load Query, key , value biases
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # load output projection layer
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # load Feed-forward NN 
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # load layer normalization weights
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])
    
    # load final normalization layer weights
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # load output head weights reuse of the token embedding weights
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

    return gpt

def _download_and_load_file(file_path, url):
    
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    if not os.path.exists(file_path):
        print(f"Downloading {url} to {file_path}...")
        with urllib.request.urlopen(url, context=ssl_context) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        print(f"{file_path} already exists. Skipping download and extraction.")
    
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    return text_data

def download_and_unzip_spam_data(url, working_path, zip_name, data_file_name):
    
    # Check if the data file already exists
    if (working_path / data_file_name).exists():
        print(f"{(working_path / data_file_name)} already exists. Skipping download and extraction.")
        return

    # Create the folder if it does not exist
    working_path.mkdir(parents=True, exist_ok=True)

    # Create an unverified SSL context
    ssl_context = ssl._create_unverified_context()

    # Downloading the file
    with urllib.request.urlopen(url, context=ssl_context) as response:
        with open(working_path / zip_name, "wb") as out_file:
            out_file.write(response.read())

    # Unzipping the file
    with zipfile.ZipFile(working_path / zip_name, "r") as zip_ref:
        zip_ref.extractall(working_path)

    # Add .tsv file extension
    original_file_path = Path(working_path) / data_file_name[:-4]
    os.rename(original_file_path, Path(working_path) / data_file_name)
    print(f"File downloaded and saved as {Path(working_path) / data_file_name[:-4]}")


def query_Ollama( prompt, model="llama3", url="http://localhost:11434/api/chat"):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }


    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )
    request.add_header("Content-Type", "application/json")

     
    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


if __name__ == '__main__':
    from pathlib import Path
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

    working_path = Path("data/raw/sms_spam_collection")
    zip_name = "sms_spam_collection.zip"
    data_file_name = "SMSSpamCollection.tsv"
    
    download_and_unzip_spam_data(url, working_path, zip_name, data_file_name)
    print("Data download and extraction complete.")