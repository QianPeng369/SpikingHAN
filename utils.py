import datetime
import errno
import os
import pickle
import random
import pynvml
import dgl
import numpy as np
import torch
from dgl.data.utils import _get_dgl_url, download, get_download_dir
from scipy import io as sio, sparse


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print("Created directory {}".format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print("Directory {} already exists.".format(path))
        else:
            raise


def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = "{}_{:02d}-{:02d}-{:02d}".format(
        dt.date(), dt.hour, dt.minute, dt.second
    )

    return post_fix


def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args["log_dir"], "{}_{}".format(args["dataset"], date_postfix)
    )

    if sampling:
        log_dir = log_dir + "_sampling"

    mkdir_p(log_dir)
    return log_dir

def setup(args):
    set_random_seed(args["seed"])
    args["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    args["log_dir"] = setup_log_dir(args)
    args["log_file"] = os.path.join(args["log_dir"],"training.log")
    args["energy_file"] = os.path.join(args["log_dir"],"epoch_energy_consumption.xlsx")
    return args


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def load_acm(remove_self_loop, training_rate, shuffle, logger):

    data_path = './datasets/ACM3025.pkl'

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    labels, features = (
        torch.from_numpy(data["label"].todense()).long(),
        torch.from_numpy(data["feature"].todense()).float(),
    )
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]
    num_nodes = data["label"].shape[0]

    if remove_self_loop:
        data["PAP"] = sparse.csr_matrix(data["PAP"] - np.eye(num_nodes))
        data["PLP"] = sparse.csr_matrix(data["PLP"] - np.eye(num_nodes))

    # Adjacency matrices for meta path based neighbors
    # (Mufei): I verified both of them are binary adjacency matrices with self loops
    author_g = dgl.from_scipy(data["PAP"])
    subject_g = dgl.from_scipy(data["PLP"])
    gs = [author_g, subject_g]

    node_ids = np.arange(num_nodes)
    train_size = int(training_rate * len(node_ids))
    val_size = int(0.1 * len(node_ids))
    test_size = len(node_ids) - train_size - val_size

    if shuffle:
        shuffled_ids = node_ids.copy()
        np.random.shuffle(shuffled_ids)
        train_ids = shuffled_ids[:train_size]
        val_ids = shuffled_ids[train_size:train_size + val_size]
        test_ids = shuffled_ids[train_size + val_size:]
        train_idx = torch.from_numpy(train_ids).long()
        val_idx = torch.from_numpy(val_ids).long()
        test_idx = torch.from_numpy(test_ids).long()
    else:
        train_idx = torch.from_numpy(data["train_idx"]).long().squeeze(0)
        val_idx = torch.from_numpy(data["val_idx"]).long().squeeze(0)
        test_idx = torch.from_numpy(data["test_idx"]).long().squeeze(0)

    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    logger.info("dataset loaded")
    logger.info("dataset: {}".format("ACM"))
    logger.info("train rate: {}".format(train_mask.sum().item() / num_nodes))
    logger.info("val rate: {}".format(val_mask.sum().item() / num_nodes))
    logger.info("test rate: {}".format(test_mask.sum().item() / num_nodes))

    return (
        gs,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
    )

def load_dblp(remove_self_loop,training_rate, shuffle, logger):
    data_path = './datasets/DBLP4057.pkl'
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    labels, features = (
        torch.from_numpy(data["label"].todense()).long(),
        torch.from_numpy(data["feature"].todense()).float(),
    )
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]
    num_nodes = data["label"].shape[0]

    if remove_self_loop:
        data["APA"] = sparse.csr_matrix(data["APA"] - np.eye(num_nodes))
        data["APTPA"] = sparse.csr_matrix(data["APTPA"] - np.eye(num_nodes))
        data["APVPA"] = sparse.csr_matrix(data["APVPA"] - np.eye(num_nodes))
    g_apa = dgl.from_scipy(data["APA"])
    g_aptpa = dgl.from_scipy(data["APTPA"])
    g_apvpa = dgl.from_scipy(data["APVPA"])
    gs = [g_apa, g_aptpa, g_apvpa]

    node_ids = np.arange(num_nodes)
    train_size = int(training_rate * len(node_ids))
    val_size = int(0.1 * len(node_ids))
    test_size = len(node_ids) - train_size - val_size

    if shuffle:
        shuffled_ids = node_ids.copy()
        np.random.shuffle(shuffled_ids)
        train_ids = shuffled_ids[:train_size]
        val_ids = shuffled_ids[train_size:train_size + val_size]
        test_ids = shuffled_ids[train_size + val_size:]
    else:
        train_ids = node_ids[:train_size]
        val_ids = node_ids[train_size:train_size + val_size]
        test_ids = node_ids[train_size + val_size:]
    train_idx = torch.from_numpy(train_ids).long()
    val_idx = torch.from_numpy(val_ids).long()
    test_idx = torch.from_numpy(test_ids).long()

    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    logger.info("dataset loaded")
    logger.info("dataset: {}".format("DBLP"))
    logger.info("train rate: {}".format(train_mask.sum().item() / num_nodes))
    logger.info("val rate: {}".format(val_mask.sum().item() / num_nodes))
    logger.info("test rate: {}".format(test_mask.sum().item() / num_nodes))

    return (
        gs,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
    )

def load_IMDB(remove_self_loop, training_rate, shuffle, logger):
    data_path = './datasets/IMDB4278.pkl'
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    labels, features = (
        torch.from_numpy(data["label"].todense()).long(),
        torch.from_numpy(data["feature"].todense()).float(),
    )
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]
    num_nodes = data["label"].shape[0]

    if remove_self_loop:
        data["MDM"] = sparse.csr_matrix(data["MDM"] - np.eye(num_nodes))
        data["MAM"] = sparse.csr_matrix(data["MAM"] - np.eye(num_nodes))
    g_mdm = dgl.from_scipy(data["MDM"])
    g_mam = dgl.from_scipy(data["MAM"])
    gs = [g_mdm, g_mam]

    node_ids = np.arange(num_nodes)
    train_size = int(training_rate * len(node_ids))
    val_size = int(0.1 * len(node_ids))
    test_size = len(node_ids) - train_size - val_size

    if shuffle:
        shuffled_ids = node_ids.copy()
        np.random.shuffle(shuffled_ids)
        train_ids = shuffled_ids[:train_size]
        val_ids = shuffled_ids[train_size:train_size + val_size]
        test_ids = shuffled_ids[train_size + val_size:]
    else:
        train_ids = node_ids[:train_size]
        val_ids = node_ids[train_size:train_size + val_size]
        test_ids = node_ids[train_size + val_size:]
    train_idx = torch.from_numpy(train_ids).long()
    val_idx = torch.from_numpy(val_ids).long()
    test_idx = torch.from_numpy(test_ids).long()

    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    logger.info("dataset loaded")
    logger.info("dataset: {}".format("IMDB"))
    logger.info("train rate: {}".format(train_mask.sum().item() / num_nodes))
    logger.info("val rate: {}".format(val_mask.sum().item() / num_nodes))
    logger.info("test rate: {}".format(test_mask.sum().item() / num_nodes))

    return (
        gs,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
    )

def load_data(dataset, remove_self_loop=False, training_rate = 0.1, shuffle = False, logger = None):
    if dataset == "DBLP":
        return load_dblp(remove_self_loop, training_rate, shuffle, logger)
    elif dataset == "ACM":
        return load_acm(remove_self_loop, training_rate, shuffle, logger)
    elif dataset == "IMDB":
        return load_IMDB(remove_self_loop, training_rate, shuffle, logger)
    else:
        return NotImplementedError("Unsupported dataset {}".format(dataset))


class EarlyStopping(object):
    def __init__(self, patience=10, directory = None, logger = None):
        dt = datetime.datetime.now()
        self.filename = "early_stop_{}_{:02d}-{:02d}-{:02d}.pth".format(
            dt.date(), dt.hour, dt.minute, dt.second
        )
        self.directory = directory
        self.logger = logger
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            self.logger.info("EarlyStopping counter: {0} out of {1}".format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        file_path = os.path.join(self.directory, self.filename)
        torch.save(model.state_dict(), file_path)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        file_path = os.path.join(self.directory, self.filename)
        model.load_state_dict(torch.load(file_path))


class GPUEnergyMonitor(object):
    def __init__(self, device_index = 0):
        self.device_index = device_index
        self.handle = None
        # Initialize NVML
        self.initialize_nvml()

    def initialize_nvml(self):
        pynvml.nvmlInit()

    def get_device_handle(self):
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

    def get_power_usage(self):
        # Gets the current power consumption of the entire GPU, so may include the power consumption of other applications that are using the GPU.
        return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0 # Convert from milliwatts to watts

    def calculate_energy_consumption(self, start_power, end_power, duration):
        power_difference = end_power - start_power
        return power_difference * duration  # Energy in joules

    def cleanup(self):
        # # Shutdown NVML
        pynvml.nvmlShutdown()
