import torch
from sklearn.metrics import f1_score
from utils import EarlyStopping, load_data, GPUEnergyMonitor
from torchinfo import summary
import logging
import time

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1

def main(args):
    gpu_monitor = GPUEnergyMonitor(0)
    gpu_monitor.get_device_handle()
    # Get the GPU power consumption before the program runs
    start_power = gpu_monitor.get_power_usage()

    (
        g,
        features,
        labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
    ) = load_data(args["dataset"], args["remove_self_loop"], args["training_rate"], args["data_shuffle"],logger)

    if hasattr(torch, "BoolTensor"):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features = features.to(args["device"])
    labels = labels.to(args["device"])
    train_mask = train_mask.to(args["device"])
    val_mask = val_mask.to(args["device"])
    test_mask = test_mask.to(args["device"])

    from model import SpikingHGNN

    model = SpikingHGNN(
        num_meta_paths=len(g),
        in_size=features.shape[1],
        hidden_size=args["hidden_units"],
        out_size=num_classes,
        T = args["T"],
        alpha = args["alpha"],
        tau = args["tau"],
        surrogate = args["surrogate"],
        neuron = args["neuron"],
        reset = args["reset"],
        threshold = args["threshold"],
        dropout1 = args["dropout1"],
        dropout2 = args["dropout2"]
    ).to(args["device"])
    g = [graph.to(args["device"]) for graph in g]

    model_params = summary(model, _input_data=(g, features), verbose=0)
    logger.info("=" * 100)
    logger.info('total_model_params_num: {}'.format(model_params.total_params))
    logger.info('total_model_param_bytes: {}'.format(model_params.total_param_bytes))
    logger.info('model_param_size (MB): {}'.format(model_params.total_param_bytes / (1024 * 1024)))  # model_params.total_param_bytes/1000000
    logger.info("=" * 100)

    stopper = EarlyStopping(patience=args["patience"], directory = args["log_dir"], logger = logger)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

    total_energy_consumption = 0.0
    energy_consumption_list = []
    start_time = time.time()
    for epoch in range(args["num_epochs"]):
        e_start_time = time.time()
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(
            logits[train_mask], labels[train_mask]
        )
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(
            model, g, features, labels, val_mask, loss_fcn
        )
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        e_end_time = time.time()
        # Get the current GPU power consumption in the epoch
        epoch_power_usage = gpu_monitor.get_power_usage()
        epoch_energy_consumption = gpu_monitor.calculate_energy_consumption(start_power, epoch_power_usage, e_end_time - e_start_time)
        energy_consumption_list.append(epoch_energy_consumption)
        total_energy_consumption += epoch_energy_consumption

        logger.info(
            "Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | "
            "Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f} | energy_consumption {:.4f} J".format(
                epoch + 1,
                loss.item(),
                train_micro_f1,
                train_macro_f1,
                val_loss.item(),
                val_micro_f1,
                val_macro_f1,
                epoch_energy_consumption
            )
        )

        if early_stop:
            break

    end_time = time.time()
    gpu_monitor.cleanup()

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(
        model, g, features, labels, test_mask, loss_fcn
    )
    logger.info(
        "Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} | Training time {:.2f} s | Max memory allocated {:.2f} MB | Total Energy Consumption: {:.2f} J".format(
            test_loss.item(), test_micro_f1, test_macro_f1, end_time - start_time,  torch.cuda.max_memory_allocated(args["device"]) / (1024 ** 2), total_energy_consumption
        )
    )

if __name__ == "__main__":
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser("SpikingHGNN")
    # dataset
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--dataset", nargs="?", default="DBLP",help="Dataset, including ACM, DBLP ,and IMDB")
    parser.add_argument('--data_shuffle', type=bool, default=True, help="Dataset shuffle" )
    parser.add_argument('--remove_self_loop', type=bool, default=False, help="remove graph self_loop" )
    parser.add_argument('--training_rate', type=float, default=0.2, help="Training rate (default: 0.2)")

    # configure
    parser.add_argument("-ld", "--log-dir", type=str, default="results", help="Dir for saving training results", )
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 0.005)", )
    parser.add_argument('--hidden_units', type=int, default=32, help="hidden units size (default: 64)")
    parser.add_argument('--dropout1', type=float, default=0.6, help="Dropout rate 1 (default: 0.6)")
    parser.add_argument('--dropout2', type=float, default=0.5, help="Dropout rate 2 (default: 0.5)")
    parser.add_argument('--weight_decay', type=float, default=0.0005, help="Weight decay (default: 0.001)")
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of epochs (default: 200)")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early stopping (default: 100)")

    # SNN
    parser.add_argument("--T",type=int,default=9,help="Time steps for spiking neural networks. (default: 11)",)
    parser.add_argument("--alpha",type=float,default=2.0,help="Smooth factor for surrogate learning. (default: 2.0)",)
    parser.add_argument("--tau",type=float,default=1.0,help="default: 1.0",)
    parser.add_argument("--surrogate",nargs="?",default="sigmoid",help="Surrogate function ('sigmoid', 'triangle', 'arctan', 'mg', 'super'). (default: 'sigmoid')",)
    parser.add_argument("--neuron",nargs="?",default="PLIF",help="Spiking neuron used for training. (IF, LIF, PLIF). (default: PLIF)",)
    parser.add_argument("--reset",nargs="?",default="subtract",help="Ways to reset spiking neuron. (zero, subtract). (default: subtract)",)
    parser.add_argument("--threshold",type=float,default=5e-2,help="Voltage threshold in spiking neuron. (default: 5e-3)",) # ACM 5e-1

    args = parser.parse_args().__dict__
    args = setup(args)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[logging.FileHandler(args["log_file"]), logging.StreamHandler()]
    )
    logger = logging.getLogger()
    logger.info(args)
    main(args)

