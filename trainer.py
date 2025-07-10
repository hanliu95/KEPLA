import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
from models import binary_cross_entropy, cross_entropy_logits, entropy_logits, RandomLayer, RMSELoss
from prettytable import PrettyTable
from domain_adaptator import ReverseLayerF
from tqdm import tqdm
import torch.nn.functional as F
import scipy.stats
from time import time
# from transformers import AutoTokenizer, EsmModel


class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, opt_da=None, discriminator=None,
                 experiment=None, alpha=1, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.is_da = config["DA"]["USE"]
        self.alpha = alpha
        self.n_class = config["DECODER"]["BINARY"]
        if opt_da:
            self.optim_da = opt_da
        if self.is_da:
            self.da_method = config["DA"]["METHOD"]
            self.domain_dmm = discriminator
            if config["DA"]["RANDOM_LAYER"] and not config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = nn.Linear(in_features=config["DECODER"]["IN_DIM"]*self.n_class, out_features=config["DA"]
                ["RANDOM_DIM"], bias=False).to(self.device)
                torch.nn.init.normal_(self.random_layer.weight, mean=0, std=1)
                for param in self.random_layer.parameters():
                    param.requires_grad = False
            elif config["DA"]["RANDOM_LAYER"] and config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = RandomLayer([config["DECODER"]["IN_DIM"], self.n_class], config["DA"]["RANDOM_DIM"])
                if torch.cuda.is_available():
                    self.random_layer.cuda()
            else:
                self.random_layer = False
        self.da_init_epoch = config["DA"]["INIT_EPOCH"]
        self.init_lamb_da = config["DA"]["LAMB_DA"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.use_da_entropy = config["DA"]["USE_ENTROPY"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.experiment = experiment

        self.best_model = None
        self.best_epoch = None
        self.best_rmse = 100

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_rmse_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        valid_metric_header = ["# Epoch", "RMSE", "MAE", "Val_loss"]
        test_metric_header = ["# Best Epoch", "RMSE", "MAE", "R", "SD", "Test_loss"]
        if not self.is_da:
            train_metric_header = ["# Epoch", "Train_loss"]
        else:
            train_metric_header = ["# Epoch", "Train_loss", "Model_loss", "epoch_lamb_da", "da_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

        self.original_random = config["DA"]["ORIGINAL_RANDOM"]

        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        # self.esm_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").cuda()

    def da_lambda_decay(self):
        delta_epoch = self.current_epoch - self.da_init_epoch
        non_init_epoch = self.epochs - self.da_init_epoch
        p = (self.current_epoch + delta_epoch * self.nb_training) / (
                non_init_epoch * self.nb_training
        )
        grow_fact = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        return self.init_lamb_da * grow_fact

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            if not self.is_da:
                train_loss = self.train_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
                if self.experiment:
                    self.experiment.log_metric("train_epoch model loss", train_loss, epoch=self.current_epoch)
            else:
                train_loss, model_loss, da_loss, epoch_lamb = self.train_da_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss, model_loss,
                                                                                        epoch_lamb, da_loss]))
                self.train_model_loss_epoch.append(model_loss)
                self.train_da_loss_epoch.append(da_loss)
                if self.experiment:
                    self.experiment.log_metric("train_epoch total loss", train_loss, epoch=self.current_epoch)
                    self.experiment.log_metric("train_epoch model loss", model_loss, epoch=self.current_epoch)
                    if self.current_epoch >= self.da_init_epoch:
                        self.experiment.log_metric("train_epoch da loss", da_loss, epoch=self.current_epoch)
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            rmse, mae, r_p, sd, val_loss = self.test(dataloader="val")
            if self.experiment:
                self.experiment.log_metric("valid_epoch model loss", val_loss, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch rmse", rmse, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch mae", mae, epoch=self.current_epoch)
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [rmse, mae, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_rmse_epoch.append(rmse)
            if rmse <= self.best_rmse:
                print('copy!')
                self.best_model = copy.deepcopy(self.model)
                self.best_rmse = rmse
                self.best_epoch = self.current_epoch
            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " RMSE "
                  + str(rmse) + " MAE " + str(mae) + " R " + str(r_p) + " SD " + str(sd))

        rmse, mae, r_p, sd, test_loss = self.test(dataloader="test")

        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " RMSE "
              + str(rmse) + " MAE " + str(mae) + " R " + str(r_p) + " SD " + str(sd))
        self.test_metrics["rmse"] = rmse
        self.test_metrics["mae"] = mae
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["r"] = r_p
        self.test_metrics["sd"] = sd

        self.save_result()
        if self.experiment:
            self.experiment.log_metric("valid_best_rmse", self.best_rmse)
            self.experiment.log_metric("valid_best_epoch", self.best_epoch)
            self.experiment.log_metric("test_rmse", self.test_metrics["rmse"])
            self.experiment.log_metric("test_mae", self.test_metrics["mae"])
            self.experiment.log_metric("r", self.test_metrics["r"])
            self.experiment.log_metric("sd", self.test_metrics["sd"])
        return self.test_metrics

    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        if self.is_da:
            state["train_model_loss"] = self.train_model_loss_epoch
            state["train_da_loss"] = self.train_da_loss_epoch
            state["da_init_epoch"] = self.da_init_epoch
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy)
        return entropy_w

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        op_time = 0
        ss = time()
        for i, (v_d, v_p, labels, smiles, ids) in enumerate(self.train_dataloader):
            self.step += 1
            s = time()

            v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)
            self.optim.zero_grad()
            v_d, v_p, f, score, kg_score = self.model(v_d, v_p, smiles, ids)  # ++
            if self.n_class == 1:
                n, loss = RMSELoss(score, labels)
                kg_loss = kg_score.squeeze(dim=1).mean()
                loss = loss + kg_loss
            else:
                n, loss = cross_entropy_logits(score, labels)
            loss.backward()
            self.optim.step()
            e = time()
            op_time += (e - s)
            loss_epoch += loss.item()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", loss.item(), step=self.step)
        ee = time()
        print(f"Total time: {round(ee - ss, 2)}s")
        print(f"Total optimize time: {round(op_time, 2)}s")
        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch

    def train_da_epoch(self):
        self.model.train()
        total_loss_epoch = 0
        model_loss_epoch = 0
        da_loss_epoch = 0
        epoch_lamb_da = 0
        if self.current_epoch >= self.da_init_epoch:
            # epoch_lamb_da = self.da_lambda_decay()
            epoch_lamb_da = 1
            if self.experiment:
                self.experiment.log_metric("DA loss lambda", epoch_lamb_da, epoch=self.current_epoch)
        num_batches = len(self.train_dataloader)
        for i, (batch_s, batch_t) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            v_d, v_p, labels = batch_s[0].to(self.device), batch_s[1].to(self.device), batch_s[2].float().to(
                self.device)
            v_d_t, v_p_t = batch_t[0].to(self.device), batch_t[1].to(self.device)
            self.optim.zero_grad()
            self.optim_da.zero_grad()
            v_d, v_p, f, score = self.model(v_d, v_p)
            if self.n_class == 1:
                n, model_loss = binary_cross_entropy(score, labels)
            else:
                n, model_loss = cross_entropy_logits(score, labels)
            if self.current_epoch >= self.da_init_epoch:
                v_d_t, v_p_t, f_t, t_score = self.model(v_d_t, v_p_t)
                if self.da_method == "CDAN":
                    reverse_f = ReverseLayerF.apply(f, self.alpha)
                    softmax_output = torch.nn.Softmax(dim=1)(score)
                    softmax_output = softmax_output.detach()
                    if self.original_random:
                        random_out = self.random_layer.forward([reverse_f, softmax_output])
                        adv_output_src_score = self.domain_dmm(random_out.view(-1, random_out.size(1)))
                    else:
                        feature = torch.bmm(softmax_output.unsqueeze(2), reverse_f.unsqueeze(1))
                        feature = feature.view(-1, softmax_output.size(1) * reverse_f.size(1))
                        if self.random_layer:
                            random_out = self.random_layer.forward(feature)
                            adv_output_src_score = self.domain_dmm(random_out)
                        else:
                            adv_output_src_score = self.domain_dmm(feature)

                    reverse_f_t = ReverseLayerF.apply(f_t, self.alpha)
                    softmax_output_t = torch.nn.Softmax(dim=1)(t_score)
                    softmax_output_t = softmax_output_t.detach()
                    if self.original_random:
                        random_out_t = self.random_layer.forward([reverse_f_t, softmax_output_t])
                        adv_output_tgt_score = self.domain_dmm(random_out_t.view(-1, random_out_t.size(1)))
                    else:
                        feature_t = torch.bmm(softmax_output_t.unsqueeze(2), reverse_f_t.unsqueeze(1))
                        feature_t = feature_t.view(-1, softmax_output_t.size(1) * reverse_f_t.size(1))
                        if self.random_layer:
                            random_out_t = self.random_layer.forward(feature_t)
                            adv_output_tgt_score = self.domain_dmm(random_out_t)
                        else:
                            adv_output_tgt_score = self.domain_dmm(feature_t)

                    if self.use_da_entropy:
                        entropy_src = self._compute_entropy_weights(score)
                        entropy_tgt = self._compute_entropy_weights(t_score)
                        src_weight = entropy_src / torch.sum(entropy_src)
                        tgt_weight = entropy_tgt / torch.sum(entropy_tgt)
                    else:
                        src_weight = None
                        tgt_weight = None

                    n_src, loss_cdan_src = cross_entropy_logits(adv_output_src_score, torch.zeros(self.batch_size).to(self.device),
                                                                src_weight)
                    n_tgt, loss_cdan_tgt = cross_entropy_logits(adv_output_tgt_score, torch.ones(self.batch_size).to(self.device),
                                                                tgt_weight)
                    da_loss = loss_cdan_src + loss_cdan_tgt
                else:
                    raise ValueError(f"The da method {self.da_method} is not supported")
                loss = model_loss + da_loss
            else:
                loss = model_loss
            loss.backward()
            self.optim.step()
            self.optim_da.step()
            total_loss_epoch += loss.item()
            model_loss_epoch += model_loss.item()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", model_loss.item(), step=self.step)
                self.experiment.log_metric("train_step total loss", loss.item(), step=self.step)
            if self.current_epoch >= self.da_init_epoch:
                da_loss_epoch += da_loss.item()
                if self.experiment:
                    self.experiment.log_metric("train_step da loss", da_loss.item(), step=self.step)
        total_loss_epoch = total_loss_epoch / num_batches
        model_loss_epoch = model_loss_epoch / num_batches
        da_loss_epoch = da_loss_epoch / num_batches
        if self.current_epoch < self.da_init_epoch:
            print('Training at Epoch ' + str(self.current_epoch) + ' with model training loss ' + str(total_loss_epoch))
        else:
            print('Training at Epoch ' + str(self.current_epoch) + ' model training loss ' + str(model_loss_epoch)
                  + ", da loss " + str(da_loss_epoch) + ", total training loss " + str(total_loss_epoch) + ", DA lambda " +
                  str(epoch_lamb_da))
        return total_loss_epoch, model_loss_epoch, da_loss_epoch, epoch_lamb_da

    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_p, labels, smiles, ids) in enumerate(data_loader):


                v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)
                if dataloader == "val":
                    v_d, v_p, f, score, kg_score = self.model(v_d, v_p, smiles, ids)
                elif dataloader == "test":
                    v_d, v_p, f, score, kg_score = self.best_model(v_d, v_p, smiles, ids)
                if self.n_class == 1:
                    n, loss = RMSELoss(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()
        rmse = np.sqrt(mean_squared_error(y_label, y_pred))
        mae = mean_absolute_error(y_label, y_pred)
        test_loss = test_loss / num_batches
        r_p = scipy.stats.pearsonr(y_label, y_pred)[0]
        slope, intercept, _, _, _ = scipy.stats.linregress(y_pred, y_label)
        sd_error = np.array(y_label) - (intercept + slope * np.array(y_pred))
        sd = np.sqrt(np.sum(np.power(sd_error, 2)) / (len(y_label) - 1))

        if dataloader == "test":

            return rmse, mae, r_p, sd, test_loss
            
        else:
            return rmse, mae, r_p, sd, test_loss
