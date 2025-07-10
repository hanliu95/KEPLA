import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from dgllife.model.gnn import GCN, GAT, GraphSAGE
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
import numpy as np



def RMSELoss(pred_output, labels):
    # loss_fct = nn.MSELoss()
    loss_fct = nn.L1Loss()
    n = torch.squeeze(pred_output, 1)
    # return n, torch.sqrt(torch.mean((pred_output-labels)**2))
    loss = loss_fct(n, labels)
    # print(n)
    # print(labels)
    return n, loss


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class KEPLA(nn.Module):
    def __init__(self, **config):
        super(KEPLA, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]
        
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        # self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)
        self.fc = nn.Sequential(
                nn.Linear(2560, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU()
        )

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)
        self.kg = KGEModel(nentity=1000, nrelation=3, entity_dim=128, gamma=12.0) 

    def forward(self, bg_d, v_p, smiles, ids, mode="train"):
        v_d = self.drug_extractor(bg_d)
        # v_p = self.protein_extractor(v_p)
        # f, att = self.bcn(v_d, v_p)
        # v_d = torch.mean(v_d, dim=1)
        # v_p = torch.mean(v_p, dim=1)
        # print(v_d.shape)
        v_p = self.fc(v_p.float())
        v_p_global = torch.mean(v_p, dim=1)
        v_d_global = torch.mean(v_d, dim=1)
        attn_d = F.tanh(torch.matmul(v_d, v_p_global.unsqueeze(2)).transpose(1, 2)).div(128 ** 0.5)
        attn_d = F.softmax(attn_d, dim=2)
        v_d = torch.matmul(attn_d, v_d).squeeze(1)

        attn_p = F.tanh(torch.matmul(v_p, v_d_global.unsqueeze(2)).transpose(1, 2)).div(128 ** 0.5)
        attn_p = F.softmax(attn_p, dim=2)
        v_p = torch.matmul(attn_p, v_p).squeeze(1)
        
        f = torch.cat((v_d, v_p), 1)
        # f = torch.cat((v_d_global, v_p_global), 1)
        score = self.mlp_classifier(f)

        # kg_score = self.kg(v_d, v_p, smiles, ids)
        kg_score = self.kg(v_d_global, v_p_global, smiles, ids)

        # print(mode)
        if mode == "train":
            return v_d, v_p, f, score, kg_score
        elif mode == "eval":
            return v_d, v_p, score, kg_score, attn_d, attn_p, v_d_global, v_p_global


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        # self.gnn = GraphSAGE(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats




class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


# newly added KG module
class KGEModel(nn.Module):
    def __init__(self, nentity, nrelation, entity_dim, gamma):

        super(KGEModel, self).__init__()

        self.nentity = nentity
        self.nrelation = nrelation
        self.entity_dim = entity_dim

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.entity_embedding = nn.Embedding(nentity, entity_dim)
        self.relation_embedding = nn.Embedding(nrelation, entity_dim)

        self.mol_entity_embedding = nn.Embedding(175, entity_dim)
        self.mol_relation_embedding = nn.Embedding(1, entity_dim)

        self.mol_feat_entity_embedding = nn.Embedding(23, entity_dim)
        self.mol_feat_relation_embedding = nn.Embedding(1, entity_dim)

        self.KG_dict = np.load('../datasets/pdbbind/KG_dict_1.npy', allow_pickle=True).item()
        self.mol_KG_dict = np.load('../datasets/pdbbind/mol_num_KG_dict.npy', allow_pickle=True).item()
        self.mol_feat_KG_dict = np.load('../datasets/pdbbind/mol_feat_KG_dict.npy', allow_pickle=True).item()
        self.fc1 = nn.Linear(entity_dim, entity_dim)

    def forward(self, v_d, v_p, smiles, ids):
        h_index = []
        r_index = []
        t_index = []
        v_d = self.fc1(v_d)
        # x = self.encoder(x)
        # x = torch.mean(x, dim=1)
        count = 0
        for id in ids:
            if id in self.KG_dict.keys():
                r_t = self.KG_dict[id]
                r_t = np.array(r_t)
                h_index.extend([count] * len(r_t))
                r_index.extend(r_t[:, 0].tolist())
                t_index.extend(r_t[:, 1].tolist())
            count += 1
        h_index = torch.LongTensor(h_index).cuda()
        r_index = torch.LongTensor(list(map(int, r_index))).cuda()
        t_index = torch.LongTensor(list(map(int, t_index))).cuda()
        
        head = torch.index_select(v_p, 0, h_index).unsqueeze(1)
        relation = self.relation_embedding(r_index).unsqueeze(1)
        tail = self.entity_embedding(t_index).unsqueeze(1)

        mol_h_index = []
        mol_r_index = []
        mol_t_index = []
        mol_feat_h_index = []
        mol_feat_r_index = []
        mol_feat_t_index = []
        count = 0
        for smi in smiles:
            r_t = self.mol_KG_dict[smi]
            r_t = np.array(r_t)
            mol_h_index.extend([count] * len(r_t))
            mol_r_index.extend(r_t[:, 0].tolist())           
            mol_t_index.extend(r_t[:, 1].tolist())

            feat_rt = self.mol_feat_KG_dict[smi]
            feat_rt = np.array(feat_rt)
            mol_feat_h_index.extend([count] * len(feat_rt))
            mol_feat_r_index.extend(feat_rt[:, 0].tolist())
            mol_feat_t_index.extend(feat_rt[:, 1].tolist())
            
            count += 1
        mol_h_index = torch.LongTensor(mol_h_index).cuda()
        mol_r_index = torch.LongTensor(list(map(int, mol_r_index))).cuda()
        mol_t_index = torch.LongTensor(list(map(int, mol_t_index))).cuda()

        mol_head = torch.index_select(v_d, 0, mol_h_index).unsqueeze(1)
        mol_relation = self.mol_relation_embedding(mol_r_index).unsqueeze(1)
        mol_tail = self.mol_entity_embedding(mol_t_index).unsqueeze(1)

        mol_feat_h_index = torch.LongTensor(mol_feat_h_index).cuda()
        mol_feat_r_index = torch.LongTensor(list(map(int, mol_feat_r_index))).cuda()
        mol_feat_t_index = torch.LongTensor(list(map(int, mol_feat_t_index))).cuda()

        mol_feat_head = torch.index_select(v_d, 0, mol_feat_h_index).unsqueeze(1)
        mol_feat_relation = self.mol_feat_relation_embedding(mol_feat_r_index).unsqueeze(1)
        mol_feat_tail = self.mol_feat_entity_embedding(mol_feat_t_index).unsqueeze(1)

        kg_score = self.TransE(head, relation, tail, mode='single')
        mol_kg_score = self.TransE(mol_head, mol_relation, mol_tail, mode='head-batch')
        mol_feat_kg_score = self.TransE(mol_feat_head, mol_feat_relation, mol_feat_tail, mode='head-batch')
        kg_score = torch.cat((kg_score, mol_kg_score * 0.1, mol_feat_kg_score * 0.1), 0)
        return kg_score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + relation - tail
            score = torch.norm(score, p=2, dim=2)
        else:
            score = head * relation - tail
            score = torch.norm(score, p=1, dim=2)
        return score





