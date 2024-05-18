import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
from recbole.model.cdib_layers import Net1, DNN, GaussianDiffusion
from torch.distributions.multivariate_normal import MultivariateNormal

class CDIB(SequentialRecommender):
    def __init__(self, config, dataset):
        super(CDIB, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['dropout_prob']
        self.attn_dropout_prob = config['dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.steps_f = config['steps_forward']
        self.steps_i = config['steps_inference']

        self.batch_size = config['train_batch_size']
        self.lmd_elbo = config['lmd_elbo']
        self.lmd_preLoss = config['lmd_pl']
        self.lmd_clLoss1 = config['lmd_cl1']
        self.lmd_clLoss2 = config['lmd_cl2']
        self.lmd_augRec = config['lmd_agr']
        self.lmd_u = config['lmd_u']
        self.pow = config['pow']
        self.tau = config['tau']
        self.scale = config['scale']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.n_users = dataset.user_num

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.user_embedding = nn.Embedding(self.n_users, self.hidden_size, padding_idx=0)
        self.mu = nn.Linear(self.hidden_size, 1)
        self.logvar = nn.Linear(self.hidden_size, 1)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.attacker = Net1(self.scale)
        self.model = DNN([self.hidden_size, 2*self.inner_size], [2*self.inner_size, self.hidden_size], 
                         self.hidden_size, act_func=self.hidden_act, dropout=self.hidden_dropout_prob)
        self.gd =  GaussianDiffusion(steps=self.steps_f)
        self.trm_encoder = TransformerEncoder(n_layers=self.n_layers, n_heads=self.n_heads,
                                              hidden_size=self.hidden_size, inner_size=self.inner_size,
                                              hidden_dropout_prob=self.hidden_dropout_prob, attn_dropout_prob=self.attn_dropout_prob,
                                              hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps)
        self.LayerNorm1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm2 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm3 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm4 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNorm5 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.softplus = nn.Softplus()
        self.item_counter = self.init_item_count(dataset)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss(reduction='none')
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.nce_fct = nn.CrossEntropyLoss(reduction='none')

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def init_item_count(self, dataset):
        item_count = dataset.item_counter
        i_c = torch.empty(self.n_items, device=self.device)
        tot = 0
        for i in range(self.n_items):
            i_c[i] = item_count[i]
            tot += item_count[i]
        try:
            i_c = i_c / tot
            i_c = torch.pow(i_c, self.pow)
        except:
            i_c = torch.ones(self.n_items, device=self.device)
        return i_c

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask


    def forward(self, item_seq, item_seq_len, x=None):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        if x is None:
            item_emb = self.item_embedding(item_seq)
        else:
            item_emb = x
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm1(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def generate(self, item_seq, user_seq):
        item_emb = self.item_embedding(item_seq)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        input_emb = item_emb + position_embedding 
        input_emb = self.LayerNorm2(input_emb)
        input_emb = self.dropout(input_emb)
        input_emb = input_emb * (item_seq > 0).unsqueeze(-1)
        user_emb = self.user_embedding(user_seq)
        seq_cond = (input_emb.sum(1, keepdim=True)) / ((item_seq > 0).sum(-1, keepdim=True).unsqueeze(-1).repeat(1, item_seq.size(1), 1))
        seq_cond = self.LayerNorm3(seq_cond)
        seq_cond = self.dropout(seq_cond)
        user_cond = user_emb.unsqueeze(1).repeat(1, item_seq.size(1), 1)
        user_cond = self.LayerNorm4(user_cond)
        user_cond = self.dropout(user_cond)
        M_mask = self.attacker(input_emb, seq_cond, user_cond)
        M_mask = M_mask * (item_seq > 0).unsqueeze(-1)
        loss_pre = (M_mask.sum(-1) / (item_seq > 0).sum(-1, keepdim=True).unsqueeze(-1)).mean()
        input_emb_se = input_emb * M_mask  # [bs, seqlen, hidden_size] * [bs, seqlen, 1]
        input_emb_gu = input_emb * (1. - M_mask)
        term = self.gd.training_losses(self.model, input_emb_se, item_seq, M_mask, False)
        elbo = term['loss'].mean()
        latent_recon = self.gd.p_sample(self.model, input_emb_se, self.steps_i, M_mask, False)  # 5 [bs, seq, dim]
        M_com = latent_recon + input_emb_gu
        return elbo, M_com, -loss_pre, M_mask
    
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        user_seq = interaction[self.USER_ID]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        elbo, M_com, loss_pre, _ = self.generate(item_seq, user_seq)
        user_emb = self.user_embedding(user_seq)
        user_emb = self.LayerNorm5(user_emb)
        user_emb = self.dropout(user_emb)
        seq_output_ori, seq_output_aug = self.forward(item_seq, item_seq_len), self.forward(item_seq, item_seq_len, M_com)

        nce_logits, nce_labels = self.info_nce(
            seq_output_ori, seq_output_aug, temp=self.tau, batch_size=item_seq.shape[0])
        cl_loss = self.nce_fct(nce_logits, nce_labels).mean()

        nce_logits_u, nce_labels_u = self.info_nce(
            user_emb, seq_output_ori, temp=self.tau, batch_size=item_seq.shape[0])
        cl_loss_u = self.nce_fct(nce_logits_u, nce_labels_u).mean()

        test_item_emb = self.item_embedding.weight
        logits_ori = torch.matmul(seq_output_ori, test_item_emb.transpose(0, 1))
        logits_aug = torch.matmul(seq_output_aug, test_item_emb.transpose(0, 1))

        rec_loss_ori = self.loss_fct(logits_ori, pos_items)
        rec_loss_aug = self.loss_fct(logits_aug, pos_items)
        rec_loss = (rec_loss_ori + rec_loss_aug).mean()
        
        p0 = torch.gather(self.item_counter, -1, pos_items)
        mu = self.mu(user_emb).squeeze(1)
        logvar = self.logvar(user_emb).squeeze(1)
        prior_loc = p0
        prior_cov = torch.eye(p0.shape[0]).to(self.device)
        prior = MultivariateNormal(prior_loc, prior_cov)
        p0_h = MultivariateNormal(mu, torch.diag_embed(self.softplus(logvar)))
        loss_u = torch.distributions.kl.kl_divergence(p0_h, prior)
        loss_u = torch.where(torch.isfinite(loss_u), loss_u, torch.zeros_like(loss_u))
        loss_u = loss_u.mean()
        
        return self.lmd_elbo*elbo, self.lmd_preLoss*loss_pre, self.lmd_clLoss1*cl_loss_u, -self.lmd_clLoss2*cl_loss, self.lmd_u*loss_u, rec_loss

    @staticmethod
    def mask_correlated_samples(batch_size):
        """
        correlated sample means the augment samples come from the same naive sample.
        """
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N - 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores