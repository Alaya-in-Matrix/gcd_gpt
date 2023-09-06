import warnings
warnings.filterwarnings('ignore', category = UserWarning)
import mlflow
import fire
import numpy as np
import torch
import torch.nn as nn
import re
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import trange, tqdm
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import TensorDataset, DataLoader


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

with open('./toy_sentences', 'r', encoding = 'utf-8') as f:
    all_sentences = f.readlines()

sentences = []
vocab     = {'<sos>', '<eos>', '<pad>'}
max_len   = 0


n_tokens = 0
tbar = tqdm(all_sentences)
for sen in tbar:
    words = sen.strip().split()
    vocab.update(words)
    sentences.append(['<sos>'] + words + ['<eos>'])
    max_len = max(max_len, 1 + len(words))
    n_tokens += len(words)

print(f'Number of tokens: {n_tokens}')
n_voc   = len(vocab)
le      = LabelEncoder().fit(list(vocab))
sen_in  = []
sen_out = []

sos_value, eos_value, padding_value, eq_value = le.transform(['<sos>', '<eos>', '<pad>', '=']).tolist()


expr_mask = []
for sen in tqdm(sentences):
    s1      = sen[:-1]
    s2      = sen[1:]
    num_pad = max_len - len(s1)
    s1      = le.transform(s1 + ['<pad>'] * num_pad)
    s2      = le.transform(s2 + ['<pad>'] * num_pad)
    eq_idx  = np.where(s1 == eq_value)[0][0]
    mask    = np.full((max_len, max_len), False)
    mask[eq_idx + 1:]    = True
    mask[:, eq_idx + 1:] = True
    expr_mask.append(mask)
    sen_in.append(s1)
    sen_out.append(s2)


tr_in, tst_in, tr_out, tst_out, tr_mask, tst_mask = train_test_split(np.array(sen_in), np.array(sen_out), np.array(expr_mask), test_size = 1024, shuffle = True, random_state = 42)

ds_tr  = TensorDataset(torch.LongTensor(tr_in), torch.LongTensor(tr_out), torch.tensor(tr_mask).bool())
ds_tst = TensorDataset(torch.LongTensor(tst_in), torch.LongTensor(tst_out), torch.tensor(tst_mask).bool())


class GPT(nn.Module):
    def __init__(self, emb_dim, nheads, num_layers, n_voc, max_len):
        super().__init__()
        self.emb = nn.Embedding(n_voc, emb_dim)
        self.pos = nn.Embedding(max_len, emb_dim)
        self.decoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(emb_dim, nheads, batch_first = True),
                num_layers = num_layers,
                )
        self.out     = nn.Linear(emb_dim, n_voc)
        self.nheads  = nheads
        self.max_len = max_len

    def forward(self, prefix, attn_mask):
        len          = prefix.shape[1]
        emb          = self.emb(prefix) + self.pos.weight[:len]
        mask         = (nn.Transformer.generate_square_subsequent_mask(len) < 0).to(device)
        src_mask     = torch.repeat_interleave(attn_mask[:, :len, :len], self.nheads, dim = 0) & mask
        padding_mask = (prefix == padding_value).to(device).to(torch.bool)
        out          = self.decoder(emb, src_mask, padding_mask)
        return self.out(out)


def validate_batch(step, model, ds, batch_size, strategy = 'greedy', prefix_tok = None):
    dl_tst   = DataLoader(ds, batch_size = batch_size, shuffle = False)
    all_y    = []
    all_pred = []
    for bx, by, bmask in tqdm(dl_tst):
        prefix       = bx.clone().to(device)
        bmask        = bmask.to(device)
        i1, curr_idx = torch.where(prefix == eq_value)
        indicator    = (curr_idx >= max_len - 1) | (prefix[i1, curr_idx] == eos_value)
        while not indicator.all():
            out = model(prefix[~indicator], bmask[~indicator]).argmax(dim = -1)
            prefix[i1[~indicator], curr_idx[~indicator]+1] = out[torch.arange(out.shape[0]), curr_idx[~indicator]] # XXX
            curr_idx  += 1
            indicator  = (curr_idx >= max_len - 1) | (prefix[i1, curr_idx.clamp(max = max_len - 1)] == eos_value)
        all_y.append(by.cpu())
        all_pred.append(prefix.cpu())
        break
    all_y    = torch.cat(all_y, dim = 0).numpy()
    all_pred = torch.cat(all_pred, dim = 0).numpy()
    all_y    = le.inverse_transform(all_y.reshape(-1)).reshape(all_y.shape)
    all_pred = le.inverse_transform(all_pred.reshape(-1)).reshape(all_y.shape)

    acc_lst  = []
    mape_lst = []
    stat     = []
    for i in range(all_y.shape[0]):
        true_expr = ''.join([s for s in all_y[i] if s not in ['<sos>', '<eos>', '<pad>']]).replace('-=-', '=').split('=')
        pred_expr = ''.join([s for s in all_pred[i] if s not in ['<sos>', '<eos>', '<pad>']]).replace('-=-', '=').split('=')

        mape = 0.
        if true_expr[1] != pred_expr[1]:
            try:
                true_result = eval(true_expr[1])
                pred_result = eval(pred_expr[1])

                if true_result == 0:
                    mape = 100.
                else:
                    mape = 100 * abs(true_result - pred_result) / abs(1e-9 + true_result)
            except (ValueError, SyntaxError, OverflowError, NameError):
                mape = 100.

        print(true_expr[0].replace('-', ''))
        print(true_expr[1])
        print(pred_expr[1])
        print(f'{prefix_tok}_mape = {mape:.2f}%')
        print('----------')


        mape_lst.append(mape)
        acc_lst.append(mape == 0)
    mape_lst    = np.array(mape_lst)
    acc         = 100 * np.mean(acc_lst)
    mape_avg    = np.mean(mape_lst)
    mape_median = np.quantile(mape_lst, 0.5)
    mape_95     = np.quantile(mape_lst, 0.95)
    mape_75     = np.quantile(mape_lst, 0.75)
    mape_25     = np.quantile(mape_lst, 0.25)
    mape_05     = np.quantile(mape_lst, 0.05)
    print(f'{prefix_tok}_ACC = {acc:.2f}%')
    print(f'{prefix_tok}_MAPE_avg = {mape_avg:.2f}%')
    print(f'{prefix_tok}_MAPE_median = {mape_median:.2f}%')
    print(f'{prefix_tok}_MAPE_95 = {mape_95:.2f}%')
    print(f'{prefix_tok}_MAPE_75 = {mape_75:.2f}%')
    print(f'{prefix_tok}_MAPE_25 = {mape_25:.2f}%')
    print(f'{prefix_tok}_MAPE_05 = {mape_05:.2f}%')
    if len(stat) > 0:
        stat = np.array(stat)
        r2   = r2_score(stat[:, 1], stat[:, 0])
        plt.figure(figsize = (8, 8))
        plt.xscale('symlog')
        plt.yscale('symlog')
        plt.plot(stat[:, 0], stat[:, 1], '+')
        plt.plot(stat[:, 0], stat[:, 0])
        plt.title(f'Step {step}, ACC = {acc:.2f}%, R2 = {r2:.2f}, MAPE = {mape_median:.2f}%')
        plt.savefig(f'./plot/show_{step:03d}.png')
        plt.close()
        print(f'Step {step}, ACC = {acc:.2f}%, R2 = {r2:.2f}, MAPE = {mape_median:.2f}%', flush = True)


def validate(step, model, ds, batch_size, strategy = 'greedy'):
    dl_tst = DataLoader(ds, batch_size = batch_size, shuffle = True)
    bx, by, bmask = next(iter(dl_tst))

    bmask = bmask.to(device)
    stat  = []
    with torch.no_grad():
        mape_lst = []
        acc_lst  = []
        for i in range(bx.shape[0]):
            sen    = bx[i]
            eq_pos = torch.where(sen == eq_value)[0].item()
            prefix = bx[i, :eq_pos + 1].clone().view(1, -1).to(device)
            out    = model(prefix, bmask[[i]]).argmax(dim = -1)
            while (not (out == eos_value).any()) and prefix.shape[1] < max_len:
                prefix = torch.cat([prefix, out[:, [-1]]], dim = 1)
                out    = model(prefix, bmask[[i]]).argmax(dim = -1)

            sen_true = le.inverse_transform(bx[i].cpu().numpy())
            sen_pred = le.inverse_transform(out.view(-1).cpu().numpy())

            sen_true = ''.join([w for w in sen_true.tolist() if w not in ['<sos>', '<eos>', '<pad>']])
            sen_pred = ''.join([w for w in sen_pred.tolist() if w not in ['<sos>', '<eos>', '<pad>']])

            expr, result = sen_true.split('=')
            try:
                pred = sen_pred.split('=')[1]
            except IndexError:
                pred = 'error'

            mape = 0.
            if pred != result:
                try:
                    mape = 100 * abs(float(result) - float(pred)) / abs(float(result))
                    if np.isnan(mape):
                        mape = 100.
                except (ValueError, ZeroDivisionError):
                    mape = 100.

            mape_lst.append(mape)
            acc_lst.append(pred == result)

            try:
                stat.append([float(pred), float(result)])
            except ValueError:
                pass

            if pred == result:
                print(i)
                print(expr)
                print(result)
                print(pred)
                print(f'MAPE = {mape:.2f}%')
                print('-------')
        mape_lst    = np.array(mape_lst)
        acc         = 100 * np.mean(acc_lst)
        mape_avg    = np.mean(mape_lst)
        mape_median = np.quantile(mape_lst, 0.5)
        mape_95     = np.quantile(mape_lst, 0.95)
        mape_75     = np.quantile(mape_lst, 0.75)
        mape_25     = np.quantile(mape_lst, 0.25)
        mape_05     = np.quantile(mape_lst, 0.05)
        print(f'ACC = {acc:.2f}%')
        print(f'MAPE_avg = {mape_avg:.2f}%')
        print(f'MAPE_median = {mape_median:.2f}%')
        print(f'MAPE_95 = {mape_95:.2f}%')
        print(f'MAPE_75 = {mape_75:.2f}%')
        print(f'MAPE_25 = {mape_25:.2f}%')
        print(f'MAPE_05 = {mape_05:.2f}%')

        if len(stat) > 0:
            stat = np.array(stat)
            r2   = r2_score(stat[:, 1], stat[:, 0])
            plt.figure(figsize = (8, 8))
            plt.xscale('symlog')
            plt.yscale('symlog')
            plt.plot(stat[:, 1], stat[:, 0], '+')
            plt.plot(stat[:, 1], stat[:, 1])
            plt.title(f'Step {step}, ACC = {acc:.2f}%, R2 = {r2:.2f}, MAPE = {mape_median:.2f}%')
            plt.savefig(f'./plot/show_{step:03d}.png')
            plt.close()



def exp(
        batch_size  = 4096,
        num_epochs  = 900,
        lr          = 1e-4,
        emb_dim     = 512,
        nheads      = 8,
        num_layers  = 6,
        start_epoch = None,
        save_every  = 10,
        checkpoints_dir = None,
        ):
    with mlflow.start_run():
        mlflow.log_params({
            'batch_size' : batch_size,
            'num_epochs' : num_epochs,
            'lr'         : lr,
            'emb_dim'    : emb_dim,
            'nheads'     : nheads,
            'num_layers' : num_layers,
            })

        if checkpoints_dir is None:
            checkpoints_dir = './checkpoints'
        Path(checkpoints_dir).mkdir(exist_ok = True)
        Path('./plot').mkdir(exist_ok = True)
        dl_tr = DataLoader(ds_tr, batch_size = batch_size, shuffle = True)
        gpt   = GPT(emb_dim    = emb_dim,
                    nheads     = nheads,
                    num_layers = num_layers,
                    n_voc      = n_voc,
                    max_len    = max_len
                    ).to(device)
        print(gpt)
        n_param = 0
        for p in gpt.parameters():
            n_param += p.numel()
        print(f'Number of parameters: {n_param}')
        opt  = torch.optim.AdamW(gpt.parameters(), lr = lr, weight_decay = 1e-3)
        sch  = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr          = lr,
                epochs          = num_epochs,
                steps_per_epoch = 1 + len(ds_tr) // batch_size,
                )
        crit = nn.CrossEntropyLoss()
        tbar = trange(num_epochs)

        for epoch in tbar:
            epoch_loss = 0.
            epoch_cnt  = 1e-6
            gpt.train()
            tbar_dl = tqdm(dl_tr)
            for bx, by, bmask in tbar_dl:
                bx    = bx.to(device)
                by    = by.to(device)
                bmask = bmask.to(device)
                out   = gpt(bx, bmask)
                loss  = crit(out.view(-1, n_voc), by.view(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sch.step()

                epoch_loss += loss * by.shape[0]
                epoch_cnt  += by.shape[0]
                tbar_dl.set_description(f'loss = {loss.item():.3f}')
            epoch_loss /= epoch_cnt
            tbar.set_description(f'Epoch {epoch + 1}, loss = {epoch_loss.item():.3e}, lr = {sch.get_last_lr()[-1]:.3e}')

            gpt.eval()
            with torch.no_grad():
                validate_batch(epoch, gpt, ds_tr,   batch_size = min(8192, batch_size), strategy = 'greedy', prefix_tok = 'tr')
                validate_batch(epoch, gpt, ds_tst,  batch_size = min(8192, batch_size), strategy = 'greedy', prefix_tok = 'tst')

            if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
                torch.save(gpt, f'{checkpoints_dir}/model_{epoch + 1}')


if __name__ == '__main__':
    fire.Fire(exp)
