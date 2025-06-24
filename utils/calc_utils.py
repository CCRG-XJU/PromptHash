import torch
import visdom
import time
import numpy as np

class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name, opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        """
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

def calc_neighbor(a: torch.Tensor, b: torch.Tensor):
    return (a.matmul(b.transpose(0, 1)) > 0).float()


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def calc_map_k(qB, rB, query_label, retrieval_label, k=None):
    num_query = query_label.shape[0]
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map

def calc_map_k_fast(qB, rB, query_label, retrieval_label, k=None):
    # qB: [num_query, hash_bits], rB: [num_retrieval, hash_bits]
    # query_label: [num_query, num_classes], retrieval_label: [num_retrieval, num_classes]
    # num_query = query_label.shape[0]
    if k is None:
        k = retrieval_label.shape[0]

    # 1. 计算ground truth矩阵 [num_query, num_retrieval]
    gnd = (query_label @ retrieval_label.t()) > 0  # bool tensor on GPU

    # 2. 计算所有query和retrieval的hamming距离 [num_query, num_retrieval]
    q = rB.shape[1]
    distH = 0.5 * (q - qB @ rB.t())  # float tensor on GPU

    # 3. 得到排序索引
    _, ind = torch.sort(distH, dim=1)  # [num_query, num_retrieval]

    # 4. 对gnd按距离排序
    gnd_sorted = torch.gather(gnd.float(), 1, ind)

    # 5. 计算AP
    # 计算每个query的relevant数目
    relevant_num = gnd.sum(dim=1)  # [num_query]
    # 只保留前k
    gnd_sorted_k = gnd_sorted[:, :k]
    # 累加relevant
    relevant_cumsum = torch.cumsum(gnd_sorted_k, dim=1)
    # 生成位置索引
    pos = torch.arange(1, k + 1, device=qB.device).float().view(1, -1)
    # 计算precision
    precision = relevant_cumsum / pos
    # 只在relevant位置取precision
    precision = precision * gnd_sorted_k
    # 计算每个query的AP
    ap = precision.sum(dim=1) / torch.clamp(relevant_num, min=1)
    # mAP
    map = ap.mean()
    return map
