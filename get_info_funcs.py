import torch
import numpy as np
from torch import nn
from gpytorch.utils.lanczos import lanczos_tridiag
from collections import defaultdict
from train import cross_entropy


def Rop(ys, xs, vs):
    if isinstance(ys, tuple):
        ws = [torch.zeros_like(y, requires_grad=True) for y in ys]
    else:
        ws = torch.zeros_like(ys, requires_grad=True)

    gs = torch.autograd.grad(ys, xs, grad_outputs=ws, create_graph=True)
    re = torch.autograd.grad(gs, ws, grad_outputs=vs)
    return tuple([j.detach() for j in re])

def Lop(ys, xs, ws):
    vJ = torch.autograd.grad(ys, xs, grad_outputs=ws)
    return tuple([j.detach() for j in vJ])

def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        n = tensor.numel()
        outList.append(vector[i : i + n].view(tensor.shape))
        i += n
    return outList

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def Hvp(vec, params, outputs, data_size, targets, criterion=cross_entropy, **kwargs):
    "Returns Hessian vec. prod."
    #criterion = criterion or nn.CrossEntropyLoss(reduction='sum')
    loss = criterion(None, outputs, targets, reduction='sum') / data_size
    grad = torch.autograd.grad(loss, params, create_graph=True)

    # Compute inner product of gradient with the direction vector
    prod = 0.
    for (g, v) in zip(grad, vec):
        prod += (g * v).sum()

    # Compute the Hessian-vector product, H*v
    Hv = torch.autograd.grad(prod, params)
    return Hv

def Fvp(vec, params, outputs, data_size, **kwargs):
    "Returns Fisher vec. prod."
    Jv, = Rop(outputs, params, vec)
    batch, dims = outputs.size(0), outputs.size(1)
    probs = torch.softmax(outputs, dim=1)
    
    M = torch.zeros(batch, dims, dims, device=probs.device)
    M.view(batch, -1)[:, ::dims + 1] = probs
    H = M - torch.einsum('bi,bj->bij', (probs, probs))
    
    HJv = torch.squeeze(H @ torch.unsqueeze(Jv, -1), -1) / data_size
    JHJv = Lop(outputs, params, HJv)
    return JHJv

def eval_mvp(Mvp, vec, params, net, dataloader, **kwargs):
    M_v = [torch.zeros_like(p) for p in params]
    data_size = len(dataloader.dataset)

    for inputs, targets in dataloader:
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = net(inputs)

        kwargs['targets'] = targets
        for i, v in enumerate(Mvp(vec, params, outputs, data_size, **kwargs)):
            M_v[i] += v
                
    return M_v


def lanczos_tridiag_to_diag(t_mat):
    orig_device = t_mat.device
    
    if t_mat.size(-1) < 32:
        retr = torch.symeig(t_mat.cpu(), eigenvectors=True)
    else:
        retr = torch.symeig(t_mat, eigenvectors=True)

    evals, evecs = retr
    return evals.to(orig_device), evecs.to(orig_device)

def eval_eigs(model, dataloader, fisher=True, train_mode=False, nsteps=10, return_evecs=False, pnames=None,
              criterion=cross_entropy):
    """
        Get "nsteps" approximate eigenvalues of the Fisher or Hessian marix via Lanczos.
        Args:
            model: the trained model.
            dataloader: dataloader for the dataset, may use a subset of it.
            fisher: whether to use Fisher MVP or Hessian MVP
            train_mode: whether to run the model in train mode.
            nsteps: number of lanczos steps to perform
            return_evecs: whether to return evecs as well
            pnames: names of parameters to consider (if None all params are used)
        Returns:
            eigvals: calculated approximate eigenvalues
            eigvecs: calculated approximate eigenvectors, optional
    """
    if train_mode:
        model.train()
    else:
        model.eval()
        
    kwargs = {}
    if fisher:
        Mvp = Fvp
    else:
        Mvp = Hvp
        #criterion = nn.CrossEntropyLoss(reduction='sum')
        kwargs['criterion'] = criterion

    params = list(model.parameters()) if pnames is None else [p for n, p in model.named_parameters() if n in pnames]
    N = sum(p.numel() for p in params)
    
    def lanczos_mvp(vec):
        vec = unflatten_like(vec.view(-1), params)
        fvp = eval_mvp(Mvp, vec, params, model, dataloader, **kwargs)
        return flatten(fvp).unsqueeze(1)
    
    # use lanczos to get the t and q matrices out
    q_mat, t_mat = lanczos_tridiag(
        lanczos_mvp,
        nsteps,
        device=params[0].device,
        dtype=params[0].dtype,
        matrix_shape=(N, N),
    )
    
    # convert the tridiagonal t matrix to the eigenvalues
    eigvals, eigvecs = lanczos_tridiag_to_diag(t_mat)
    eigvecs = q_mat @ eigvecs if return_evecs else None
    return eigvals, eigvecs

def eval_trace(model, dataloader, fisher=True, train_mode=False, n_vecs=5, pnames=None,
               criterion=cross_entropy):
    "Returns Fisher or Hessian traces divided by number of parameters."
    if train_mode:
        model.train()
    else:
        model.eval()
        
    kwargs = {}
    if fisher:
        Mvp = Fvp
    else:
        Mvp = Hvp
        #criterion = nn.CrossEntropyLoss(reduction='sum')
        kwargs['criterion'] = criterion

    trace = 0.0
    #params = list(model.parameters()) if pnames is None else [p for n, p in model.named_parameters() if n in pnames]
    if pnames is None:
        params = list([param for param in model.parameters() if param.requires_grad == True])
    else:
        params = [p for n, p in model.named_parameters() if (n in pnames and p.requires_grad == True)]
        
    N = sum(p.numel() for p in params)
    
    for _ in range(n_vecs):
        vec = torch.randn(N, device=params[0].device)
        vec /= torch.norm(vec)
        vec = unflatten_like(vec, params)
        M_v = eval_mvp(Mvp, vec, params, model, dataloader, **kwargs)
        for m, v in zip(M_v, vec):
            trace += (m * v).sum().item()
    
    return trace / n_vecs

def eval_model(model, dataloader, criterion, train_mode=False, return_mean=True):
    "Evaluate a model on test or train set: compute CEL and accuracy."
    if train_mode:
        model.train()
    else:
        model.eval()
    
    loss = 0.0
    correct = 0
    
    #criterion = nn.CrossEntropyLoss(reduction="sum")
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            labels = labels.cuda()
            
            loss_batch, outputs = criterion(model, images, labels, reduction='sum')
            predicted = outputs.argmax(-1)
            
            loss += loss_batch.item()
            correct += (predicted == labels).sum().item()

    if return_mean:
        loss /= len(dataloader.dataset)
        correct /= len(dataloader.dataset)
    
    return loss, correct 

def calc_grads(model, dataloader, criterion, train_mode=False, return_numpy=False, pnames=None):
    "Calculate batch gradients of the model."
    if train_mode:
        model.train()
    else:
        model.eval()

    if pnames is None:
        params = list([param for param in model.parameters() if param.requires_grad == True])
    else:
        params = [p for n, p in model.named_parameters() if (n in pnames and p.requires_grad == True)]
        
    grads_list = []
    #criterion = nn.CrossEntropyLoss()
    
    for images, labels in dataloader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = criterion(None, outputs, labels)
        grads = torch.autograd.grad(loss, params)
        if return_numpy:
            grads = np.concatenate([g.cpu().numpy().ravel() for g in grads])
        grads_list.append(grads)

    return grads_list

def get_batch_outliers_gnorms(model, dataloader, criterion, train_mode=False, pnames=None):
    if train_mode:
        model.train()
    else:
        model.eval()

    if pnames is None:
        pnames = [n for n, p in model.named_parameters() if p.requires_grad]

    params = [p for n, p in model.named_parameters() if n in pnames]
    #criterion = nn.CrossEntropyLoss()
    gnorms_list = []
    outliers_list = []

    for images, labels in dataloader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        predicted = outputs.argmax(-1)
        loss = criterion(None, outputs, labels)

        grads = torch.autograd.grad(loss, params)
        gnorms_list.append(sum((g ** 2).sum().item() for g in grads) ** 0.5)

        outliers = torch.zeros_like(outputs[0])
        incorrect = (predicted != labels)
        if sum(incorrect) > 0:
            inc_labels = labels[incorrect]
            outliers.put_(inc_labels, torch.ones_like(inc_labels, dtype=outliers.dtype), accumulate=True)
        outliers_list.append(outliers.cpu().numpy())

    return np.array(outliers_list), np.array(gnorms_list)

def get_batch_outliers(model, dataloader, train_mode):
    if train_mode:
        model.train()
    else:
        model.eval()

    outliers_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            predicted = outputs.argmax(-1)

            outliers = torch.zeros_like(outputs[0])
            incorrect = (predicted != labels)
            if sum(incorrect) > 0:
                inc_labels = labels[incorrect]
                outliers.put_(inc_labels, torch.ones_like(inc_labels, dtype=outliers.dtype), accumulate=True)
            outliers_list.append(outliers.cpu().numpy())

    return np.array(outliers_list)


def calc_probs_stats(model, dataloader, train_mode=False):
    "Calculate probs statistics per each correct class (mean, std, min, max, median)."
    if train_mode:
        model.train()
    else:
        model.eval()

    stats = ["mean", "std", "min", "max", "median"]
    p_stats = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            outputs = model(images)
            p = torch.softmax(outputs, -1).cpu().numpy()

            p_stat = []

            for c in range(outputs.size(-1)):
                c_mask = labels == c
                p_c = p[c_mask]
                p_stat.append([getattr(np, stat)(p_c, axis=0) for stat in stats])

            p_stats.append(p_stat)

    return np.array(p_stats).transpose(2, 0, 1, 3)  # n_stats x n_batches x C x C

def calc_grads_norms(model, dataloader, criterion, bs_list, train_mode=False, pnames=None):
    """Calculate gradients norms of the model for each batch size in bs_list. 
    Dataloader batch size must be 1.
    """
    if train_mode:
        model.train()
    else:
        model.eval()

    params = list(model.parameters()) if pnames is None else [p for n, p in model.named_parameters() if n in pnames]
    grads_dict = dict((bs, [torch.zeros_like(p) for p in params]) for bs in bs_list)
    grads_norms_dict = defaultdict(list)
    #criterion = nn.CrossEntropyLoss()
    
    for i, (image, label) in enumerate(dataloader, 1):
        image = image.cuda()
        label = label.cuda()

        output = model(image)
        loss = criterion(None, output, label)
        grad = torch.autograd.grad(loss, params)

        for bs in bs_list:
            for g_batch, g in zip(grads_dict[bs], grad):
                g_batch += g

            if (i % bs == 0) or (i == len(dataloader)):
                g_norm = 0.0
                for g in grads_dict[bs]:
                    g_norm += (g ** 2).sum().item()
                    g.zero_()
                g_norm = np.sqrt(g_norm) / bs
                grads_norms_dict[bs].append(g_norm)

    return grads_norms_dict


def calc_grads_norms_small_memory(model, dataloader, train_mode=False, return_numpy=False, pnames=None):
    "Calculate batch gradients of the model."
    if train_mode:
        model.train()
    else:
        model.eval()

    if pnames is None:
        params = list([param for param in model.parameters() if param.requires_grad == True])
    else:
        params = [p for n, p in model.named_parameters() if (n in pnames and p.requires_grad == True)]

    grads_list = []
    criterion = nn.CrossEntropyLoss()

    gm = []
    for images, labels in dataloader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = criterion(outputs, labels)
        grads = torch.autograd.grad(loss, params)
        gm.append(np.sqrt(sum((t ** 2).sum().item() for t in grads)))

    return np.array(gm).mean()

def calc_grads_corrs(model, dataset, criterion, bs_list, n_samples, train_mode=False, pnames=None):
    """Calculate gradients correlations of the model for each batch size in bs_list.
    """
    def calc_grad(idx):
        image, label = dataset[idx]
        image = image.unsqueeze(0).cuda()
        label = torch.tensor(label).unsqueeze(0).cuda()
        output = model(image)
        loss = criterion(None, output, label)
        grad = torch.autograd.grad(loss, params)
        return grad

    if train_mode:
        model.train()
    else:
        model.eval()

    params = list(model.parameters()) if pnames is None else [p for n, p in model.named_parameters() if n in pnames]
    grads_corrs_dict = defaultdict(list)
    #criterion = nn.CrossEntropyLoss()
    max_bs = max(bs_list)
    
    for _ in range(n_samples):
        subset_ids = np.random.choice(len(dataset), 2 * max_bs, False).reshape(max_bs, 2)
        grads1_dict = dict((bs, [torch.zeros_like(p) for p in params]) for bs in bs_list)
        grads2_dict = dict((bs, [torch.zeros_like(p) for p in params]) for bs in bs_list)

        for i, (idx1, idx2) in enumerate(subset_ids, 1):
            grad1 = calc_grad(idx1)
            grad2 = calc_grad(idx2)

            for bs in bs_list:
                for g_batch, g in zip(grads1_dict[bs], grad1):
                    g_batch += g

                for g_batch, g in zip(grads2_dict[bs], grad2):
                    g_batch += g

                if i % bs == 0:
                    g1_norm = 0.0
                    g2_norm = 0.0
                    g1g2 = 0.0
                    for g1, g2 in zip(grads1_dict[bs], grads2_dict[bs]):
                        g1_norm += (g1 ** 2).sum().item()
                        g2_norm += (g2 ** 2).sum().item()
                        g1g2 = (g1 * g2).sum().item()
                        g1.zero_()
                        g2.zero_()
                    grads_corrs_dict[bs].append(g1g2 / np.sqrt(g1_norm * g2_norm))

    return grads_corrs_dict
