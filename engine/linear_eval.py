import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from augmentations import get_aug
from engine.dataset import get_dataset
from engine.dataset.utils import classes
from engine.models import get_model

def _extract_feats(backbone, loader, device):
    backbone.eval()
    feats, targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extract features", leave=False):
            images = images.to(device, non_blocking=True)
            f = backbone(images)
            if isinstance(f, (list, tuple)):
                f = f[0]
            # đảm bảo [N, D]
            if f.ndim > 2:
                f = f.flatten(1)
            feats.append(f.detach().cpu())
            targets.append(labels.detach().cpu())
    feats = torch.cat(feats, dim=0)
    targets = torch.cat(targets, dim=0)
    return feats, targets

def _accuracy_top1(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()

def main(args):
    device = getattr(args, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # ====== cấu hình eval (fallback nếu thiếu) ======
    eval_bs     = getattr(getattr(args, "eval", None), "batch_size", args.train.batch_size)
    eval_epochs = getattr(getattr(args, "eval", None), "num_epochs", 50)
    eval_lr     = getattr(getattr(args, "eval", None), "lr", 0.1)
    num_workers = args.dataloader_kwargs.get("num_workers", 0)

    # ====== datasets / loaders ======
    eval_tf = get_aug(train=False, train_classifier=True, **args.aug_kwargs)
    dataset_kwargs = dict(args.dataset_kwargs)
    dataset_kwargs.pop('dataset', None)

    train_set = get_dataset(dataset='endo_labeled', transform=eval_tf, train=True,  **dataset_kwargs)
    val_set   = get_dataset(dataset='endo_labeled', transform=eval_tf, train=False, **dataset_kwargs)

    train_loader_img = DataLoader(train_set, batch_size=eval_bs, shuffle=False,
                                  pin_memory=True, num_workers=num_workers)
    val_loader_img   = DataLoader(val_set,   batch_size=eval_bs, shuffle=False,
                                  pin_memory=True, num_workers=num_workers)

    # ====== build backbone và load checkpoint self-supervised ======
    model = get_model(args.model).to(device)
    ckpt_path = getattr(args, "eval_from", None)
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        target = model.module if isinstance(model, (nn.DataParallel,)) else model
        missing, unexpected = target.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"[linear_eval] Loaded {ckpt_path} | missing={missing} unexpected={unexpected}")
    else:
        print(f"[linear_eval] WARNING: args.eval_from is None hoặc file không tồn tại: {ckpt_path}")

    backbone = model.backbone
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    # ====== trích xuất đặc trưng một lần ======
    train_feats, train_targets = _extract_feats(backbone, train_loader_img, device)
    val_feats,   val_targets   = _extract_feats(backbone, val_loader_img,   device)

    feat_dim = int(train_feats.shape[1])
    # try:
    #     if args.num_classes is not None:
    #         num_classes = args.num_classes
    #     else:
    #         num_classes = len(classes)
    # except:
    #     num_classes = len(classes)
    
    if args.train_binary:
        num_classes = 2
    else:
        num_classes = len(classes)

    # ====== linear head ======
    clf = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = optim.SGD(clf.parameters(), lr=eval_lr, momentum=0.9, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=eval_epochs)
    criterion = nn.CrossEntropyLoss()

    # ====== dataloader trên feature ======
    train_ds = TensorDataset(train_feats, train_targets)
    val_ds   = TensorDataset(val_feats,   val_targets)

    # batch lớn cho feature để nhanh hơn
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=4096, shuffle=False, drop_last=False)

    # ====== TensorBoard writer (ghi CHUNG thư mục log) ======
    writer = SummaryWriter(log_dir=args.log_dir)
    global_step = 0
    best_acc = 0.0
    best_path = None

    # ====== train linear ======
    for epoch in range(eval_epochs):
        clf.train()
        tot_loss, tot_correct, tot_num = 0.0, 0, 0

        for xb, yb in tqdm(train_loader, desc=f"[Linear] Epoch {epoch+1}/{eval_epochs}", leave=False):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = clf(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                tot_loss += loss.item() * xb.size(0)
                tot_correct += (logits.argmax(1) == yb).sum().item()
                tot_num += yb.size(0)

            writer.add_scalar("linear/loss_step", float(loss.item()), global_step)
            global_step += 1

        train_loss = tot_loss / max(1, tot_num)
        train_acc = tot_correct / max(1, tot_num)
        writer.add_scalar("linear/loss_epoch", train_loss, epoch)
        writer.add_scalar("linear/acc_epoch",  train_acc,  epoch)
        writer.add_scalar("linear/lr_epoch",   optimizer.param_groups[0]['lr'], epoch)

        # ====== validate ======
        clf.eval()
        v_loss, v_correct, v_num = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = clf(xb)
                loss = criterion(logits, yb)
                v_loss    += loss.item() * xb.size(0)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_num     += yb.size(0)

        val_loss = v_loss / max(1, v_num)
        val_acc  = v_correct / max(1, v_num)
        writer.add_scalar("linear/val_loss", val_loss, epoch)
        writer.add_scalar("linear/val_acc",  val_acc,  epoch)

        scheduler.step()

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(args.ckpt_dir, exist_ok=True)
            best_path = os.path.join(args.ckpt_dir, f"{args.name}_linear_best.pth")
            torch.save({
                "state_dict": clf.state_dict(),
                "feat_dim": feat_dim,
                "num_classes": num_classes,
                "best_acc": best_acc,
                "epoch": epoch + 1,
            }, best_path)

    writer.close()
    if best_path:
        print(f"[linear_eval] Best val_acc={best_acc:.4f} -> {best_path}")
    else:
        print("[linear_eval] Done (no best saved)")

    return best_acc
