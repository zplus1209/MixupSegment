import torch
import numpy as np

class LR_Scheduler(object):
    """
    - Warmup: tuyến tính từ warmup_lr -> base_lr trong warmup_epochs * iter_per_epoch bước.
    - Cosine: từ base_lr -> final_lr cho phần còn lại (num_epochs - warmup_epochs) * iter_per_epoch bước.
    - Hỗ trợ:
        * constant_predictor_lr: giữ nguyên lr cho param_group có 'name' == 'predictor'.
        * state_dict()/load_state_dict(): resume chính xác theo 'iter' đã chạy.
    """
    def __init__(self, optimizer, warmup_epochs, warmup_lr,
                 num_epochs, base_lr, final_lr, iter_per_epoch,
                 constant_predictor_lr=False):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.final_lr = float(final_lr)
        self.warmup_lr = float(warmup_lr)
        self.warmup_epochs = int(warmup_epochs)
        self.num_epochs = int(num_epochs)
        self.iter_per_epoch = int(iter_per_epoch)
        self.constant_predictor_lr = bool(constant_predictor_lr)

        # Build schedule
        warmup_iter = self.iter_per_epoch * self.warmup_epochs
        if warmup_iter > 0:
            warmup_lr_schedule = np.linspace(self.warmup_lr, self.base_lr, warmup_iter, dtype=np.float64)
        else:
            warmup_lr_schedule = np.array([], dtype=np.float64)

        decay_iter = self.iter_per_epoch * max(0, self.num_epochs - self.warmup_epochs)
        if decay_iter > 0:
            cosine_lr_schedule = self.final_lr + 0.5 * (self.base_lr - self.final_lr) * \
                                 (1.0 + np.cos(np.pi * np.arange(decay_iter, dtype=np.float64) / max(1, decay_iter)))
        else:
            cosine_lr_schedule = np.array([], dtype=np.float64)

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule), axis=0)
        self.total_iters = int(self.lr_schedule.shape[0])

        self.iter = 0
        self.current_lr = None  # sẽ set khi step()

    def _get_lr_at_iter(self, i: int) -> float:
        if self.total_iters == 0:
            # Không có schedule => giữ nguyên lr hiện tại của optimizer
            # (fallback an toàn)
            return self.base_lr
        # nếu vượt quá length schedule, giữ giá trị cuối
        i = min(max(i, 0), self.total_iters - 1)
        return float(self.lr_schedule[i])

    def step(self):
        lr = self._get_lr_at_iter(self.iter)

        for param_group in self.optimizer.param_groups:
            # Nếu bạn có gán param_group['name'] = 'predictor' ở optimizer:
            if self.constant_predictor_lr and param_group.get('name', None) == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                param_group['lr'] = lr

        self.current_lr = lr
        self.iter += 1
        return lr

    def get_lr(self):
        # Trả về lr sau lần step() gần nhất (hoặc lr tại iter hiện thời nếu chưa step lần nào)
        if self.current_lr is None:
            return self._get_lr_at_iter(self.iter)
        return float(self.current_lr)

    # --------- NEW: để lưu/khôi phục state ----------
    def state_dict(self):
        # Dùng kiểu Python builtin (list/float/int) để torch.save dễ serializable
        return {
            "iter": int(self.iter),
            "current_lr": float(self.get_lr()),  # đảm bảo có giá trị hợp lệ
            "base_lr": float(self.base_lr),
            "final_lr": float(self.final_lr),
            "warmup_lr": float(self.warmup_lr),
            "warmup_epochs": int(self.warmup_epochs),
            "num_epochs": int(self.num_epochs),
            "iter_per_epoch": int(self.iter_per_epoch),
            "constant_predictor_lr": bool(self.constant_predictor_lr),
            "lr_schedule": self.lr_schedule.tolist(),  # để resume đúng y schedule cũ
        }

    def load_state_dict(self, state):
        # Khôi phục tham số quan trọng
        self.base_lr = float(state.get("base_lr", self.base_lr))
        self.final_lr = float(state.get("final_lr", self.final_lr))
        self.warmup_lr = float(state.get("warmup_lr", self.warmup_lr))
        self.warmup_epochs = int(state.get("warmup_epochs", self.warmup_epochs))
        self.num_epochs = int(state.get("num_epochs", self.num_epochs))
        self.iter_per_epoch = int(state.get("iter_per_epoch", self.iter_per_epoch))
        self.constant_predictor_lr = bool(state.get("constant_predictor_lr", self.constant_predictor_lr))

        # schedule: nếu có trong state thì dùng y nguyên để đảm bảo continuity
        sched = state.get("lr_schedule", None)
        if sched is not None:
            self.lr_schedule = np.array(sched, dtype=np.float64)
            self.total_iters = int(self.lr_schedule.shape[0])
        else:
            # nếu không có, build lại theo cấu hình hiện tại
            warmup_iter = self.iter_per_epoch * self.warmup_epochs
            warmup_lr_schedule = np.linspace(self.warmup_lr, self.base_lr, warmup_iter, dtype=np.float64) \
                                 if warmup_iter > 0 else np.array([], dtype=np.float64)
            decay_iter = self.iter_per_epoch * max(0, self.num_epochs - self.warmup_epochs)
            cosine_lr_schedule = self.final_lr + 0.5 * (self.base_lr - self.final_lr) * \
                                 (1.0 + np.cos(np.pi * np.arange(decay_iter, dtype=np.float64) / max(1, decay_iter))) \
                                 if decay_iter > 0 else np.array([], dtype=np.float64)
            self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule), axis=0)
            self.total_iters = int(self.lr_schedule.shape[0])

        # iter & current_lr
        self.iter = int(state.get("iter", 0))
        # Đồng bộ current_lr với optimizer để không bị “nhảy” bất ngờ:
        self.current_lr = float(state.get("current_lr", self._get_lr_at_iter(max(0, self.iter - 1)) ))

        # Cập nhật ngay lr hiện tại vào optimizer (khớp với state đã load)
        lr_now = self.current_lr
        for pg in self.optimizer.param_groups:
            if self.constant_predictor_lr and pg.get('name', None) == 'predictor':
                pg['lr'] = self.base_lr
            else:
                pg['lr'] = lr_now


if __name__ == "__main__":
    import torchvision
    model = torchvision.models.resnet50()
    optimizer = torch.optim.SGD(model.parameters(), lr=999)
    epochs = 100
    n_iter = 1000
    scheduler = LR_Scheduler(optimizer, 10, 1, epochs, 3, 0, n_iter)
    import matplotlib.pyplot as plt
    lrs = []
    for epoch in range(epochs):
        for it in range(n_iter):
            lr = scheduler.step()
            lrs.append(lr)
    plt.plot(lrs)
    plt.show()