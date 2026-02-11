# classes = [
#     'barretts',
#     'barretts-short-segment',
#     'bbps-0-1',
#     'bbps-2-3',
#     'cecum',
#     'dyed-lifted-polyps',
#     'dyed-resection-margins',
#     'esophagitis-a',
#     'esophagitis-b-d',
#     'hemorrhoids',
#     'ileum',
#     'impacted-stool',
#     'polyps',
#     'pylorus',
#     'retroflex-rectum',
#     'retroflex-stomach',
#     'ulcerative-colitis-grade-0-1',
#     'ulcerative-colitis-grade-1',
#     'ulcerative-colitis-grade-1-2',
#     'ulcerative-colitis-grade-2',
#     'ulcerative-colitis-grade-2-3',
#     'ulcerative-colitis-grade-3',
#     'z-line',
# ]

classes = [
    "bbps-0-1",
    "bbps-2-3",
    "cecum",
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis-a",
    "esophagitis-b-d",
    "impacted-stool",
    "polyps",
    "pylorus",
    "retroflex-rectum",
    "retroflex-stomach",
    "ulcerative-colitis-grade-1",
    "ulcerative-colitis-grade-2",
    "ulcerative-colitis-grade-3",
    "z-line",
]

import torch

class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.iterator = None
        self.next_batch = None

    def __iter__(self):
        self.iterator = iter(self.loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_batch is None:
            raise StopIteration
        batch = self.next_batch
        self.preload()
        return batch

    def preload(self):
        try:
            (img1, img2), labels = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            img1 = img1.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            img2 = img2.to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            # labels không dùng trong SSL, nhưng cứ move nếu là Tensor
            if torch.is_tensor(labels):
                labels = labels.to(self.device, non_blocking=True)
        self.next_batch = ((img1, img2), labels)