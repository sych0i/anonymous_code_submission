import torch

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n
        self.count = n

    def find(self, x):
        while self.parent[x] != x:
            # path compression
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # union by rank
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.count -= 1

def binarized_images_uniqueness_score(imgs: torch.LongTensor, thresh: float) -> int:
    """
    imgs: LongTensor of shape (B,1,28,28) with values {0,1}
    thresh: Jaccard threshold in (0,1)—if overlap > thresh, they are 'same'

    returns: number of unique clusters (connected components)
    """
    return len(binarized_images_unique_representative_indices(imgs, thresh))


def binarized_images_unique_representative_indices(imgs: torch.LongTensor, thresh: float) -> list[int]:
    """
    Returns one representative index per Jaccard-similarity cluster.
    """
    B, _, H, W = imgs.shape
    if B == 0:
        return []
    if B == 1:
        return [0]

    N = H*W

    # flatten and cast to float for dot products
    X = imgs.view(B, N).float()  # shape (B, N)

    # compute per-image sums (|A| for each row)
    sums = X.sum(dim=1)                # shape (B,)

    # pairwise intersection = X @ X^T
    inter = X @ X.t()                  # shape (B, B)

    # pairwise union = |A| + |B| - intersection
    union = sums.unsqueeze(1) + sums.unsqueeze(0) - inter  # (B, B)

    # Jaccard similarity matrix
    jaccard = inter / (union + 1e-8)   # avoid zero‐division

    # build DSU over edges where jaccard > thresh (excluding self‐loops)
    uf = UnionFind(B)
    # Only need to iterate upper‐triangle
    iu = torch.triu_indices(B, B, offset=1)
    rows, cols = iu[0], iu[1]
    sim_pairs = (jaccard[rows, cols] > thresh).nonzero(as_tuple=False)
    for idx in sim_pairs:
        i, j = rows[idx], cols[idx]
        uf.union(int(i), int(j))

    representatives = {}
    for i in range(B):
        root = uf.find(i)
        representatives.setdefault(root, i)

    return list(representatives.values())


def binarized_images_diversity(imgs):
    """
    imgs: LongTensor of shape (B,1,28,28) with values {0,1,2}.
          2 = masked (ignore).
    returns: scalar—mean normalized Hamming distance over all image pairs.
    """
    imgs = torch.from_numpy(imgs)
    B = imgs.size(0)
    # flatten to (B, 28*28)
    x = imgs.view(B, -1)
    # mask where !=2
    valid = (x != 2)  # (B, P)
    
    # preallocate
    total = 0.0
    count = 0

    for i in range(B):
        xi, mi = x[i], valid[i]
        for j in range(i+1, B):
            xj, mj = x[j], valid[j]
            # only compare pixels both unmasked
            both = mi & mj                  # (P,)
            n = both.sum().item()
            if n == 0:
                continue  # no common pixels—skip
            # XOR counts where bits differ
            diffs = (xi[both] != xj[both]).sum().item()
            total += diffs / n
            count += 1

    val = total / count if count > 0 else 0.0
    return val * 100
