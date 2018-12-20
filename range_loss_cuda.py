from __future__ import absolute_import

import torch
from torch import nn


class RangeLoss(nn.Module):
    """
        Range_loss = alpha * intra_class_loss + beta * inter_class_loss
        intra_class_loss is the harmonic mean value of the top_k largest distances beturn intra_class_pairs
        inter_class_loss is the shortest distance between different class centers
    """
    def __init__(self, k=2, margin=0.1, alpha=0.5, beta=0.5, use_gpu=True):
        super(RangeLoss, self).__init__()
        self.use_gpu = use_gpu
        self.margin = margin
        self.k = k
        self.alpha = alpha
        self.beta = beta

    def _pairwise_distance(self, features):
        """
         Args:
            features: prediction matrix (before softmax) with shape (batch_size, num_classes)
         Return: 
            pairwise distance matrix with shape(batch_size, batch_size)
        """
        n = features.size(0)
        dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, features, features.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def _compute_top_k(self, features):
        """
         Args:
            features: prediction matrix (before softmax) with shape (batch_size, num_classes)
         Return: 
            top_k largest distances
        """
        dist_array = self._pairwise_distance(features)
        dist_array = dist_array.view(1, -1)
        return dist_array.sort()[0][0, -self.k:]

    def _compute_min_dist(self, center_features):
        """
         Args:
            center_features: center matrix (before softmax) with shape (center_number, num_classes)
         Return: 
            minimum center distance
        """
        dist_array = self._pairwise_distance(center_features)
        dist_array = dist_array.view(1, -1)
        dist_array = dist_array[torch.gt(dist_array, 0)]
        return dist_array.min()

    def _calculate_centers(self, features, targets):
        """
         Args:
            features: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
         Return: 
            center_features: center matrix (before softmax) with shape (center_number, num_classes)
        """
        unique_labels = targets.cpu().unique().cuda()
        center_features = torch.zeros(unique_labels.size()[0], features.size()[1])
        if self.use_gpu:
            center_features.cuda()
        for i in range(unique_labels.size()[0]):
            label = unique_labels[i]
            same_class_features = features[targets == label]
            center_features[i] = same_class_features.mean(dim=0)
        return center_features

    def _inter_class_loss(self, features, targets):
        """
         Args:
            features: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
            margin: inter class ringe loss margin
         Return: 
            inter_class_loss
        """
        center_features = self._calculate_centers(features, targets)
        min_inter_class_center_distance = self._compute_min_dist(center_features)
        return torch.max(self.margin - min_inter_class_center_distance, 0)[0]

    def _intra_class_loss(self, features, targets):
        """
         Args:
            features: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
         Return: 
            intra_class_loss
        """
        unique_labels = targets.cpu().unique().cuda()
        same_class_distances = torch.zeros(unique_labels.size()[0], self.k)
        intra_distance = torch.zeros(unique_labels.size()[0])
        if self.use_gpu:
            same_class_distances.cuda()
        for i in range(unique_labels.size()[0]):
            label = unique_labels[i]
            same_class_distances[i, :] = self._compute_top_k(features[targets == label])
            intra_distance[i] = self.k / torch.sum(self._compute_top_k(features[targets == label]))
        return torch.sum(intra_distance)

    def _range_loss(self, features, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        Return:
             range_loss
        """
        inter_class_loss = self._inter_class_loss(features, targets)
        intra_class_loss = self._intra_class_loss(features, targets)
        range_loss = self.alpha * inter_class_loss + self.beta * intra_class_loss
        return range_loss

    def forward(self, features, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        Return:
             range_loss
        """
        if self.use_gpu:
            targets = targets.cuda()

        range_loss = self._range_loss(features, targets)
        return range_loss


if __name__ == '__main__':
        range_loss = RangeLoss()
        features = torch.rand(16, 2048).cuda()
        targets = torch.Tensor([0,1,2,3,2,3,1,4,5,3,2,1,0,0,5,4]).cuda()
        loss = range_loss(features, targets)
        print(loss)
