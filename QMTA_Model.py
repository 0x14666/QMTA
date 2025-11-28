import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool, global_max_pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class PrototypicalContrastiveLoss(nn.Module):
    """
    Prototypical Contrastive Learning Loss Function
    
    Combines the advantages of supervised contrastive learning and prototypical learning.
    This loss function maintains dynamic class prototypes and computes both prototype-based
    and instance-based contrastive losses for improved representation learning.
    """
    
    def __init__(self, temperature=0.07, proto_momentum=0.99, num_classes=2, feature_dim=64):
        """
        Initialize the Prototypical Contrastive Loss.
        
        Args:
            temperature (float): Temperature parameter for softmax scaling
            proto_momentum (float): Momentum factor for prototype updates
            num_classes (int): Number of classes in the dataset
            feature_dim (int): Dimensionality of feature representations
        """
        super(PrototypicalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.proto_momentum = proto_momentum
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Initialize prototype vectors (dynamic class centers)
        self.register_buffer('prototypes', torch.randn(num_classes, feature_dim))
        self.prototypes = F.normalize(self.prototypes, dim=1)
        
        # Prototype update counters
        self.register_buffer('prototype_counts', torch.zeros(num_classes))
        
    def update_prototypes(self, features, labels):
        """
        Update prototype vectors using momentum.
        
        Args:
            features (torch.Tensor): Normalized feature vectors [batch_size, feature_dim]
            labels (torch.Tensor): Corresponding labels [batch_size]
        """
        with torch.no_grad():
            for class_id in range(self.num_classes):
                # Find samples belonging to current class
                mask = (labels == class_id)
                if mask.sum() == 0:
                    continue
                
                # Compute mean of current class samples in the batch
                class_features = features[mask]
                current_proto = class_features.mean(dim=0)
                current_proto = F.normalize(current_proto, dim=0)
                
                # Momentum update of prototypes
                self.prototypes[class_id] = (
                    self.proto_momentum * self.prototypes[class_id] + 
                    (1 - self.proto_momentum) * current_proto
                )
                self.prototypes[class_id] = F.normalize(self.prototypes[class_id], dim=0)
                
                # Update counters
                self.prototype_counts[class_id] += mask.sum().item()

    def compute_proto_contrastive_loss(self, features, labels):
        """
        Compute prototype-based contrastive loss.
        
        Args:
            features (torch.Tensor): Feature vectors [batch_size, feature_dim]
            labels (torch.Tensor): Labels [batch_size]
            
        Returns:
            torch.Tensor: Prototype contrastive loss
        """
        batch_size = features.shape[0]
        
        # Compute similarity between features and all prototypes
        proto_sim = torch.mm(features, self.prototypes.t()) / self.temperature
        
        # Create positive/negative sample masks
        labels_expanded = labels.view(-1, 1)
        class_mask = torch.eq(labels_expanded, torch.arange(self.num_classes).to(features.device)).float()
        
        # Compute log probabilities
        proto_logits = proto_sim - torch.logsumexp(proto_sim, dim=1, keepdim=True)
        
        # Compute prototype contrastive loss
        proto_loss = -(class_mask * proto_logits).sum(dim=1).mean()
        
        return proto_loss

    def compute_instance_contrastive_loss(self, features, labels):
        """
        Compute instance-wise contrastive loss (original supervised contrastive learning).
        
        Args:
            features (torch.Tensor): Feature vectors [batch_size, n_views, feature_dim]
            labels (torch.Tensor): Labels [batch_size]
            
        Returns:
            torch.Tensor: Instance contrastive loss
        """
        if len(features.shape) < 3:
            # Skip instance contrastive loss if no multi-view data
            return torch.tensor(0.0, device=features.device)
        
        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        
        # Flatten multi-view features
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.mm(anchor_feature, contrast_feature.t()), self.temperature)
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create positive sample mask
        labels_contrast = labels.repeat(contrast_count)
        mask = torch.eq(labels_contrast.view(-1, 1), labels_contrast.view(1, -1)).float()
        
        # Remove self-correlation
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean positive log-likelihood
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        # Instance contrastive loss
        instance_loss = -mean_log_prob_pos.mean()
        
        return instance_loss

    def forward(self, features, labels, update_proto=True):
        """
        Forward pass to compute total loss.
        
        Args:
            features (torch.Tensor): Feature vectors, either single-view [batch_size, feature_dim] 
                                   or multi-view [batch_size, n_views, feature_dim]
            labels (torch.Tensor): Labels [batch_size]
            update_proto (bool): Whether to update prototypes
            
        Returns:
            tuple: (prototype_loss, instance_loss)
        """
        # Handle multi-view case
        if len(features.shape) == 3:
            # Multi-view: use first view for prototype update and loss computation
            proto_features = features[:, 0, :]  # [batch_size, feature_dim]
            instance_features = features  # [batch_size, n_views, feature_dim]
        else:
            # Single view
            proto_features = features
            instance_features = features.unsqueeze(1)  # Add view dimension
        
        # Update prototypes
        if update_proto and self.training:
            self.update_prototypes(proto_features, labels)
        
        # Compute prototype contrastive loss
        proto_loss = self.compute_proto_contrastive_loss(proto_features, labels)
        
        # Compute instance contrastive loss
        instance_loss = self.compute_instance_contrastive_loss(instance_features, labels)
        
        return proto_loss, instance_loss


class QMTA(nn.Module):
    """
    Quantum Malware Traffic Analysis with Prototypical Contrastive Learning
    
    A Graph Neural Network model for malware traffic analysis that incorporates
    prototypical contrastive learning for improved representation learning.
    """
    
    def __init__(self, input_dim, hidden_dim=128, projection_dim=64, num_classes=2, dropout=0.3):
        """
        Initialize the QMTA model.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            projection_dim (int): Projection head output dimension
            num_classes (int): Number of classes
            dropout (float): Dropout rate
        """
        super(QMTA, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.num_classes = num_classes
        
        # Encoder components (same as original)
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim//4, heads=4, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim//4, heads=4, concat=True, dropout=dropout)
        self.global_conv = GraphConv(hidden_dim, hidden_dim//2)
        
        # Prototypical contrastive learning projection head (deeper network)
        self.prototype_projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim//2, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward_encoder(self, data):
        """
        Encoder forward pass (same as original).
        
        Args:
            data: Graph data object
            
        Returns:
            torch.Tensor: Graph representation
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN layers
        x = F.relu(self.gcn1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.gcn2(x, edge_index))
        x = self.dropout(x)
        
        # GAT layers
        x = F.relu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.gat2(x, edge_index))
        x = self.dropout(x)
        
        # Graph-level feature extraction
        x = self.global_conv(x, edge_index)
        
        # Graph-level pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        graph_representation = torch.cat([x_mean, x_max], dim=1)
        
        return graph_representation
    
    def forward(self, data, mode='classification'):
        """
        Forward pass with different modes.
        
        Args:
            data: Graph data object
            mode (str): One of 'classification', 'prototypical_contrastive', 'encoding'
            
        Returns:
            torch.Tensor: Output based on the specified mode
        """
        # Get graph representation
        graph_repr = self.forward_encoder(data)
        
        if mode == 'classification':
            # Classification mode
            logits = self.classification_head(graph_repr)
            return logits
        elif mode == 'prototypical_contrastive':
            # Prototypical contrastive learning mode
            projected = self.prototype_projection_head(graph_repr)
            # L2 normalization
            projected = F.normalize(projected, dim=1)
            return projected
        elif mode == 'encoding':
            # Return raw encoding
            return graph_repr
        else:
            raise ValueError(f"Unknown mode: {mode}")


class QMTA_Trainer:
    """
    QMTA Trainer with Prototypical Contrastive Learning
    
    A comprehensive trainer class that handles training, validation, and evaluation
    of the QMTA model with prototypical contrastive learning.
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu',
                 proto_weight=0.4, instance_weight=0.2, classification_weight=0.4,
                 temperature=0.07, proto_momentum=0.99):
        """
        Initialize the QMTA trainer.
        
        Args:
            model: QMTA model instance
            device (str): Device to use for training
            proto_weight (float): Weight for prototype loss
            instance_weight (float): Weight for instance loss
            classification_weight (float): Weight for classification loss
            temperature (float): Temperature parameter for contrastive learning
            proto_momentum (float): Momentum for prototype updates
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss weights
        self.proto_weight = proto_weight
        self.instance_weight = instance_weight 
        self.classification_weight = classification_weight
        
        # Initialize loss functions
        self.prototypical_criterion = PrototypicalContrastiveLoss(
            temperature=temperature,
            proto_momentum=proto_momentum,
            num_classes=model.num_classes,
            feature_dim=model.projection_dim
        ).to(device)
        
        self.classification_criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.proto_losses = []
        self.instance_losses = []
        self.classification_losses = []
        
        print("Prototypical Contrastive Learning Configuration:")
        print(f"  Prototype loss weight: {proto_weight}")
        print(f"  Instance loss weight: {instance_weight}")
        print(f"  Classification loss weight: {classification_weight}")
        print(f"  Prototype momentum: {proto_momentum}")
        
    def create_augmented_views(self, data_batch):
        """
        Create augmented views for contrastive learning (improved version).
        
        Args:
            data_batch: Input graph data batch
            
        Returns:
            list: List of augmented views
        """
        original_features = data_batch.x
        
        # View 1: Light Gaussian noise + feature masking
        noise_std = 0.05
        mask_ratio = 0.05
        
        view1_features = original_features + torch.randn_like(original_features) * noise_std
        feature_mask = (torch.rand_like(original_features) > mask_ratio).float()
        view1_features = view1_features * feature_mask
        
        # View 2: Feature dropout + scaling
        dropout_ratio = 0.1
        scale_factor = 0.95 + torch.rand_like(original_features) * 0.1
        
        dropout_mask = (torch.rand_like(original_features) > dropout_ratio).float()
        view2_features = original_features * dropout_mask * scale_factor
        
        # View 3: Original features (as anchor)
        view3_features = original_features
        
        # Create data objects for the three views
        views = []
        for features in [view1_features, view2_features, view3_features]:
            view_data = data_batch.clone()
            view_data.x = features
            views.append(view_data)
        
        return views
    
    def train_epoch(self, train_loader, optimizer):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer instance
            
        Returns:
            tuple: Training metrics (loss, proto_loss, instance_loss, classification_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        total_proto_loss = 0
        total_instance_loss = 0
        total_classification_loss = 0
        correct = 0
        total = 0
        
        for data in train_loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            
            # 1. Create augmented views
            views = self.create_augmented_views(data)
            
            # 2. Get prototypical contrastive features for all views
            proto_features = []
            for view in views:
                proto_feat = self.model(view, mode='prototypical_contrastive')
                proto_features.append(proto_feat)
            
            # Stack multi-view features [batch_size, n_views, feature_dim]
            proto_features = torch.stack(proto_features, dim=1)
            
            # 3. Compute prototypical contrastive loss
            proto_loss, instance_loss = self.prototypical_criterion(
                proto_features, data.y, update_proto=True)
            
            # 4. Classification loss (using original data)
            classification_logits = self.model(data, mode='classification')
            classification_loss = self.classification_criterion(classification_logits, data.y)
            
            # 5. Total loss
            total_batch_loss = (
                self.proto_weight * proto_loss + 
                self.instance_weight * instance_loss +
                self.classification_weight * classification_loss
            )
            
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            total_loss += total_batch_loss.item()
            total_proto_loss += proto_loss.item()
            total_instance_loss += instance_loss.item()
            total_classification_loss += classification_loss.item()
            
            pred = classification_logits.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            total += data.y.size(0)
        
        avg_loss = total_loss / len(train_loader)
        avg_proto_loss = total_proto_loss / len(train_loader)
        avg_instance_loss = total_instance_loss / len(train_loader)
        avg_classification_loss = total_classification_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, avg_proto_loss, avg_instance_loss, avg_classification_loss, accuracy
    
    def evaluate(self, val_loader):
        """
        Evaluate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            tuple: Validation metrics (loss, accuracy, predictions, labels)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                
                # Use only classification loss for evaluation
                logits = self.model(data, mode='classification')
                loss = self.classification_criterion(logits, data.y)
                
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += pred.eq(data.y).sum().item()
                total += data.y.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy, all_preds, all_labels
    
    def get_prototype_analysis(self):
        """
        Analyze prototype vectors.
        
        Returns:
            tuple: (prototypes, counts, similarity_matrix)
        """
        prototypes = self.prototypical_criterion.prototypes.cpu().numpy()
        counts = self.prototypical_criterion.prototype_counts.cpu().numpy()
        
        print("\n=== Prototype Analysis ===")
        for i in range(len(prototypes)):
            proto_norm = np.linalg.norm(prototypes[i])
            print(f"Class {i}: Prototype norm = {proto_norm:.4f}, Sample count = {counts[i]:.0f}")
        
        # Compute inter-prototype similarity
        from sklearn.metrics.pairwise import cosine_similarity
        proto_sim = cosine_similarity(prototypes)
        print("\nInter-prototype cosine similarity:")
        print(proto_sim)
        
        return prototypes, counts, proto_sim
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, patience=20):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs (int): Number of training epochs
            lr (float): Learning rate
            patience (int): Early stopping patience
        """
        # Use AdamW optimizer for better regularization
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=7, factor=0.7, verbose=True)
        
        best_val_acc = 0
        patience_counter = 0
        
        print("Starting prototypical contrastive learning training...")
        for epoch in range(epochs):
            # Training
            train_loss, proto_loss, instance_loss, class_loss, train_acc = self.train_epoch(train_loader, optimizer)
            
            # Validation
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.proto_losses.append(proto_loss)
            self.instance_losses.append(instance_loss)
            self.classification_losses.append(class_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'  Train Loss: {train_loss:.4f} (Proto: {proto_loss:.4f}, Instance: {instance_loss:.4f}, Class: {class_loss:.4f})')
                print(f'  Train Acc: {train_acc:.4f}')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                print(f'  Current LR: {optimizer.param_groups[0]["lr"]:.6f}')
                
                # Analyze prototypes every 20 epochs
                if (epoch + 1) % 20 == 0:
                    self.get_prototype_analysis()
            
            # Early stopping mechanism
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'prototypes': self.prototypical_criterion.prototypes,
                    'prototype_counts': self.prototypical_criterion.prototype_counts,
                    'epoch': epoch,
                    'val_acc': val_acc
                }, 'best_malware_proto_gnn_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        print(f'Training completed! Best validation accuracy: {best_val_acc:.4f}')
        
        # Load best model
        checkpoint = torch.load('best_malware_proto_gnn_model.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.prototypical_criterion.prototypes = checkpoint['prototypes']
        self.prototypical_criterion.prototype_counts = checkpoint['prototype_counts']
        
        # Final prototype analysis
        print("\n=== Final Prototype Analysis ===")
        self.get_prototype_analysis()