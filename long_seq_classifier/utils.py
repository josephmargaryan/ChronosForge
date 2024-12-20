import torch
from tqdm import tqdm


def apply_pooling(cls_embeddings, pooling_strategy="mean"):
    """
    Apply pooling to CLS token embeddings to generate a document representation.

    Parameters:
    - cls_embeddings: Tensor of shape (num_chunks, hidden_size)
    - pooling_strategy: Pooling method ('mean', 'max', 'self_attention')

    Returns:
    - document_representation: Tensor of shape (hidden_size,)
    """
    if pooling_strategy == "mean":
        document_representation = torch.mean(cls_embeddings, dim=0)
    elif pooling_strategy == "max":
        document_representation = torch.max(cls_embeddings, dim=0)[0]
    elif pooling_strategy == "self_attention":
        attn_weights = torch.softmax(
            torch.mm(cls_embeddings, cls_embeddings.transpose(0, 1)), dim=-1
        )
        document_representation = torch.mm(attn_weights, cls_embeddings).mean(0)
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
    return document_representation


def get_document_representations(dataloader, device):
    """
    Extract document embeddings using the DataLoader.
    """
    all_representations = []
    all_labels = []
    with torch.no_grad():
        for document_representation, label in tqdm(
            dataloader, desc="Extracting embeddings"
        ):
            all_representations.append(document_representation.to(device))
            all_labels.append(label)

    embeddings = torch.cat(all_representations, dim=0).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()
    return embeddings, labels
