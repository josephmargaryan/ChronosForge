import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import apply_pooling
from dataset import HierarchicalDataset
import pickle
from transformers import AutoTokenizer, AutoModel
from model import DocumentClassifier


def process_document(
    text, tokenizer, model, chunk_size=510, pooling_strategy="mean", device="cpu"
):
    """
    Process a single document into a representation using the specified pooling strategy.

    Parameters:
    - text: The input document as a string.
    - tokenizer: Tokenizer for text tokenization.
    - model: Pretrained transformer model for feature extraction.
    - chunk_size: Maximum size of a single token chunk.
    - pooling_strategy: Pooling strategy for CLS embeddings ('mean', 'max', 'self_attention').
    - device: Device to perform computations on ('cpu' or 'cuda').

    Returns:
    - document_representation: Numpy array representing the document.
    """
    tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].squeeze(0)

    if tokens.size(0) == 0:  # Handle empty input
        tokens = torch.tensor([tokenizer.unk_token_id])

    chunks = [
        tokens[i : i + (chunk_size - 2)] for i in range(0, len(tokens), chunk_size - 2)
    ]
    padded_chunks = []
    for chunk in chunks:
        chunk = torch.cat(
            [
                torch.tensor([tokenizer.cls_token_id]),
                chunk,
                torch.tensor([tokenizer.sep_token_id]),
            ]
        )
        padding_length = chunk_size - chunk.size(0)
        if padding_length > 0:
            chunk = torch.cat(
                (chunk, torch.full((padding_length,), tokenizer.pad_token_id))
            )
        padded_chunks.append(chunk)

    input_ids = torch.stack(padded_chunks).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    with torch.no_grad():
        cls_embeddings = []
        for i in range(input_ids.size(0)):
            outputs = model(
                input_ids[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0)
            )
            cls_embeddings.append(outputs.last_hidden_state[:, 0, :])
        cls_embeddings = torch.cat(cls_embeddings, dim=0)

    return apply_pooling(cls_embeddings, pooling_strategy).cpu().numpy()


def inference(
    transformer_model,
    classifier_model,
    tokenizer,
    df,
    device,
    label_encoder_path,
    chunk_size=510,
    pooling_strategy="mean",
    batch_size=16,
):
    """
    Perform inference on a dataset using the pretrained models.

    Parameters:
    - transformer_model: Pretrained transformer model for feature extraction.
    - classifier_model: Trained classifier model for prediction.
    - tokenizer: Tokenizer for text tokenization.
    - df: DataFrame containing the input text in `df["x"]`.
    - device: Device to perform computations on.
    - label_encoder_path: Path to the saved label encoder file.
    - chunk_size: Maximum size of a single token chunk.
    - pooling_strategy: Pooling strategy for CLS embeddings.
    - batch_size: Batch size for the dataloader.

    Returns:
    - predictions: List of predicted class labels.
    """
    # Load the label encoder
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Ensure models are on the correct device and in evaluation mode
    transformer_model.to(device)
    classifier_model.to(device)
    transformer_model.eval()
    classifier_model.eval()

    # Create a DataLoader for the dataset
    dataloader = DataLoader(
        HierarchicalDataset(
            texts=df["x"].tolist(),
            labels=[0] * len(df),  # Labels are not used in inference
            tokenizer=tokenizer,
            model=transformer_model,
            device=device,
            chunk_size=chunk_size,
            pooling_strategy=pooling_strategy,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    all_preds = []
    for batch in tqdm(dataloader, desc="Performing inference"):
        document_representations, _ = batch  # Document representations and dummy labels
        document_representations = document_representations.to(device)

        with torch.no_grad():
            logits = classifier_model(
                document_representations
            )  # Pass through classifier
            preds = torch.argmax(logits, dim=1)  # Predicted class labels
            all_preds.extend(preds.cpu().numpy())

    # Decode the class labels back to their original form
    return label_encoder.inverse_transform(all_preds)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    # Load the pretrained models and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    transformer_model = AutoModel.from_pretrained("bert-base-uncased").to(device)

    # Load the trained DocumentClassifier
    hidden_size = (
        transformer_model.config.hidden_size
    )  # Hidden size from transformer model
    num_classes = 3  # Replace with the number of classes in your dataset
    classifier_model = DocumentClassifier(hidden_size, num_classes)
    classifier_model.load_state_dict(torch.load("best_model.pth", map_location=device))
    classifier_model.to(device)

    # Load the dataset
    train_df = pd.read_csv("train_data.csv")
    label_encoder_path = "label_encoder.pkl"

    # Perform inference
    predictions = inference(
        transformer_model=transformer_model,
        classifier_model=classifier_model,
        tokenizer=tokenizer,
        df=train_df,
        device=device,
        label_encoder_path=label_encoder_path,
        chunk_size=510,
        pooling_strategy="self_attention",
        batch_size=16,
    )

    # Add predictions to the DataFrame
    train_df["predicted_labels"] = predictions

    # Save the predictions
    train_df.to_csv("train_data_with_predictions.csv", index=False)
    print("Inference complete. Predictions saved to 'train_data_with_predictions.csv'.")
