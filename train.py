from utils import *
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fit(checkpoint, criterion, embeddings ,train_loader, val_loader, epochs, clip=float('inf'),entropy_param=0., writer=None):
    """Full training loop"""

    print("Training on", 'GPU' if device.type == 'cuda' else 'CPU', '\n')
    net, optimizer = checkpoint.model, checkpoint.optimizer
    min_loss = float('inf')
    iteration = 1

    def train_epoch():
        """
        Returns:
            The epoch loss
        """
        nonlocal iteration
        epoch_loss = 0.
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', dynamic_ncols=True,  position=0, leave=True)  # progress bar
        net.train()
        n_entropy  = len(train_loader)
        #n_epoch_entropy = 5
        for (i, batch) in enumerate(pbar):
            texts, lengths, mask, labels = batch
            embedding_batch = get_embeddings(embeddings, texts).to(device)
            texts, lengths, mask, labels = texts.to(device),lengths.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(embedding_batch, mask, lengths)
            #entropy = torch.distributions.Categorical(weights).entropy()
            #entropy = entropy.mean()
            loss = criterion(output, labels) 
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4e}')
            if writer:
                writer.add_scalar('Iteration_loss', loss.item(), iteration)
                #writer.add_histogram("entropies", entropy, epoch)
            # compute gradients, update parameters
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(),max_norm=clip)
            if writer:
                writer.add_scalar('Total_norm_of_parameters_gradients', total_norm, iteration)
            optimizer.step()
            iteration += 1
        epoch_loss /= len(train_loader)
        return epoch_loss
    def get_embeddings(embeddings, texts):
        """
        Returns the embeddings corresponding to sequence texts
        """
        return embeddings[texts]
    
    def categorical_accuracy(preds, y):
        """
        Returns accuracy per batch
        """
        y = y.cpu()
        preds = preds.cpu()
        #print(preds.shape, y.shape)
        max_preds = preds.argmax(dim=1, keepdim=True)  
        correct = max_preds.squeeze(1).eq(y) # 
        return correct.sum() / torch.FloatTensor([y.shape[0]])
    
    def evaluate_epoch(loader, role='Val'):
        """
        Args:
            loader (torch.utils.data.DataLoader): either the train of validation loader
            role (str): either 'Val' or 'Train'
        Returns:
            Tuple containing mean loss and accuracy
        """
        net.eval()
        correct = 0
        mean_loss = 0.
        with torch.no_grad():
            for batch in loader:
                texts, lengths, mask, labels = batch
                embedding_batch = embeddings[texts].to(device)
                texts, lengths, mask, labels = texts.to(device),lengths.to(device), mask.to(device), labels.to(device)
                output = net(embedding_batch, mask, lengths)
                loss = criterion(output, labels)
                mean_loss += loss.item()
                acc = categorical_accuracy(output, labels)
                correct += acc.item()
        return mean_loss / len(loader), correct / len(loader)

    begin_epoch = checkpoint.epoch

    for epoch in range(begin_epoch, epochs+1):
        train_epoch()
        loss_train, acc_train = evaluate_epoch(train_loader, 'Train')
        loss_test, acc_test =  evaluate_epoch(val_loader, 'Val')

        print(f"Epoch {epoch}/{epochs}, Train Loss: {loss_train:.4e}, Test Loss: {loss_test:.4f}")
        print(f"Epoch {epoch}/{epochs}, Train Accuracy: {acc_train*100:.2f}%, Test Accuracy: {acc_test*100:.2f}%")
        if writer:
            writer.add_scalars("Loss", {"Train": loss_train, "Test" : loss_test}, epoch)
            writer.add_scalars("Accuracy", {"Train": acc_train*100, "Test" : acc_test*100}, epoch)
        checkpoint.epoch += 1
        if loss_test < min_loss:
            min_loss = loss_test
            best_acc = acc_test
            checkpoint.save('_best')
        checkpoint.save()

    print("\nFinished.")
    print(f"Best validation loss: {min_loss:.4e}")
    print(f"With accuracy: {best_acc}")
